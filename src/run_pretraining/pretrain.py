"""
  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""
import json
import os
import pickle

import numpy as np
import torch
import transformers
from torch.utils.data import ConcatDataset
from os.path import join
import pytorch_lightning as pl
from datetime import datetime
import argparse
from cycler import cycler
from tqdm import tqdm

from tevatron.modeling import rnn
from tevatron.pretrain import pretrain
import pandas as pd

PRETRAIN_DATA_PATH = "../resources/pretrain_data"


def parse(parser):
    parser.add_argument('--teacher', type=str, default="condenser", choices=["condenser", "coco"],
                        help='Teacher model.')
    parser.add_argument('--model', type=str, default="gru",
                        choices=["distilbert", "gru", "lstm", "rand-bert", "condenser", "coco"],
                        help='student model type.')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs.')
    parser.add_argument('--ff', type=int, default=0, help='number of ff layers.')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers.')
    parser.add_argument('--dataset', type=str, default="MARCO", choices=["MARCO"], help='dataset to use.')
    parser.add_argument('--dont_tie_embeddings', action='store_true', help='tie embeddings between models.')
    args = parser.parse_args()
    return args


def get_included_datasets(args):
    datasets = []
    datasets.append("train_queries")
    if args.teacher == "coco":  # coco files are saved with _pretrained suffix
        datasets = [i + "_coco" for i in datasets]
    return datasets


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def torch_to_memmap_format(teacher="condenser"):
    suffix = "" if teacher == "condenser" else "_coco"
    token_file = join(PRETRAIN_DATA_PATH, "train_queries_tokens.jsonl")
    vector_file = join(PRETRAIN_DATA_PATH, f"train_queries{suffix}.pt")
    np_path_prefix = join(PRETRAIN_DATA_PATH, f"train_queries{suffix}")
    if os.path.exists(vector_file):
        raise RuntimeError("Please create train queries embeddings")
    tokens = pd.read_json(token_file, lines=True)
    vectors = load_pickle(vector_file)
    assert vectors[1] == tokens.text_id.tolist()
    with open(np_path_prefix + ".desc", "w") as handle:
        json.dump({"length": len(tokens)}, handle)
    np_tokens = np.memmap(np_path_prefix + "_tokens.np", dtype=np.int32,
                          shape=(len(tokens), 16), mode="write")
    for i, _, t in tqdm(tokens.itertuples()):
        np_tokens[i, :min(len(t), 16)] = t[:16]
    np_vectors = np.memmap(np_path_prefix + "_vectors.np", dtype=np.float16,
                           shape=(len(tokens), 768), mode="write")
    np_vectors[:, :] = vectors[0]
    np_tokens.flush()
    np_vectors.flush()


if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser(description='Run distillation pretraining on a small model.')
    args = parse(parser)

    # load embeddings and tokenizer
    model = "Luyu/co-condenser-marco-retriever" if args.teacher == "coco" else "Luyu/co-condenser-marco"
    baseline_model = transformers.AutoModel.from_pretrained(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    print("loaded teacher model embeddings and tokenizer")

    # load dataset
    included_datasets = get_included_datasets(args)
    query_size = 16
    dataset = ConcatDataset([pretrain.NpDataset(join(PRETRAIN_DATA_PATH, desc + ".desc"), tokenizer, query_size)
                             for desc in included_datasets])
    dl = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)
    print("loaded dataset")
    # training args
    jp = cycler(train_data=["_".join([arg for arg in vars(args) if arg and type(getattr(args, arg)) == bool] +
                                     [args.teacher] + [args.dataset] +
                                     ["tie_" + str(not bool(getattr(args, "dont_tie_embeddings")))] +
                                     ["epochs", str(args.epochs)])])
    num_layers = cycler(num_layers=[args.num_layers])
    ff_layers = cycler(ff_layers=[args.ff])
    model_type = cycler(model_type=[args.model])
    tie = cycler(tie_embeddings=[not bool(getattr(args, "dont_tie_embeddings"))])
    target_dataset = cycler(target_dataset=[args.dataset])
    args_list = [rnn.RnnArgs(**x) for x in jp * num_layers * ff_layers * model_type * tie * target_dataset]
    print("args:", args_list)
    epochs = args.epochs
    small_model = pretrain.RnnPlModel(baseline_model, args_list, num_train_steps=len(dataset) // dl.batch_size * epochs)

    # train
    trainer = pl.Trainer(accumulate_grad_batches=1, max_epochs=epochs)
    trainer.fit(small_model, dl)

    # save
    model_dir_path = "../outputs/pretrained_models/"
    out_names = []
    for m in small_model.models:
        name = datetime.now().strftime("%y%m%d-%H%M%S-%f")
        m.save_pretrained(join(model_dir_path, name))
        out_names.append(name)
    print("done")
    print(" ".join(out_names))

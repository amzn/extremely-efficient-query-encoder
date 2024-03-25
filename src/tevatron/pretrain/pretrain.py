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
from typing import List
import numpy as np
import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertModel
from torch.utils.data import Dataset, ConcatDataset
from cycler import cycler
from datetime import datetime
from os.path import join
from tevatron.modeling import rnn
from tevatron.modeling.rnn import RnnArgs


class RnnPlModel(pl.LightningModule):
    def __init__(self, baseline_model, args_list: List[RnnArgs],
                 learning_rate=1e-4, lr_warmup_steps=1000, num_train_steps=1000000):
        super().__init__()
        baseline_model.embeddings.requires_grad_(False)
        self.models = nn.ModuleList([rnn.Rnn(baseline_model, args) for args in args_list])
        self.loss_func = nn.CosineSimilarity()
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.num_train_steps = num_train_steps

    def forward(self, batch):  # Not implemented
        return batch

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        losses = []
        for mm in self.models:
            out = mm(input_ids=batch["tokens"])
            loss = -self.loss_func(out.last_hidden_state[:, 0, :], batch["target"]).mean()
            losses.append(loss)
        if batch_idx % 1000 == 0:
            for i, l in enumerate(losses):
                self.log(f"l{i}", round(l.detach().cpu().item(), 4), prog_bar=True)
        return torch.stack(losses).sum()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.lr_warmup_steps, self.num_train_steps)
        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'reduce_on_plateau': False,
                    'monitor': 'val_loss',
                }
            ]
        )


class NpDataset(Dataset):
    """
    Represent dataset in memory mapped format to reduce DRAM consumption.
    """

    def __init__(self, desc_path, tokenizer, query_size):
        self.desc_path = desc_path
        self.tokenizer = tokenizer
        with open(desc_path) as handle:
            desc = json.load(handle)
        self.length = desc["length"]
        self.np_tokens = np.memmap(desc_path.replace(".desc", "_tokens.np"), dtype=np.int32,
                                   shape=(self.length, query_size), mode="readonly")
        self.np_vectors = np.memmap(desc_path.replace(".desc", "_vectors.np"), dtype=np.float16,
                                    shape=(self.length, 768), mode="readonly")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        tokens = torch.tensor(np.array(self.np_tokens[item]))
        target = torch.tensor(np.array(self.np_vectors[item]))
        return {"tokens": tokens, "target": target}

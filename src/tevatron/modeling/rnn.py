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
import dataclasses
import json
import os.path
from os.path import join, exists
import copy

import torch
import transformers
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from transformers import BertModel, TrainingArguments, BertLayer, AutoModel, BertForMaskedLM, BertConfig
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput

from tevatron.arguments import ModelArguments
from tevatron.data import QPCollator
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from torch import nn


@dataclasses.dataclass
class RnnArgs:
    model_type: str = "gru"
    hidden_size: int = 768
    num_layers: int = 1
    rnn_dropout: float = 0
    rnn_layernorm: bool = False
    ff_layers: int = 0
    train_data: str = "final_model"
    tie_embeddings: bool = True
    target_dataset: str = 'MARCO'

    @staticmethod
    def create_from_external_args(model_args: ModelArguments, train_args: TrainingArguments):
        return RnnArgs()


class FF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, layer_input):
        intermediate_output = self.intermediate(layer_input)
        layer_output = self.output(intermediate_output, layer_input)
        return layer_output


def get_distilbert(num_layers=6):
    mm: transformers.DistilBertModel = transformers.AutoModel.from_pretrained("distilbert-base-uncased",
                                                                              n_layers=num_layers)
    return mm.transformer


def generate_random_module(module):
    random_module = copy.deepcopy(module)
    for p in random_module.parameters():
        p.data = torch.rand_like(p.data)

    return random_module


class PureRnn(nn.Module):
    def __init__(self, args: RnnArgs):
        super().__init__()
        self.args = args
        h = args.hidden_size
        if args.model_type == "gru":
            self.recurring_model = nn.GRU(h, h, batch_first=True, num_layers=args.num_layers, dropout=args.rnn_dropout)
        elif args.model_type == "lstm":
            self.recurring_model = nn.LSTM(h, h, batch_first=True, num_layers=args.num_layers, dropout=args.rnn_dropout)
        elif args.model_type == "rnn":
            self.recurring_model = nn.RNN(h, h, batch_first=True, num_layers=args.num_layers, dropout=args.rnn_dropout)
        assert not args.rnn_layernorm, "not implemented"
        self.rnn_layer_norm = nn.LayerNorm(h) if args.rnn_layernorm else None
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        self.ff = nn.ModuleList([FF(config) for _ in range(args.ff_layers)])

    def forward(self, embeddings: PackedSequence):
        timestep_out, recurring_out = self.recurring_model(embeddings)
        if isinstance(recurring_out, tuple):  # LSTM has two recurring out, hidden out and cell state
            recurring_out = recurring_out[0]
        # if self.rnn_layer_norm is not None:
        #     recurring_out = self.rnn_layer_norm(recurring_out)
        rnn_out = recurring_out[-1, :, :]
        curr = rnn_out
        for ff_layer in self.ff:
            curr = ff_layer(curr)
        return curr


class Rnn(nn.Module):
    def __init__(self, model: BertModel, args: RnnArgs = None):
        super().__init__()
        self.args = args if args is not None else RnnArgs()
        if self.args.tie_embeddings:
            self.embedder = model.embeddings
        else:
            self.embedder = generate_random_module(model.embeddings)

        self.rnn = PureRnn(self.args)
        if self.args.model_type == "distilbert":
            self.transformer = AutoModel.from_pretrained("distilbert-base-uncased", n_layers=args.num_layers)
        elif self.args.model_type == 'condenser':
            if self.args.target_dataset == 'MARCO':
                self.transformer = AutoModel.from_pretrained("Luyu/co-condenser-marco", num_hidden_layers=args.num_layers)
            elif self.args.target_dataset == 'NQ':
                self.transformer = AutoModel.from_pretrained("Luyu/co-condenser-wiki", num_hidden_layers=args.num_layers)
        elif self.args.model_type == 'coco':
            self.transformer = AutoModel.from_pretrained("Luyu/co-condenser-marco-retriever", num_hidden_layers=args.num_layers)
        elif self.args.model_type == 'rand-bert':
            config = BertConfig(num_hidden_layers=args.num_layers)
            self.transformer = BertModel(config)

        if self.args.tie_embeddings and (self.args.model_type == "distilbert" or
                                         self.args.model_type == "condenser" or
                                         self.args.model_type == "rand-bert" or
                                         self.args.model_type == "coco"):
            self.transformer.embeddings = self.embedder

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=True):
        if self.args.model_type == "distilbert" or \
                self.args.model_type == "condenser" or\
                self.args.model_type == "rand-bert" or\
                self.args.model_type == "coco":
            if attention_mask is None:
                input_ids = torch.concat( # add [CLS] and [SEP] tokens.
                    (torch.ones(input_ids.size(0), 1, dtype=torch.long, device=input_ids.device) * 101,
                     torch.where((input_ids == 0).cumsum(1) == 1, 102, input_ids)), dim=1)
                attention_mask = input_ids != 0
            return self.transformer(input_ids, attention_mask)
        if attention_mask is not None:  # during tevatron training. input is coming from QPCollator.
            embeddings = self.embedder(input_ids[:, 1:-1])
            lengths = (attention_mask.sum(dim=1) - 2).cpu()
        else:  # during pretraining.
            embeddings = self.embedder(input_ids)
            lengths = (input_ids != 0).sum(dim=1).cpu()
        inpt = pack_padded_sequence(embeddings, lengths=lengths, batch_first=True, enforce_sorted=False)
        out = self.rnn(inpt)
        last_hidden_state = out.unsqueeze(1)
        return BaseModelOutputWithNoAttention(last_hidden_state, None)

    def save_pretrained(self, path):
        if self.args.model_type == "distilbert" or \
                self.args.model_type == "condenser" or \
                self.args.model_type == "rand-bert" or \
                self.args.model_type == "coco":
            self.transformer.save_pretrained(path)
            with open(join(path, "args_config.json"), "w") as handle:
                json.dump(self.args.__dict__, handle)
            return

        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.exists(join(path, "rnn_state.pt")):
            os.remove(join(path, "rnn_state.pt"))
        torch.save(self.rnn.state_dict(), join(path, "rnn_state.pt"))
        torch.save(self.embedder.state_dict(), join(path, "embeddings_state.pt"))
        with open(join(path, "config.json"), "w") as handle:
            json.dump(self.args.__dict__, handle)

    @staticmethod
    def load_pretrained(path, p_model):
        if exists(join(path, "args_config.json")):
            with open(join(path, "args_config.json"), "r") as handle:
                args = RnnArgs(**json.load(handle))
            model = Rnn(model=p_model, args=args)
            if args.model_type == "distilbert" or \
                    args.model_type == "condenser" or \
                    args.model_type == "rand-bert" or \
                    args.model_type == "coco":
                model.transformer = AutoModel.from_pretrained(path)
            return model
        with open(join(path, "config.json"), "r") as handle:
            args = json.load(handle)
        args = RnnArgs(**args)
        model = Rnn(model=p_model, args=args)
        model.rnn.load_state_dict(torch.load(join(path, "rnn_state.pt")))
        return model

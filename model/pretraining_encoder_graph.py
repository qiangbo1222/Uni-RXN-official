import copy
import heapq
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.optim as optim
from torch.autograd import Variable
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import graph_feat as ft
from . import transformer_encoder as te


class graph_encoder(nn.Module):
    def __init__(self, cfg, cross=False):
        super(graph_encoder, self).__init__()
        self.cfg = cfg
        self.cross = cross
        if not cross:
            self.node_embedder = ft.GraphNodeFeature(**cfg.node_embedder)
        self.attn_bias = ft.GraphAttnBias(**cfg.attn_bias)
        self.rnn = te.tf_encoder(cfg.hidden_size, cfg.num_heads, cfg.layers_n)
    
    def forward(self, input_graph, cross_node=None, return_bias=False):
        if not self.cross:
            node_feat = self.node_embedder(input_graph)
        else:
            node_feat = cross_node
        attn_bias = self.attn_bias(input_graph)

        mask = input_graph['mask']
        out = self.rnn(node_feat, mask, bias=attn_bias)
        if return_bias:
            return out, attn_bias
        return out
        #return out[:, 0, :]
    
class utter_encoder(nn.Module):
    def __init__(self, hidden_size, layers_n, num_heads, embedder=None):
        super(utter_encoder, self).__init__()

        self.hidden_size = hidden_size
        self.pos_embed = copy.deepcopy(embedder)
        self.layers_n = layers_n
        self.rnn = te.tf_encoder(hidden_size, num_heads, layers_n)

    def forward(self, inputs):
        if 0 in inputs.shape:
            return inputs
        mask = (inputs != 0)
        mask = mask.to(inputs.device)
        inputs = self.pos_embed(inputs)
        out = self.rnn(inputs, mask=mask)
        return out

class agg_encoder(nn.Module):
    def __init__(self, hidden_size, layers_n, num_heads):
        super(agg_encoder, self).__init__()

        self.hidden_size = hidden_size
        self.layers_n = layers_n
        self.rnn = te.tf_encoder(hidden_size, num_heads, layers_n)
    
    def forward(self, inputs, reactant_nums):
        #add aux token
        #inputs = torch.cat([torch.zeros(inputs.shape[0], 1, inputs.shape[2]).to(inputs.device), inputs], dim=1)
        mask = torch.zeros([inputs.shape[0], inputs.shape[1]]).to(inputs.device).bool()
        for i, r in enumerate(reactant_nums):
            mask[i, :r] = True
        out = self.rnn(inputs, mask=mask)
        out = out * mask.unsqueeze(-1)
        return torch.sum(out, dim=1)

class Pos_embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Pos_embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model
        #self.pos_emb = nn.Parameter(torch.zeros(1, 2048, d_model))
        self.pos = PositionalEncoding(d_model, 0.1)

    def forward(self, x):
        t = x.shape[-1]
        if x.max() > 486:
            print("got error emedding")
        return self.pos(self.lut(x) * math.sqrt(self.d_model))
        #return self.lut(x) + self.pos_emb[:, :t, :]

class MLP_head(nn.Module):
    def __init__(self, hidden_size, out_size, input_size=None, dropout=0.1, first_dropout=True):
        super(MLP_head, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if input_size is None:
            self.linear1 = nn.Linear(hidden_size, hidden_size)
        else:
            self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()
        self.first_dropout = first_dropout
    
    def forward(self, x):
        if self.first_dropout:
            x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
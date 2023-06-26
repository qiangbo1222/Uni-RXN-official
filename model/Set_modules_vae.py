import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from model.pretraining_encoder_graph import (MLP_head, Pos_embeddings,
                                             agg_encoder, graph_encoder,
                                             utter_encoder)
from model.set_generation import TopNGenerator
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def load_from_pl(model, checkpoint, key):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.startswith(key + '.')}
    pretrained_dict = {k.replace(key + '.', ''): v for k, v in pretrained_dict.items()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


class Path_encoder(nn.Module):
    def __init__(self, cfg):
        super(Path_encoder, self).__init__()
        self.react_graph_model = graph_encoder(cfg.graph_encoder)

        self.rnn = nn.LSTM(
            cfg.lstm.input_size,
            cfg.lstm.hidden_size,
            bidirectional=False,
            num_layers=cfg.lstm.num_layers,
            batch_first=True,
            dropout=cfg.lstm.dropout
        )

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        load_from_pl(self.react_graph_model, checkpoint, 'react_graph_model')
        self.react_graph_model.eval()

    
    def forward(self, graphs, context_lens):#only use the main reactant in each reaction
        batch_size = len(context_lens)
        graph_encoding = self.react_graph_model(graphs)[:, 0, :]
        device = graph_encoding.device
        #split according to context_lens
        graph_encoding = graph_encoding.split(context_lens, dim=0)
        #pad to max_len
        max_len = max(context_lens)
        graph_encoding_pad = torch.zeros(batch_size, max_len, graph_encoding[0].shape[-1]).to(device)
        for i, graph_encoding_i in enumerate(graph_encoding):
            graph_encoding_pad[i, :graph_encoding_i.shape[0], :] = graph_encoding_i
        graph_encoding = graph_encoding_pad

        tmp = torch.tensor(context_lens).to(graph_encoding.device)
        context_lens_sorted, indices = tmp.sort(descending=True)
        graph_encoding = graph_encoding.index_select(0, indices)
        hidden_size = graph_encoding.shape[-1]
        graph_encoding = pack_padded_sequence(
            graph_encoding, context_lens_sorted.data.tolist(), batch_first=True
        )
        init_hidden = torch.zeros(1, batch_size, hidden_size)
        init_hidden = init_hidden.to(device)
        init_hidden_c = torch.zeros(1, batch_size, hidden_size)
        init_hidden_c = init_hidden_c.to(device)
        _, h_n = self.rnn(graph_encoding, (init_hidden, init_hidden_c))

        _, inv_indices = indices.sort()
        h_n = h_n[0].index_select(1, inv_indices)

        enc = h_n.transpose(1, 0).contiguous().view(batch_size, -1)
        return enc
        
class Target_encoder(nn.Module):
    def __init__(self, cfg):
        super(Target_encoder, self).__init__()
        self.react_graph_model = graph_encoder(cfg.graph_encoder)
        pos_embed = Pos_embeddings(**cfg.pos_embed)
        self.reagent_model = utter_encoder(**cfg.utter_encoder, embedder=pos_embed)
        self.reagent_encoder = agg_encoder(**cfg.reagent_encoder)
    
    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint)

        load_from_pl(self.react_graph_model, checkpoint, 'react_graph_model')
        self.react_graph_model.eval()
        load_from_pl(self.reagent_model, checkpoint, 'reagent_model')
        self.reagent_model.eval()
        load_from_pl(self.reagent_encoder, checkpoint, 'reagent_encoder')
        self.reagent_encoder.eval()
    
    def forward(self, graphs, graph_num, reagents, reagent_num):

        with torch.no_grad():
            if sum(reagent_num) > 0:
                bs, reactant_num_max, reactant_len = reagents.shape
                utter = reagents.reshape(bs*reactant_num_max, -1)
                utter_encoding = self.reagent_model(utter)
                utter_encoding = utter_encoding.reshape(bs, reactant_num_max, reactant_len, -1)
                utter_encoding = utter_encoding[:, :, 0, :]
                reagent_encoding = self.reagent_encoder(utter_encoding, reagent_num)
                utter_encoding = [utter_encoding[i, :reagent_num[i],:] for i in range(bs)]
                reagent_encoding = [reagent_encoding[i] for i in range(bs)]
            else:
                utter_encoding = [None for _ in range(len(graph_num))]
                reagent_encoding = [None for _ in range(len(graph_num))]

            if sum(graph_num) > 0:
                bs = len(graph_num)
                graph_encoding = self.react_graph_model(graphs)
                graph_encoding = graph_encoding[:, 0, :]
                graph_encoding = graph_encoding.split(graph_num, dim=0)
            else:
                graph_encoding = [None for _ in range(len(graph_num))]

        target_encoding = []
        for i, (graph, utter) in enumerate(zip(graph_encoding, reagent_encoding)):
            if graph_num[i] != 0 and reagent_num[i] != 0:
                target_encoding.append(torch.sum(graph, dim=0) + utter)
            elif graph_num[i] != 0:
                target_encoding.append(torch.sum(graph, dim=0))
            elif reagent_num[i] != 0:
                target_encoding.append(utter)
            else:
                raise ValueError('graph_num and reagent_num cannot be both 0')
                
        target_encoding = torch.stack(target_encoding, dim=0).detach()

        #max_output_num = (torch.tensor(graph_num) + torch.tensor(reagent_num)).max().item()
        output = []
        for i in range(bs):
            if graph_encoding[i] is not None and utter_encoding[i] is not None:
                output.append(torch.cat([graph_encoding[i], utter_encoding[i]], dim=0).detach())
            elif graph_encoding[i] is not None:
                output.append(graph_encoding[i].detach())
            elif utter_encoding[i] is not None:
                output.append(utter_encoding[i].detach())

            
    
        return target_encoding, output
        
    
class Dis_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Dis_net, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), 
        nn.LeakyReLU(),
        nn.Linear(hidden_size, output_size))
    
    def forward(self, inputs):
        inputs = self.net(inputs)
        mu, var = torch.chunk(inputs, 2, 1)
        return mu, var


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld



def sample_gaussian(mu, logvar):
    epsilon = logvar.new_empty(logvar.size()).normal_()
    std = torch.exp(0.5 * logvar)
    z= mu + std * epsilon
    return z

class End_MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.hidden_size + cfg.z_size, cfg.hidden_size),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_size, cfg.hidden_size),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_size, 1),
                    nn.Sigmoid())
        self.net.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.1, 0.1)
            m.bias.data.fill_(0)
    
    def forward(self, x):
        return self.net(x)


import os
import pickle
import random
import time

import dataset.syn_dag_from_qb.chem_utils_for_reactant as chem_utils
import dataset.syn_dag_from_qb.parser_selfies as parser_selfies
import numpy as np
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import torch
import torch.nn.utils.rnn as rnn_utils
import tqdm
from data_utils.parser import mol_to_graph, preprocess_item
from easydict import EasyDict as edict
from torch.utils import data


class reaction_pretrain_dataset(data.Dataset):
    def __init__(self, reaction_dataset: str, vocab_loc: str):
        super(reaction_pretrain_dataset, self).__init__()
        with open(reaction_dataset, 'rb') as f:
            self.data = pickle.load(f)
        if isinstance(self.data, dict):
            self.data, self.labels = self.data['reactions'], self.data['label']
            self.ft = True
        else:
            self.ft = False

        with open(vocab_loc, 'rb') as f:
            self.vocab = pickle.load(f)
    
    def __getitem__(self, index):
        main_input = self.data[index][0][0]
        #main_input = chem_utils.can_smiles(main_input)
        '''
        main_reactants = self.data[index][0][1] + '.' + self.data[index][1]
        main_reactants = main_reactants.split('.')
        main_reactants = [parser_selfies.smiles_to_vec(chem_utils.can_smiles(smi), 
                                                       self.vocab['symbol_to_idx'], 
                                                       add_aux=True) for smi in main_reactants if len(smi)> 0]
        main_input = mol_to_graph(Chem.MolFromSmiles(main_input))
        main_input = preprocess_item(main_input)

        return main_input, main_reactants
        '''
        reactants = self.data[index][0][1]
        reagents = self.data[index][1]
        reagents = reagents.split('.')
        reagents = [chem_utils.can_smiles(smi) for smi in reagents if len(smi)> 0]
        
        reactants_change_idx = [r[1] for r in reactants]
        reactants = [chem_utils.can_smiles(r[0]) for r in reactants]
        reactants = [r for r in reactants if r is not None]
        reactants = [r for r in reactants if len(r) > 0]
        #cumsum for reactants_change_idx
        reactants_atom_num = [Chem.MolFromSmiles(r).GetNumAtoms() for r in reactants]
        reactants_change_idx = [flat_add(cum, sum(reactants_atom_num[:i])) for i, cum in enumerate(reactants_change_idx)]
        #flat the list
        reactants_change_idx = [item for sublist in reactants_change_idx for item in sublist]


        reactants = Chem.MolFromSmiles('.'.join(reactants))
        reagents = [parser_selfies.smiles_to_vec(smi, 
                                                self.vocab['symbol_to_idx'],
                                                add_aux=True) for smi in reagents]
        if len(reagents) == 0:
            reagents = [parser_selfies.smiles_to_vec('', self.vocab['symbol_to_idx'], add_aux=True)]

        prod = chem_utils.can_smiles(self.data[index][2])
        prod = mol_to_graph(Chem.MolFromSmiles(prod))

        change_idx = [main_input[1], reactants_change_idx]
        main_input = mol_to_graph(Chem.MolFromSmiles(main_input[0]))
        reactants = mol_to_graph(reactants)

        main_input, reactants, prod = preprocess_item(main_input), preprocess_item(reactants), preprocess_item(prod)
        if self.ft:
            return main_input, reactants, reagents, prod, change_idx, self.labels[index]
        else:
            return main_input, reactants, reagents, prod, change_idx

        

    def __len__(self):
        return len(self.data)

def forfor(a):
    return [item for sublist in a for item in sublist]

def flat_add(list1, add):
    return [l + add for l in list1]


def collate_fn(data):
    #batch.graph, batch.utter
    #graph: mask, x, in_degree, out_degree, attn_bias, spatial_pos, edge_input, attn_edge_type
    #utter:--
    for i in range(len(data)):
        data[i][0].idx = i
        data[i][1].idx = i
        data[i][3].idx = i

    batch_graph = []
    for data_id in [0, 1, 3]:
        graph_data =  graph_collator([item[data_id] for item in data])
        graph_data['mask'] = torch.zeros(graph_data['x'].size(0), graph_data['x'].size(1) + 1, dtype=torch.bool)# +1 for virtual node
        for ind in range(graph_data['x'].size(0)):
            graph_data['mask'][ind, :data[ind][data_id]['x'].shape[0] + 1] = True
        batch_graph.append(graph_data)
    
    change_idx_batch = [torch.zeros([len(data), batch_graph[0]['x'].size(1)], dtype=torch.long),
                        torch.zeros([len(data), batch_graph[1]['x'].size(1)], dtype=torch.long)]
    
    for i, change_idx in enumerate([item[4] for item in data]):
        for j, idx in enumerate(change_idx):
            change_idx_batch[j][i, idx] = 1

    
    bs = len(data)
    max_num_reagents = max([len(d[2]) for d in data])
    reagents = forfor([d[2] for d in data])
    max_len_reagents = max([len(r) for r in reagents])
    reagent_num = [len(d[2]) for d in data]

    data_reagents = torch.zeros([bs, max_num_reagents, max_len_reagents], dtype=torch.long)
    
    for i, d in enumerate(data):
        for j, r in enumerate(d[2]):
            data_reagents[i, j, :len(r)] = torch.tensor(r)

    if len(data[0]) == 6:
        return {'graph_input': batch_graph[0], 
                'graph_reactants': batch_graph[1],
                'graph_prod': batch_graph[2],
                'utter': data_reagents, 
                'reagent_num': reagent_num,
                'change_idx': change_idx_batch,
                'labels': torch.tensor([d[5] for d in data]).long()}
    else:
        return {'graph_input': batch_graph[0], 
                'graph_reactants': batch_graph[1],
                'graph_prod': batch_graph[2],
                'utter': data_reagents, 
                'reagent_num': reagent_num,
                'change_idx': change_idx_batch}

#borrow from graphormer code

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(-1e8)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def graph_collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
    )

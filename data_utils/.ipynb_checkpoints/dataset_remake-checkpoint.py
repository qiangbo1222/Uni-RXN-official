import os
import pickle
import random
import time

import psutil
import rdkit
import torch
import tqdm
from torch.autograd import Variable
from torch.utils import data

from data_utils.mol_to_feat import mol_feature_builder
from dataset.syn_dag_from_qb import chem_utils


class chains_dataset(data.Dataset):
    def __init__(self, chains_dataset_loc: str, name):
        """
        construct a dataset class from the chains file

        Args:
            chains_dataset_loc: the absolute path of the pickle file of chains(mol-mol-mol)
        """
        self.name = name

        with open(chains_dataset_loc[name]['input'], 'rb') as f:
            self.input_ = pickle.load(f)
        with open(chains_dataset_loc[name]['output'], 'rb') as f:
            self.output_ = pickle.load(f)
        #split train/ test set  and save it to disk 
        #add the if/else that determine whether to preprocess the dataset or read from the disk
    def __getitem__(self, index):
        feater = mol_feature_builder(self.input_[index])
        mol_graph = feater.graph_to_matrix()
        return mol_graph, torch.tensor(chem_utils.smiles_to_vec(self.output_[index]))

    def __len__(self):
        return len(self.input_)


class reaction_dataset(data.Dataset):
    def __init__(self, reaction_file_path: str or list, name):
        """
        construct a dataset class from the reactions file

        Args:
            chains_dataset_loc: the absolute path of the pickle file of chains(mol>reag>pro)
        """
        super(reaction_dataset, self).__init__()
        self.name = name
        with open(reaction_file_path[name]['input'], 'rb') as f:
            self._input = pickle.load(f)
        with open(reaction_file_path[name]['output'], 'rb') as f:
            self._output = pickle.load(f)

    def __getitem__(self, index, chained=False):
        '''
        input_size = len(self._input[index])
        output_size = len(self._output[index])

        max_length = 1028
 
        pad_space = max_length - output_size - 2
        output_ = torch.tensor([64, ] + self._output[index] + [65, ] + [66 for i in range(pad_space)]).to(self.device)

        pad_space = max_length - input_size
        input_ = torch.tensor([64, ] + self._input[index] + [65, ] + [66 for i in range(pad_space)]).to(self.device)
        '''

        return torch.tensor(chem_utils.smiles_to_vec(self._input[index])), torch.tensor(chem_utils.smiles_to_vec(self._output[index]))

    def __len__(self):
        return len(self._input)

# partly borrowed from https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).to(vec.device)], dim=dim)

def pad_matrix(mat, pad):
    pad_size = list(mat.shape)
    pad_size[0], pad_size[1] = pad, pad
    mat_pad = torch.zeros(*pad_size)
    if len(pad_size) == 2:
        mat_pad[:mat.shape[0], :mat.shape[1]] = mat
    elif len(pad_size) == 3:
        mat_pad[:mat.shape[0], :mat.shape[1], :] = mat
    return mat_pad
        
def pad_edge_path(edge_vec, spa_encode, path_pad, vec_pad):
    attn_size = len(edge_vec)
    edge_embed = torch.zeros([vec_pad, vec_pad, int(path_pad), 2])
    for i in range(1, attn_size):
        for j in range(i):
            edge_embed[i, j, :int(spa_encode[i, j]) - 1, :] = torch.stack(
                edge_vec[i][j]
            ).squeeze(0).squeeze(0)
    return edge_embed

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        if torch.is_tensor(batch[0][0]):
            # find longest sequence
            max_len_x = max(map(lambda x: x[0].shape[self.dim], batch))
            max_len_y = max(map(lambda x: x[1].shape[self.dim], batch))
            # pad according to max_len
            batch_pad = [(pad_tensor(x[0], pad=max_len_x, dim=self.dim), pad_tensor(x[1], pad=max_len_y, dim=self.dim)) for x in batch]
            # stack all
            xs = torch.stack([i[0] for i in batch_pad], dim=0)
            ys = torch.stack([i[1] for i in batch_pad], dim=0)
            return xs.long(), ys.long()
        elif isinstance(batch[0][0], tuple):
            
            # make padding for the smiles sequence
            max_len_x = max(map(lambda x: x[1].shape[self.dim], batch))
            batch_pad = [pad_tensor(x[1], pad=max_len_x, dim=self.dim) for x in batch]
            seq = torch.stack(batch_pad, dim=0)
            # make padding for the node vector
            max_len_x = max(map(lambda x: x[0][0].shape[self.dim], batch))
            batch_pad = [pad_tensor(x[0][0], pad=max_len_x, dim=self.dim) for x in batch]
            node_vec = torch.stack(batch_pad, dim=0)
            # make padding for spatial encoding matrix
            max_len_mat = max(map(lambda x: x[0][1].shape[0], batch))
            batch_pad = [pad_matrix(x[0][1], max_len_mat) for x in batch]
            spatial_encode = torch.stack(batch_pad, dim=0)
            # make padding for the edge feature matrix
            edge_vec = [x[0][2] for x in batch]
            max_path_length = spatial_encode.max()
            batch_pad = [pad_edge_path(edge_v, spatial_enc, max_path_length, max_len_x) for edge_v, spatial_enc in zip(edge_vec, spatial_encode)]
            edge_encode = torch.stack(batch_pad, dim=0)
            return [node_vec.long(), spatial_encode.long(), edge_encode.long()], seq.long(), seq.long()

    def __call__(self, batch):
        return self.pad_collate(batch)



import pickle
import random

import dataset.chem_utils_for_reactant as chem_utils
import dataset.parser_selfies as parser_selfies
import numpy as np
import rdkit
import rdkit.Chem as Chem
import torch
from data_utils.parser import mol_to_graph, preprocess_item
from rdkit.Chem import QED
from torch.utils import data


class chains_dataset(data.Dataset):
    """Dataset for reaction paths; used for training the Uni-Rxn generative model."""
    def __init__(self, reaction_dataset: str, vocab_loc: str):
        """
        args:
            reaction_dataset: path to the reaction dataset
            vocab_loc: path to the vocabulary for SELFIES parsing
        """
        super(chains_dataset, self).__init__()
        with open(reaction_dataset, 'rb') as f:
            self.data = pickle.load(f)
        
        with open(vocab_loc, 'rb') as f:
            self.vocab = pickle.load(f)
    
    def __getitem__(self, index):
        """
        returns:
            path_data: list of the reaction path (list of graphs)
            reactants_data: list of target reactants (list of graphs)
            reagents_data: list of target reagents (list of parsed SELFIES)
            is_end: whether the reaction path is complete
            property_dic: dictionary of the chemical properties of the final product
            check_smiles: list of the target reactants and reagentes (list of SMILES), used for masking the same target representation in contrastive learning
        """
        whole_chain, final_prod = self.data[index]['chain'], self.data[index]['final_prod']
        cut_num = random.randint(0, len(whole_chain))
        if cut_num == 0:
            path = [None]
            target = [whole_chain[0][0], '']
        
        else:
            chain = whole_chain[:cut_num]
            path = [r[0] for r in chain]
            target = chain[-1][1]
            path = [chem_utils.can_smiles(smi) for smi in path]
            path = [Chem.MolFromSmiles(smi) for smi in path]


        property_dic = self.data[index]['property']

        reactants, reagents = target[0], target[1]
        reactants = [chem_utils.can_smiles(smi) for smi in reactants.split('.') if len(smi)> 0]
        reagents = [chem_utils.can_smiles(smi) for smi in reagents.split('.') if len(smi)> 0]
        reactants = [Chem.MolFromSmiles(smi) for smi in reactants]

        reagents_data = [parser_selfies.smiles_to_vec(smi,
                                                self.vocab['symbol_to_idx'],
                                                add_aux=True) for smi in reagents]
        
       

        path_data = [preprocess_item(mol_to_graph(mol)) for mol in path]
        reactants_data = [preprocess_item(mol_to_graph(mol)) for mol in reactants]
        
        check_smiles = [Chem.MolToSmiles(demap(mol)) for mol in reactants] + reagents

        
        return path_data, reactants_data, reagents_data, cut_num==len(whole_chain), property_dic, check_smiles

        

    def __len__(self):
        return len(self.data)


def forfor(a):
    return [item for sublist in a for item in sublist]

def flat_add(list1, add):
    return [l + add for l in list1]


def collate_fn(data):
    """
    #function for collating the data
    """
    path_i, reactant_i = 0, 0
    for i, d in enumerate(data):
        for j, g_piece in enumerate(d[0]):
            d[0][j].idx = path_i
            path_i += 1
        for j, g_piece in enumerate(d[1]):
            d[1][j].idx = reactant_i
            reactant_i += 1
    
    path_lens = [len(d[0]) for d in data]
    target_graph_lens = [len(d[1]) for d in data]
    target_reagent_lens = [len(d[2]) for d in data]
    target_lens = [len(d[1]) + len(d[2]) for d in data]

    path = forfor([d[0] for d in data])
    target_graphs = forfor([d[1] for d in data])
    

    path_data = forfor([item[0] for item in data])
    graph_data_path =  graph_collator(path_data)
    graph_data_path['mask'] = torch.zeros(graph_data_path['x'].size(0), graph_data_path['x'].size(1) + 1, dtype=torch.bool)# +1 for virtual node
    for ind in range(graph_data_path['x'].size(0)):
        graph_data_path['mask'][ind, :path_data[ind]['x'].shape[0] + 1] = True
    
    react_data = forfor([item[1] for item in data])
    if len(react_data) > 0:
        graph_data_react =  graph_collator(react_data)
        graph_data_react['mask'] = torch.zeros(graph_data_react['x'].size(0), graph_data_react['x'].size(1) + 1, dtype=torch.bool)# +1 for virtual node
        for ind in range(graph_data_react['x'].size(0)):
            graph_data_react['mask'][ind, :react_data[ind]['x'].shape[0] + 1] = True
    else:
        graph_data_react = None
    
    bs = len(data)
    max_num_reagents = max([len(d[2]) for d in data])
    reagents = forfor([d[2] for d in data])
    if len(reagents) > 0:
        max_len_reagents = max([len(r) for r in reagents])

        data_reagents = torch.zeros([bs, max_num_reagents, max_len_reagents], dtype=torch.long)
    
        for i, d in enumerate(data):
            for j, r in enumerate(d[2]):
                data_reagents[i, j, :len(r)] = torch.tensor(r)
    else:
        data_reagents = None
    
    end_flag = torch.tensor([d[3] for d in data], dtype=torch.bool)

    full_property_dic = {k: torch.tensor([float(data[i][5][k]) for i in range(len(data))]) for k in data[0][5].keys()}
    check_smiles = [d[6] for d in data]
    #flatten the list of smiles
    check_smiles = forfor(check_smiles)
    #mask of size len(check_smiles) * len(check_smiles); 1 if the two smiles are the same
    mask = torch.zeros(len(check_smiles), len(check_smiles), dtype=torch.bool)
    for i in range(len(check_smiles)):
        for j in range(len(check_smiles)):
            if check_smiles[i] == check_smiles[j]:
                mask[i, j] = True

    return graph_data_path, path_lens, graph_data_react, target_graph_lens, data_reagents, target_reagent_lens, target_lens, end_flag, full_property_dic, mask


def demap(mol):
    """
    remove the atom mapping in rdkit mol object
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


#the following graph processing code are borrow from graphormer code (https://github.com/Microsoft/Graphormer)

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

import easydict as edict
import numpy as np
import pyximport
import rdkit
import torch
from rdkit import Chem

pyximport.install(setup_args={"include_dirs": np.get_include()})
from data_utils import algos

"""
This file contains the functions to convert rdkit mol to graph
"""


def atom_to_fp(atom):
    """
    convert rdkit atom to feature vector
    including atom type, aromatic, ring, formal charge, hybridization
    shape: 5
    """
    ATOM_LIST = [1, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 24, 25, 26, 29, 30, 34, 35, 46, 53]
    HYBRIDIZATION_LIST = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
    fp = [0] * ((len(ATOM_LIST) + 1) + 1 + 1 + 1 + 1)
    fp = np.array(fp)
    atom_index = atom.GetAtomicNum()
    if atom_index in ATOM_LIST:
        fp[ATOM_LIST.index(atom.GetAtomicNum())] = 1
    else:
        fp[len(ATOM_LIST)] = 1

    if atom.GetIsAromatic():
        fp[len(ATOM_LIST) + 1] = 1
    if atom.IsInRing():
        fp[len(ATOM_LIST) + 2] = 1
    fp[len(ATOM_LIST) + 3] = atom.GetFormalCharge()
    if atom.GetHybridization() in HYBRIDIZATION_LIST:
        fp[len(ATOM_LIST) + 4] = HYBRIDIZATION_LIST.index(atom.GetHybridization())
    else:
        fp[len(ATOM_LIST) + 4] = len(HYBRIDIZATION_LIST)
    fp_flat = [0,0,0,0,0]
    fp_flat[0] = list(fp[:len(ATOM_LIST) + 1]).index(1)
    fp_flat[1:] = fp[len(ATOM_LIST) + 1:]
    return fp_flat

def bond_to_fp(bond):
    """
    convert rdkit bond to feature vector
    including bond type, ring
    shape: 2
    """
    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    fp = [0] * (len(BOND_LIST) + 1)
    fp = np.array(fp)
    bond_index = bond.GetBondType()
    
    fp[BOND_LIST.index(bond.GetBondType())] = 1
    if bond.IsInRing():
        fp[len(BOND_LIST)] = 1
    fp_flat = [0,0]
    fp_flat[0] = list(fp[:len(BOND_LIST)]).index(1)
    fp_flat[1] = fp[len(BOND_LIST)]
    
    return fp_flat

    

def mol_to_graph(mol):
    """
    convert rdkit mol to graph
    """
    if mol is None:
        # return empty graph if mol is None
        return edict.EasyDict(x=torch.zeros([0, 5], dtype=torch.long), edge_index=torch.zeros([2, 0], dtype=torch.long), edge_attr=torch.zeros([0, 2], dtype=torch.long))
    else:
        node_feature = []
        for atom in mol.GetAtoms():
            fp = torch.tensor(atom_to_fp(atom), dtype=torch.long)
            node_feature.append(fp)
        if len(node_feature) == 0:
            node_feature = torch.zeros([0,5], dtype=torch.long)
        else:
            node_feature = torch.stack(node_feature, dim=0)
        
        bond_feature = []
        edge_index = []
        for bond in mol.GetBonds():
            fp = torch.tensor(bond_to_fp(bond), dtype=torch.float)
            bond_feature.append(fp)
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_feature.append(fp)
            edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

        if len(bond_feature) == 0:
            bond_feature = torch.zeros((0, 2), dtype=torch.long)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            bond_feature = torch.stack(bond_feature, dim=0).long()
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        assert bond_feature.size(0) == edge_index.size(1)
        return edict.EasyDict(x=node_feature, edge_index=edge_index, edge_attr=bond_feature)



def convert_to_single_emb(x, offset: int = 32):
    if len(x.size()) > 1:
        feature_num = x.size(1)
    else:
        feature_num = 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )
    if x.size(0) == 0:
        attn_bias = torch.zeros([N, N], dtype=torch.float)
        spatial_pos = torch.zeros([N, N], dtype=torch.long)
        edge_input = np.zeros([0, N, N, edge_attr.size(-1)])
    else:
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    return item

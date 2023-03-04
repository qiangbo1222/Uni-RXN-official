import gc
import os
import pickle
import time
import typing as t

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

__all__ = ["mol_feature_builder"]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    #return np.array([x==i for i in allowable_set], dtype=float)
    return allowable_set.index(x)


def atom_featurizer(atom: Chem.Atom):
    # convert a rdkit atom into an list of features
    Atom_idx = atom.GetIdx()
    Atom_symbol = atom.GetSymbol()
    Atom_degree = len(atom.GetBonds())
    Atom_charge = atom.GetFormalCharge() + 4
    Atom_radelect = atom.GetNumRadicalElectrons()
    Atom_aromatic = atom.GetIsAromatic()
    Atom_hybridiz = atom.GetHybridization()
    Atom_H = atom.GetTotalNumHs()

    Atom_symbol = one_of_k_encoding_unk(
        Atom_symbol, ['C', 'N', 'O', 'P', 'S', 'B', 'F', 'Cl', 'Br', 'I']
    )
    Atom_hybridiz = one_of_k_encoding_unk(
        Atom_hybridiz,
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ],
    )
    '''
    return Atom_idx, torch.cat(
        [
            torch.tensor([feat, ])
            for feat in [
                Atom_symbol,
                np.array([Atom_degree, ]),
                np.array([Atom_charge, ]),
                np.array([Atom_radelect, ]),
                np.array([Atom_aromatic, ]),
                Atom_hybridiz,
                np.array([Atom_H, ]),
            ]
        ]
        ,dim = 1
    ).squeeze(0)
    '''
    return Atom_idx, torch.cat(
        [
            torch.tensor([feat, ])
            for feat in [
                Atom_symbol,
                Atom_degree,
                Atom_charge,
                Atom_radelect,
                Atom_aromatic,
                Atom_hybridiz,
                Atom_H,
            ]
        ]
    )

def bond_featurizer(bond: Chem.Bond):
    # convert a rdkit bond into an list of features
    #bond_type = [0, 0, 0, 0]
    dictionary_of_bond = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }
    #bond_type[dictionary_of_bond[bond.GetBondType()]] = 1 
    bond_type = [dictionary_of_bond[bond.GetBondType()], ]
    bond_isRing = [bond.IsInRing(), ]

    bond_idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]

    return bond_idx, (torch.tensor(bond_type + bond_isRing) + 1).long()


class mol_feature_builder:
    def __init__(self, smiles: str):
        """
        A molecule feature(array, graph) builder for DAG Transformer

        Args:
            smiles(str):
                canonical smiles is optional
            mol_role(str):
                one of ['root', 'reangant', 'reactant', 'intermediate']
        """
        self.smiles = smiles
        try:
            self.mol = Chem.MolFromSmiles(self.smiles)
        except:
            print("RDkit is not able to process this smiles.")

    def mol_to_graph(self, package):
        """
        convert the molecule into a dgl / networkx graph
        """
        bond_matrix_start = [
            bond_featurizer(bond)[0][0] for bond in self.mol.GetBonds()
        ]
        bond_matrix_end = [bond_featurizer(bond)[0][1] for bond in self.mol.GetBonds()]
        if package == 'dgl':
            mol_graph = dgl.graph((bond_matrix_start, bond_matrix_end))

            mol_graph.ndata["atom_feat"] = torch.stack([
                atom_featurizer(atom)[1] for atom in self.mol.GetAtoms()]
            )
            mol_graph = dgl.to_bidirected(mol_graph, copy_ndata=True)
            '''
            mol_graph.edata["bond_feat"] = torch.cat(
                [
                    torch.stack([bond_featurizer(bond)[1] for bond in self.mol.GetBonds()])
                    for i in range(2)
                ]
            )
            '''
            return mol_graph

        if package == 'networkx':
            mol_graph = nx.Graph()
            bond_feats = [bond_featurizer(bond)[1] for bond in self.mol.GetBonds()]
        
            for i, bond in enumerate(zip(bond_matrix_start, bond_matrix_end)):
                mol_graph.add_edges_from([tuple(bond), ], bond_feat=bond_feats[i])

            return mol_graph


        

    def graph_to_matrix(self):
        """
        convert a molecule graph into the [1] node feature vector [2] node degree vector [3] spatial encoding (if connected xij=shortest path else 0) [4] egde encoding (collect and sum/average the edge feature embedding along shortest path)
        [4] due to memory crash we chage the encoding into only get those edge feature which forms bonds
        """
        graph_dgl = self.mol_to_graph(package='dgl')
        graph_nx = self.mol_to_graph(package='networkx')
        #[1]
        node_vector = graph_dgl.ndata['atom_feat']
        #[2]
        #degree_vector = torch.tensor([it[1] for it in list(graph_nx.degree)])
        #[3]
        
        atom_num = node_vector.shape[0]
        spa_encode = torch.zeros([atom_num, atom_num])
        shortest_path_length_dic = dict(nx.shortest_path_length(graph_nx))
        for i in range(atom_num):
            for j in range(atom_num):
                spa_encode[i, j] = shortest_path_length_dic[i][j]
        #[4]
        
        shortest_path_dic = nx.shortest_path(graph_nx)
        #edge_encode = torch.zeros([atom_num, atom_num, 5])
        edge_encode = [[[ ] for i in range(atom_num)] for j in range(atom_num)]
        for i in range(atom_num):
            for j in range(i + 1):
                along_bonds  = shortest_path_dic[i][j]
                for k in range(len(along_bonds) - 1):
                    edge_encode[i][j].append(graph_nx.get_edge_data(along_bonds[k], along_bonds[k + 1])['bond_feat'])
        '''
        edge_encode = torch.zeros([atom_num, atom_num, 2])
        for i in range(atom_num):
            for j in range(atom_num):
                if graph_nx.has_edge(i, j):
                    edge_encode[i][j] = graph_nx.get_edge_data(i, j)['bond_feat']
        '''
        del graph_dgl
        del graph_nx
        gc.collect()
        return (node_vector + 1).int(), (spa_encode + 1).int(), edge_encode

'''
#testing code
mol_feater = mol_feature_builder('ClCC(C(CC)=O)C(O1)CNCC1C(C(F)(F)F)C2=CC=CC=C2')
start_time = time.time()
print(mol_feater.graph_to_matrix()[0].shape)
#print(mol_feater.graph_to_matrix()[2].shape)
print(time.time() - start_time)
#print(mol_feater.graph_to_matrix()[2])
'''


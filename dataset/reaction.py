import itertools
import typing
from os import path

import numpy as np
import pandas as pd
import tqdm
from rdkit import Chem
from torch.utils import data

"""
This file include the dataset class for reaction dataset.
"""


class Reactions(data.Dataset):
    def __init__(self, reaction_file_path:list or str, chained=False, cut=None):
        """
        args:
            reaction_file_path: path to the reaction dataset (csv or txt)
            chained: whether to split reactants according to the reaction center and atom mapping
            cut: exclude reactions with more than cut tokens
        """
        super(Reactions, self).__init__()

        if isinstance(reaction_file_path, str):
            reaction_file_path = [reaction_file_path, ]
        

        reactions_raw = []
        if reaction_file_path[0].endswith('.csv'):
            for file in reaction_file_path:
                with open(file, 'rb') as f:
                    for line_counter, line in enumerate(tqdm.tqdm(f.readlines(), desc='reading reaction file')):
                        try:
                            if line_counter >= 3:
                                reactions_raw.append(str(line).split('\\t')[3].split(' ')[0])
                        except:
                            pass
            self.label = None
        
        elif reaction_file_path[0].endswith('.txt'):
            for file in reaction_file_path:
                with open(file, 'rb') as f:
                    for line_counter, line in enumerate(tqdm.tqdm(f.readlines(), desc='reading reaction file')):
                        try:
                            if line_counter >= 1:
                                reactions_raw.append(str(line).split(' ')[0][2:])
                        except:
                            pass
            self.label = None
            
        
        elif reaction_file_path[0].endswith('.tsv'):
            file = reaction_file_path[0]
            df = pd.read_csv(file, sep='\t')
            print(df.columns)
            reactions_raw = df['mapped_rxn'].tolist()
            self.label = df['labels'].tolist()
            
        print(f'{len(reactions_raw)} reactions is readout')

        self.reactions_split = []
        
        for reaction in tqdm.tqdm(reactions_raw, desc='spliting main reactant'):
            if chained:
                try:
                    reactants, reagants, products = reaction.split('>')
                except:
                    continue
                reactants_smiles = reactants.split('.')
                reactants = [Chem.MolFromSmarts(reactant) for reactant in reactants_smiles]
                map_list = []
                map_list_len = []
                product_atom_map = set([atom.GetAtomMapNum() for atom in Chem.MolFromSmiles(products).GetAtoms()])
                for qmol in reactants:
                    mapping = [atom.GetAtomMapNum() for atom in qmol.GetAtoms() if atom.GetAtomMapNum() and atom.GetAtomMapNum() in product_atom_map]
                    map_list.append(mapping)
                    map_list_len.append(len(mapping))
                
                #here we pack rdkit mol directly to file
                max_ind = map_list_len.index(max(map_list_len))
                sub_ind = [i for i in range(len(map_list_len)) if i != max_ind and map_list_len[i] > 0]
                main_reactant = reactants[max_ind]
                sub_reactants = [r for i, r in enumerate(reactants) if i in sub_ind]
                reagents = reagants.split('.') + [Chem.MolToSmiles(r) for i, r in enumerate(reactants) if i != max_ind and len(map_list[i]) == 0]
                reagents = '.'.join([r for r in reagents if r != ''])
                products = Chem.MolFromSmarts(products)
                #check all atom_change for reactants
                atom_change = []
                for reactant in [main_reactant, ] + sub_reactants:
                    atom_change.append(get_changed_atom(reactant, products))
                
                main_reactant = [main_reactant, atom_change[0]]
                sub_reactants = [[r, atom_change[i+1]] for i, r in enumerate(sub_reactants)]
                max_atom = max([mol.GetNumAtoms() for mol in [main_reactant[0], ] + [r[0] for r in sub_reactants] + [products, ]])
                max_atom = max([max_atom, len(reagents)/3])
                if max_atom <= cut:
                    #save str to save disk memory
                    main_reactant = [reactants_smiles[max_ind], main_reactant[1]]
                    sub_reactants = [[reactants_smiles[j], sub_reactants[i][1]] for i, j in enumerate(sub_ind)]
                    self.reactions_split.append([[main_reactant, sub_reactants], reagents, Chem.MolToSmiles(products), reaction])


            else:
                reactants, reagants, products = reaction.split('>')
                reactants = reactants.split('.')
                self.reactions_split.append([reactants, reagants, products, reaction])
        

    def __getitem__(self, index, chained=False):
        return self.reactions_split[index]

    def __len__(self):
        return len(self.reactions_split)



def count_C(mol):
    return sum([1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6])

def get_changed_atom(mol1, mol2):
    """
    get the index of changed atom 
        mol1: reactant to check
        mol2: product
    """
    atom_idx = []
    for atom1 in mol1.GetAtoms():
        for atom2 in mol2.GetAtoms():
            if atom1.GetAtomMapNum() == atom2.GetAtomMapNum():
                if not check_atom_env(atom1, atom2):
                    atom_idx.append(atom1.GetIdx())
                break
    return atom_idx




def check_atom_env(atom1, atom2):
    #get neighbor atoms
    atom1_neighbors = [atom for atom in atom1.GetNeighbors()]
    atom2_neighbors = [atom for atom in atom2.GetNeighbors()]
    #get bond type
    atom1_bond = [bond for bond in atom1.GetBonds()]
    atom2_bond = [bond for bond in atom2.GetBonds()]
    if set_equal(atom1_neighbors, atom2_neighbors, atom_equal) and set_equal(atom1_bond, atom2_bond, bond_equal):
        return True
    else:
        return False
    

def atom_equal(atom1, atom2):
    if atom1.GetSymbol() == atom2.GetSymbol() and atom1.GetAtomMapNum() == atom2.GetAtomMapNum():
        return True
    else:
        return False

def bond_equal(bond1, bond2):
    if bond1.GetBondType() == bond2.GetBondType():
        return True
    else:
        return False

def set_equal(set1, set2, equal_func):
    if len(set1) != len(set2):
        return False
    else:
        equal_check = 0
        for item1 in set1:
            for item2 in set2:
                if equal_func(item1, item2):
                    equal_check += 1
                    break
        if equal_check == len(set1):
            return True
        else:
            return False
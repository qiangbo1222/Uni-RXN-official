import os
import pickle
import random
import sys
from os.path import join

import faiss
import numpy as np
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from rxnmapper import RXNMapper


def clean_for_model(path):
    return [p if not isinstance(p, list) else p[0] for p in path]

def calc_carbon(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum([a.GetAtomicNum() == 6 for a in mol.GetAtoms()])

def remove_map(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)

def clean_mapper(main_r, gen_r, product, rxn_mapper):
    rxn = Chem.MolToSmiles(main_r) + '.' + '.'.join(gen_r) + '>>' + Chem.MolToSmiles(product)
    if len(rxn) > 512:
        return False#too long for mapping model
    mapped_rxn = rxn_mapper.get_attention_guided_atom_maps([rxn])[0]['mapped_rxn']
    mapped_reactant = [r for r in mapped_rxn.split('>>')[0].split('.') if r != '']
    mapped_num = [len(r.split(':')) - 1 for r in mapped_reactant]
    main_react = mapped_reactant[mapped_num.index(max(mapped_num))]
    if remove_map(main_react) != remove_map(Chem.MolToSmiles(main_r)):
        return False
    else:
        return True


def sample_prob(probs, sample_num):
    probs = probs / np.sum(probs)
    sample = np.random.choice(len(probs), sample_num, p=probs)
    if sample_num == 1:
        return int(sample)
    else:
        return sample
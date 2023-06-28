import argparse
import os
import pickle

import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import tqdm
from rxnmapper import RXNMapper

rxn_mapper = RXNMapper()

def make_dummy_data(rxn, need_map=True):
    if len(rxn) > 512:
        return None
    if need_map:
        rxn = rxn_mapper.get_attention_guided_atom_maps([rxn])[0]['mapped_rxn']
    line1, line2, line3 = rxn.split('>')
    line1 = line1.split('.')
    line2 = line2.split('.')
    reactant = line1 + line2
    product = line3
    reagent = [r for r in reactant if ':' not in r]
    reactant = [r for r in reactant if ':' in r]
    map_num = [len(r.split(':')) - 1 for r in reactant]
    main_reactant = reactant[map_num.index(max(map_num))]
    sub_reactant = [r for r in reactant if r != main_reactant]
    #pad to dummies
    sub_reactant = [r for r in sub_reactant if r != '']
    main_reactant = [main_reactant, []]
    sub_reactant = [[r, []] for r in sub_reactant]
    reagent = '.'.join(reagent)
    return [[main_reactant, sub_reactant], reagent, product, rxn]

# Run this script with the input file as the first argument
# python prepare_rxn_for_feat.py example_rxn.txt

argparser = argparse.ArgumentParser()
argparser.add_argument('--input_file', type=str, help='Input file containing reactions rxn')
argparser.add_argument('--output_file', type=str, default='prepared_rxn_data.pkl', help='Output file containing prepared rxn data')
argparser.add_argument('--need_map', type=bool, default=True, help='Whether the rxn needs atom mapping, recommended to be True')
args = argparser.parse_args()

with open(args.input_file, 'r') as f:
    rxns = f.readlines()

rxns = [r.strip() for r in rxns]
dumm_data = []
for rxn in tqdm.tqdm(rxns):
    dummy_data = make_dummy_data(rxn, need_map=args.need_map)
    if dummy_data is not None:
        dumm_data.append(dummy_data)

with open(os.path.join(args.output_file, 'dummy_data.pkl'), 'rb') as f:
    dummy_data = pickle.load(f)


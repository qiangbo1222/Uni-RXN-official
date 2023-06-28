import argparse
import sys

sys.path.append('./')
import os
import pickle
import sys
from os.path import join

import numpy as np
import rdkit
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import tqdm
import yaml
from easydict import EasyDict as edict
from train_module.CATH_vae import CATH
from yaml import Dumper, Loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_dir', type=str, default='ckpt/unirxn_gen.ckpt', help='path to the generative model checkpoint')
argparser.add_argument('--input_file', type=str, default='dataset/data/react_lib_smi.pkl',
                      help='a dictionary file containing your reactant + reagent library, molecules are represented using smiles')
argparser.add_argument('--config_path', type=str, default='config/')

args = argparser.parse_args()

cfg = edict({
    'model':
    yaml.load(open(join(args.config_path, 'model/cath_vae.yaml')),
              Loader=Loader),
    'dataset':
    yaml.load(open(join(args.config_path, 'dataset/chains.yaml')),
              Loader=Loader)
})


model = CATH.load_from_checkpoint(args.model_dir, cfg=cfg, stage='inferencing')
model = model.to(device)
model.eval()

all_smi = pickle.load(open(args.input_file, 'rb'))
reactant_smi = all_smi['reactant']
reagent_smi = all_smi['reagent']
ouput_lib = []

for d in tqdm.tqdm(reagent_smi):
    mol_rep = model.generate_target_rep(d, 'reagent', device)[0].detach().cpu().numpy()
    ouput_lib.append((d, mol_rep))

for d in tqdm.tqdm(reactant_smi):
    mol_rep = model.generate_target_rep(d, 'reactant', device)[0].detach().cpu().numpy()
    ouput_lib.append((d, mol_rep))

output_dir = args.input_file.split('.')[0] + '_rep.pkl'

with open(output_dir, 'wb') as f:
    pickle.dump(ouput_lib, f)

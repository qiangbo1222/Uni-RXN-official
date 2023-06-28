import argparse
import os
import sys

sys.path.append('.')
sys.path.append('./LocalTransform')

import pickle
import random
import sys
from os.path import join

import numpy as np
import torch
import tqdm
import yaml
from easydict import EasyDict as edict
from train_module.Pretrain_Graph import Pretrain_Graph
from yaml import Dumper, Loader

device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
#device = torch.device('cpu')



argparser = argparse.ArgumentParser()
argparser.add_argument('--model_dir', type=str, default='ckpt/uni_rxn_base.ckpt', help='path to the pretrained base model checkpoint')
argparser.add_argument('--input_file', type=str, help='path to the input file for featurization')
argparser.add_argument('--config_path', type=str, default='config/')

args = argparser.parse_args()

cfg = edict({
    'model':
    yaml.load(open(join(args.config_path, 'model/pretrain_graph.yaml')),
              Loader=Loader),
    'dataset':
    yaml.load(open(join(args.config_path, 'dataset/pretrain.yaml')),
              Loader=Loader),
})

model = Pretrain_Graph(cfg, stage='inference')
model = model.load_from_checkpoint(args.model_dir)
model = model.to(device)
model.eval()


for input_f in [args.input_file]:
    data = pickle.load(open(input_f, 'rb'))
    collect = []
    output_f = input_f.split('.')[0] + '_unirxnfp.pkl'
    for idx in tqdm.tqdm(range(len(data))):
        
        fp = model.generate_reaction_fp_mix(data[idx], device, no_reagent=False)#set to True if you want to ignore reagents
        fp = fp.cpu().numpy()
        collect.append(fp)
    pickle.dump(collect, open(output_f, 'wb'))

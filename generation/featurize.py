import argparse
import os
import sys

sys.path.append('..')
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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')



argparser = argparse.ArgumentParser()
argparser.add_argument('--model_dir', type=str, default='/root/jupyter/DAG_Transformer/DAG_Transformer/pl_logs/pretrain_graph/version_52/checkpoints/epoch=65-step=411048.ckpt')
argparser.add_argument('--input_file_train', type=str, default='/root/jupyter/DAG_Transformer/eval_forpaper/ranking/permute_data/permute_dataorg_reaction_data_train.pkl')
argparser.add_argument('--input_file_test', type=str, default='/root/jupyter/DAG_Transformer/eval_forpaper/ranking/permute_data/permute_dataorg_reaction_data_test.pkl')
argparser.add_argument('--config_path', type=str, default='/root/jupyter/DAG_Transformer/DAG_Transformer/config/')

args = argparser.parse_args()

cfg = edict({
    'model':
    yaml.load(open(join(args.config_path, 'model/pretrain_graph.yaml')),
              Loader=Loader),
    'dataset':
    yaml.load(open(join(args.config_path, 'dataset/pretrain.yaml')),
              Loader=Loader),
    'optim':
    yaml.load(open(join(args.config_path, 'optim/adam.yaml')), Loader=Loader),
    'scheduler':
    yaml.load(open(join(args.config_path, 'scheduler/step.yaml')),
              Loader=Loader),
    'trainer':
    yaml.load(open(join(args.config_path, 'trainer/default.yaml')),
              Loader=Loader)
})

model = Pretrain_Graph(cfg, stage='inference')
model = model.load_from_checkpoint(args.model_dir)
model = model.to(device)
model.eval()

#input_files = ['/root/jupyter/DAG_Transformer/eval_forpaper/ranking/permute_data/permute_dataorg_reaction_data_test.pkl',
#               '/root/jupyter/DAG_Transformer/eval_forpaper/ranking/permute_data/permute_dataorg_reaction_data_train.pkl',
#               '/root/jupyter/DAG_Transformer/eval_forpaper/ranking/permute_data_full/permute_dataorg_reaction_data_test.pkl',
#               '/root/jupyter/DAG_Transformer/eval_forpaper/ranking/permute_data_full/permute_dataorg_reaction_data_train.pkl']
#input_files = ['/root/jupyter/DAG_Transformer/eval_forpaper/class_1k/data/test_set_5k/org_reaction_data_train.pkl',
#               '/root/jupyter/DAG_Transformer/eval_forpaper/class_1k/data/test_set_5k/org_reaction_data_test.pkl']
#input_files = ['/root/jupyter/DAG_Transformer/eval_forpaper/dis_high_low_yield/data_preprocess/train_data.pkl',
#               '/root/jupyter/DAG_Transformer/eval_forpaper/dis_high_low_yield/data_preprocess/val_data.pkl',
#               '/root/jupyter/DAG_Transformer/eval_forpaper/dis_high_low_yield/data_preprocess/test_data.pkl']
input_files = ['/root/jupyter/DAG_Transformer/eval_forpaper/compare_predictor_based/process_data/dingos_reactions.pklel']


#for input_f in input_files:
#yield_data_path = '/root/jupyter/DAG_Transformer/eval_forpaper/yield/data/preprocess'
#yield_p = os.listdir(yield_data_path)
#yield_p.sort()
#for input_f in yield_p:
#    if not input_f.endswith("yield.pkl"):
#        input_f = os.path.join(yield_data_path, input_f)

#        data = pickle.load(open(input_f, 'rb'))
#        output_f = input_f.split('.')[0] + '_our_graphfp_yield.pkl'
#        
#        collect = []
'''
        for i in tqdm.tqdm(range(len(data))):
            if isinstance(data[i][1], list):
                data[i][1] = '.'.join(data[i][1])
            fp = model.generate_reaction_fp_mix(data[i], device).detach().cpu().numpy()
            collect.append(fp)
        pickle.dump(collect, open(output_f, 'wb'))

    '''
for input_f in input_files:
#input_f = '/root/jupyter/DAG_Transformer/DAG_Transformer/dataset/syn_dag_from_qb/data/full_ustpo_yield_data.pkl'
    data = pickle.load(open(input_f, 'rb'))
    collect = []
    output_f = input_f.split('.')[0] + '_our_graphfp_yield.pkl'
    for idx in tqdm.tqdm(range(len(data))):
        
        fp = model.generate_reaction_fp_mix(data[idx], device, no_reagent=True)
        #fp_tuple = [fp_tuple[0].cpu().numpy(), fp_tuple[1].cpu().numpy()]
        fp = fp.cpu().numpy()
        #label = data['labels'][idx]
        #collect.append((fp, label))
        collect.append(fp)
    pickle.dump(collect, open(output_f, 'wb'))

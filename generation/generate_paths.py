import argparse
import sys

sys.path.append('..')
sys.path.append('../predictor')
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
import tqdm
import yaml
from easydict import EasyDict as edict
from reaction_predictor import *
from rxnmapper import RXNMapper

rxn_mapper = RXNMapper()
from torch.nn import functional as F
from train_module.CATH_vae import CATH
from yaml import Dumper, Loader

#select cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_dir', type=str, default='../pl_logs/set_gen_vae/epoch=358-step=123855.ckpt')
argparser.add_argument('--input_file', type=str, default='hyw_jnk3.smi')
argparser.add_argument('--react_lib_file', type=str, default='../dataset/syn_dag_from_qb/data/react_lib_smi_rep.pkl')
argparser.add_argument('--sample_mode', type=str, default='fix')
argparser.add_argument('--sample_len', type=int, default=1)
argparser.add_argument('--sample_num', type=int, default=50000)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--config_path', type=str, default='../config')
argparser.add_argument('--output_dir', type=str, default='/root/jupyter/DAG_Transformer/wet_lab/jnk3')

args = argparser.parse_args()

cfg = edict({
    'model':
    yaml.load(open(join(args.config_path, 'model/cath_vae.yaml')),
                Loader=Loader),
    'dataset':
    yaml.load(open(join(args.config_path, 'dataset/chains.yaml')),
                Loader=Loader)
})

predict_model = OpenNMTServerPredictor()
def predict_product(smis):
    smi_sets = [set(smi.split('.')) for smi in smis]
    return [product[0] for product in predict_model._run_list_of_reactant_sets(smi_sets)]

model = CATH(cfg)
checkpoint = torch.load(args.model_dir, map_location=device)['state_dict']
#for key in list(checkpoint):
#    checkpoint[key.replace('model.', '')] = checkpoint.pop(key)
model.load_state_dict(dict(checkpoint))
model.to(device)
#model.device = device
#model.eval()


react_lib = pickle.load(open(args.react_lib_file, 'rb'))
reactant_smi = [d[0] for d in react_lib]
reactant_rep = [d[1][0] for d in react_lib]
rep_shape = reactant_rep[0].shape

res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatIP(rep_shape[0])
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

reactant_rep = np.array(np.stack(reactant_rep))
#normalize
reactant_rep = reactant_rep / np.linalg.norm(reactant_rep, axis=1, keepdims=True)
gpu_index_flat.add(reactant_rep)

#get sdf from input
if args.input_file.endswith('.sdf'):
    suppl = Chem.SDMolSupplier(args.input_file)
    input_mols = [x for x in suppl if x is not None]
    #input_mols = [m for m in input_mols if Descriptors.MolWt(m) < 500]
else:
    input_mols = [Chem.MolFromSmiles(smi) for smi in open(args.input_file, 'r').readlines()]

#get the test set for debug
#mols = pickle.load(open('../dataset/syn_dag_from_qb/data/reaction_graph/chaines_dataset.pkl', 'rb'))
#input_mols = [Chem.MolFromSmiles(m[0][0]) for m in mols[:100]]

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

def clean_mapper(main_r, gen_r, product):
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

def generate_deravatives_batch(paths):
    paths_input = [clean_for_model(path) for path in paths]
    generate_rep, end_results = model.generate(paths_input, device)
    generate_reactant = [[] for _ in range(len(paths))]
    for rep_id, reps in enumerate(generate_rep):
        for rep in reps:
            rep = F.normalize(rep.unsqueeze(0), dim=1).cpu().numpy().astype('float32')
            D, I = gpu_index_flat.search(rep, 8)
            sample_d = sample_prob(D[0], 1)
            generate_reactant[rep_id].append(reactant_smi[I[0][sample_d]])
    #predict product
    generate_product = predict_product(['.'.join(r) + '.' + Chem.MolToSmiles(paths[i][-1]) for i, r in enumerate(generate_reactant)])
    generate_product = [p if p != 1 and p not in generate_reactant[i] else None for i, p in enumerate(generate_product)]
    generate_product = [Chem.MolFromSmiles(p) if p is not None and calc_carbon(p) else None for p in generate_product]
    paths = [p + [generate_reactant[i], generate_product[i]] for i, p in enumerate(paths)]
    paths = [p if p[-1] is not None else None for p in paths]
    end_results = [e > 0.5 for e in end_results]
    return paths, end_results

results = []
if args.sample_mode == 'fix':
    for _ in range(args.sample_num):
        paths = [[m] for m in input_mols]
        for i in range(args.sample_len):
            for path_id in tqdm.tqdm(range(0, len(paths), args.batch_size), desc='running fix step generation'):
                paths[path_id:path_id+args.batch_size], _ = generate_deravatives_batch(paths[path_id:path_id+args.batch_size])
                #clean unsuccesful paths
            paths = [p for p in paths if p is not None]
            for p in paths:
                if clean_mapper(p[0], p[1], p[-1]):
                    results.append(p)
    
elif args.sample_mode == 'adaptive':
    for _ in range(args.sample_num):
        paths_queue = set([m for m in input_mols])
        paths_queue = [[m] for m in paths_queue]
        paths = []
        while len(paths_queue) > 0:
            paths_batch = paths_queue[:args.batch_size]
            paths_queue = paths_queue[args.batch_size:]
            paths_batch, end_results = generate_deravatives_batch(paths_batch)
            for i, e in enumerate(end_results):
                if e and paths_batch[i] is not None:
                    paths.append(paths_batch[i])
                elif paths_batch[i] is not None:
                    paths_queue.append(paths_batch[i])
        results.extend(paths)

with open(os.path.join(args.output_dir, f'results_{args.sample_mode}_{args.sample_len}'), 'wb') as f:
    pickle.dump(results, f)
            
print(f'finished {args.sample_mode} {args.sample_len} generation output to {args.output_dir}')
print(f'generated {len(results)} paths')
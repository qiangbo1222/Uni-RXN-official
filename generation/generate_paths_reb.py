import argparse
import sys

sys.path.append('.')
sys.path.append('./LocalTransform')

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
from rxnmapper import RXNMapper

rxn_mapper = RXNMapper()
from Synthesis import init_LocalTransform, predict_product_batch
from torch.nn import functional as F
from train_module.CATH_vae import CATH
from utils import *
from yaml import Dumper, Loader

#select cuda device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_dir', type=str, default='ckpt/uni_rxn_gen.ckpt', help='path to the generative model checkpoint')
argparser.add_argument('--input_file', type=str, help='input seed molecules path (sdf or raw smiles)')
argparser.add_argument('--react_lib_file', type=str, default='dataset/data/react_lib_smi_rep.pkl', help='pre-encoded reactants+reagents fingerprints path')
argparser.add_argument('--sample_mode', type=str, default='fix', help='choose from <fix | adaptive>, determine whether let the model predict when to stop generation')
argparser.add_argument('--sample_len', type=int, default=1, help='need to be set for fix step generation, generate x steps for each seed structure')
argparser.add_argument('--sample_num', type=int, default=500, help='number of molecules generated for each seed structure')
argparser.add_argument('--batch_size', type=int, default=64, help='batch size for Uni-RXN generative model')
argparser.add_argument('--config_path', type=str, default='config', help='generative model config and predictor model config')
argparser.add_argument('--output_dir', type=str, default='data/samples', help='output directory for generated paths')

args = argparser.parse_args()


cfg = edict({
    'model':
    yaml.load(open(join(args.config_path, 'model/cath_vae.yaml')),
                Loader=Loader),
    'dataset':
    yaml.load(open(join(args.config_path, 'dataset/chains.yaml')),
                Loader=Loader),
    'predictor':
    yaml.load(open(join(args.config_path, 'predictor/model.yaml')),
                Loader=Loader),
})


#prepare a reaction predictor model
predictor_model, graph_functions, template_dicts, template_infos = init_LocalTransform(cfg.predictor)
def predict_product_wrap(smis):
    products = []
    for i in range(0, len(smis), args.batch_size // 2):
        reactants_str = smis[i:i + args.batch_size // 2]
        try:#need to throw out a batch if predict failed for template mismatch
            results_df = predict_product_batch(cfg.predictor, reactants_str, predictor_model, graph_functions, template_dicts, template_infos, verbose = False, sep = False)
        except:
            results_df = [None] * len(reactants_str)
        for idx in range(len(results_df)):
            if results_df[idx] is not None:
                products.append(results_df[idx])
            else:
                products.append(reactants_str[idx].split('.')[0])
    return products


#prepare Uni-RXN generative model
model = CATH.load_from_checkpoint(args.model_dir, cfg=cfg, stage='inferencing')
model.to(device)

#prepare the pre-encoded reactants library
react_lib = pickle.load(open(args.react_lib_file, 'rb'))
reactant_smi = [d[0] for d in react_lib]
reactant_rep = [d[1][0] for d in react_lib]
rep_shape = reactant_rep[0].shape
print(f"Loading a reactants library from ZINC, size: {len(reactant_rep)}")

res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatIP(rep_shape[0])
index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

reactant_rep = np.array(np.stack(reactant_rep))
reactant_rep = reactant_rep / np.linalg.norm(reactant_rep, axis=1, keepdims=True)
index_flat.add(reactant_rep)




def generate_deravatives_batch(paths):
    paths_input = [clean_for_model(path) for path in paths]
    generate_rep, end_results = model.generate(paths_input, device)
    generate_reactant = [[] for _ in range(len(paths))]

    for rep_id, reps in enumerate(generate_rep):
        for rep in reps:
            rep = F.normalize(rep.unsqueeze(0), dim=1).cpu().numpy().astype('float32')
            D, I = index_flat.search(rep, 8)
            sample_d = sample_prob(D[0], 1)
            generate_reactant[rep_id].append(reactant_smi[I[0][sample_d]])
    #predict product
    generate_product = predict_product_wrap([Chem.MolToSmiles(paths[i][-1]) + '.' + '.'.join(r) for i, r in enumerate(generate_reactant)])
    generate_product = [p if p != 1 and p not in generate_reactant[i] else None for i, p in enumerate(generate_product)]
    generate_product = [Chem.MolFromSmiles(p) if p is not None and calc_carbon(p) else None for p in generate_product]
    paths = [p + [generate_reactant[i], generate_product[i]] for i, p in enumerate(paths)]
    paths = [p if p[-1] is not None else None for p in paths]
    end_results = [e > 0.5 for e in end_results]
    return paths, end_results


if __name__ == '__main__':
    #get sdf files from input
    if args.input_file.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(args.input_file)
        input_mols = [x for x in suppl if x is not None]
    else:
        input_mols = [Chem.MolFromSmiles(smi) for smi in open(args.input_file, 'r').readlines()]

    results = []

    #generate molecule derivatives with fixed steps of virtual reactions (recommand within 3 steps)
    if args.sample_mode == 'fix':
        for _ in range(args.sample_num):
            paths = [[m] for m in input_mols]
            for i in range(args.sample_len):
                for path_id in tqdm.tqdm(range(0, len(paths), args.batch_size), desc='running fix step generation'):
                    paths[path_id:path_id+args.batch_size], _ = generate_deravatives_batch(paths[path_id:path_id+args.batch_size])
                    #clean unsuccesful paths
                paths = [p for p in paths if p is not None]
                for p in paths:
                    if clean_mapper(p[0], p[1], p[-1], rxn_mapper):
                        results.append(p)


    #generate molecule derivatives where the number of steps are predicted by the model
    elif args.sample_mode == 'adaptive':
        for _ in range(args.sample_num):
            paths_queue = set([m for m in input_mols])
            paths_queue = [[m] for m in paths_queue]
            paths = []
            while len(paths_queue) > 0:
                paths_batch = paths_queue[:args.batch_size]
                paths_queue = paths_queue[args.batch_size:]
                if len(paths_batch) < args.batch_size:#fix bug for batch norm layer
                    cut_lengths = len(paths_batch)
                    paths_batch.extend(paths_batch * (args.batch_size - ( len(paths_batch) // len(paths_batch) ) ))
                else:
                    cut_lengths = args.batch_size
                paths_batch, end_results = generate_deravatives_batch(paths_batch)
                paths_batch = paths_batch[:cut_lengths]
                end_results = end_results[:cut_lengths]
                for i, e in enumerate(end_results):
                    if e and paths_batch[i] is not None:
                        paths.append(paths_batch[i])
                    elif paths_batch[i] is not None:
                        paths_queue.append(paths_batch[i])
            results.extend(paths)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f'results_{args.sample_mode}_{args.sample_len}.pkl'), 'wb') as f:
        pickle.dump(results, f)
                
    print(f'finished {args.sample_mode} {args.sample_len} generation output to {args.output_dir}')
    print(f'generate {len(results)} paths')

    results_prod = [p[-1] for p in results]
    results_smiles = [Chem.MolToSmiles(p) for p in results_prod]
    results_smiles = list(set(results_smiles))
    writer = Chem.SDWriter(os.path.join(args.output_dir, f'results_{args.sample_mode}_{args.sample_len}.sdf'))
    for smi in results_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            writer.write(mol)
    writer.close()

    print(f'generate unique smiles: {len(results_smiles)}')
import pickle

import numpy as np
import rdkit
import rdkit.Chem as Chem
import selfies as sf
import tqdm
from dataset.chem_utils_for_reactant import can_smiles

"""
This file is used to preprocess the dataset into vocabulary for selfies and function for encoding smiles into selfies.
"""

def sf_encoder(smi):
    #selfies cannot encode isomer for double bond
    smi = smi.replace('\\', '')
    smi = smi.replace('/', '')
    smi = smi.replace('*', '')
    smi = smi.replace('[SnH10+6]', '')
    if smi == '':
        return ''
    else:
        return sf.encoder(can_smiles(smi), strict=False)


def preprocess_into_vocab(dataset_for_vocab):
    """
    args:
        dataset_for_vocab: path to the pretrained preprocessed dataset
    """
    with open(dataset_for_vocab, 'rb') as f:
        dataset = pickle.load(f)

    collect_data_flat = []
    blank_list = ['', '.']
    for reaction in tqdm.tqdm(dataset):
        if reaction[0][0] not in blank_list:
            collect_data_flat.append(sf_encoder(reaction[0][0]))#reactants main
        for r in reaction[0][1].split('.'):
            if r not in blank_list:
                collect_data_flat.append(sf_encoder(r))
        for r in reaction[1].split('.'):
            if r not in blank_list:
                collect_data_flat.append(sf_encoder(r))#reagents
        if reaction[2] not in blank_list:
            collect_data_flat.append(sf_encoder(reaction[2]))#product


    alphabet = sf.get_alphabet_from_selfies(collect_data_flat)
    alphabet = list(sorted(alphabet))
    alphabet = ['[nop]', '[bos]', '[eos]'] + alphabet
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    return symbol_to_idx, len(alphabet)

def replace_unfind(sef, vocab):
    sef_refine = []
    for s in sef.split('[')[1:]:
        if '[' + s not in vocab:
            sef_refine.append('[nop]')
        else:
            sef_refine.append('[' + s)
    return ''.join(sef_refine)

def smiles_to_vec(smi, symbol_to_idx, add_aux=True):
    """
    convert smiles to selfies and then to parsed vector
    args:
        smi: smiles string or list of smiles string
        symbol_to_idx: vocabulary for selfies
        add_aux: whether to add [bos] and [eos] to the selfies
    """
    if isinstance(smi, str):
        sef = sf_encoder(smi)
        if add_aux:
            sef = '[bos]' + sef + '[eos]'
        sef = replace_unfind(sef, symbol_to_idx.keys())
        return np.array(sf.selfies_to_encoding(selfies=sef, vocab_stoi=symbol_to_idx, enc_type='label'))
    elif isinstance(smi, list):
        sef = [sf_encoder(s) for s in smi]
        if add_aux:
            sef = ['[bos]' + s + '[eos]' for s in sef]
        pad_to_len = max([sf.len_selfies(s) for s in sef])
        sef_encoding = [sf.selfies_to_encoding(
            selfies=s, vocab_stoi=symbol_to_idx, pad_to_len=pad_to_len, enc_type='label'
        ) for s in sef]
        return np.array(sef_encoding)

def vec_to_smiles(vec, idx_to_symbol, symbol_to_idx):
    """
    convert parsed vector to selfies and then to smiles
    args:
        vec: parsed vector or list of parsed vector
        idx_to_symbol: vocabulary for selfies
        symbol_to_idx: vocabulary for selfies (reverse)
    """

    aux_token = [symbol_to_idx['[bos]'], symbol_to_idx['[eos]']]
    if len(vec.shape) == 1:
        vec = vec[vec!=aux_token[0]]
        vec = vec[vec!=aux_token[1]]
        selfies = sf.encoding_to_selfies(vec, idx_to_symbol, enc_type='label')
        smiles = sf.decoder(selfies)
        return smiles
    else:
        vec = np.stack([v[(v!=aux_token[0]) & (v!=aux_token[1])] for v in vec] ,axis=0)
        selfies = [
            sf.encoding_to_selfies(v, idx_to_symbol, enc_type='label')\
            for v in vec
        ]
        smiles = [sf.decoder(s) for s in selfies]
        return smiles



# no need to run yourself we provide the vocab already
# if you wish to use your own dataset, you can run this file to generate the vocab
# specify "dataset_for_vocab" and "save_dir" in the following code to run this script


'''
if __name__ == '__main__':
    symbol_to_idx, len_ = preprocess_into_vocab(dataset_for_vocab)
    idx_to_symbol = {value:key for key,value in symbol_to_idx.items()}
    with open(save_dir, 'wb') as f:
        pickle.dump({'symbol_to_idx': symbol_to_idx, 'idx_to_symbol': idx_to_symbol, 'size': len_}, f)
        print(f'save a vocab of size {len_}')#478
'''
import os
import pickle

import reaction
from rdkit import Chem

"""
create the pretrain dataset for base Uni-Rxn model
1. first download the raw USPTO MIT dataset from (https://github.com/wengong-jin/nips17-rexgen/blob/master/USPTO/data.zip)
2. unzip and put the data folder under the data/raw folder
3. run this script to create the pretrain dataset pickle file
"""

name = ["train", "valid", "test"]
dataset_path = [
    f"data/raw/{i}.txt"
    for i in name
]

reaction_dataset_pretrain = reaction.Reactions(dataset_path, chained=True, cut=1000)
with open('data/pretrain_reaction_dataset.pkl', 'wb') as f:
    pickle.dump(reaction_dataset_pretrain.reactions_split, f)



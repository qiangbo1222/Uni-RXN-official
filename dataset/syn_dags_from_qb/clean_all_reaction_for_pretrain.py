import os
import pickle

from rdkit import Chem

import reaction

'''
name = ["train", "valid"]
dataset_path = [
    f"/root/jupyter/DAG_Transformer/DAG_Transformer/dataset/raw/from_ref/Jin_USPTO_1product_{i}.txt"
    for i in name
]

reaction_dataset_pretrain = reaction.Reactions(dataset_path, chained=True, cut=100)
print(len(reaction_dataset_pretrain.reactions_split))
with open('/root/jupyter/DAG_Transformer/DAG_Transformer/dataset/syn_dag_from_qb/data/pretrain_reaction_dataset_Jin_notest.pkl', 'wb') as f:
    pickle.dump(reaction_dataset_pretrain.reactions_split, f)


'''
name = [ "test", "train_valid"]
dataset_path = [
    f"/root/jupyter/DAG_Transformer/eval/classification/data/uspto_1k_TPL_{i}.tsv"
    for i in name
]

for i in range(2):
    reaction_dataset_pretrain = reaction.Reactions(dataset_path[i], chained=True, cut=1e5)
    print(len(reaction_dataset_pretrain.reactions_split))
    with open(f'/root/jupyter/DAG_Transformer/eval_forpaper/class_1k/data/preprocess/ustpo_1k_TPL_org_{name[i]}.pkl', 'wb') as f:
        pickle.dump({'reactions': reaction_dataset_pretrain.reactions_split, 'label': reaction_dataset_pretrain.label}, f)
'''

dataset_path = [
    '/root/jupyter/DAG_Transformer/eval/classification/data/schneider50k.tsv'
]
label_set = ['8.1.5', '6.1.3', '6.2.2', '1.3.7', '3.4.1', '1.7.6', '1.7.7', '6.1.5', '1.7.9', '1.2.1', '2.6.3', '7.1.1', 
            '3.1.5', '3.1.1', '9.3.1', '2.1.7', '2.2.3', '5.1.1', '1.6.4', '7.9.2', '10.4.2', '10.1.5', '2.6.1', '9.1.6', 
            '7.2.1', '2.7.2', '6.2.1', '8.2.1', '1.8.5', '1.6.2', '6.2.3', '1.7.4', '6.3.7', '7.3.1', '6.3.1', '10.1.1', 
            '2.3.1', '3.3.1', '1.3.6', '2.1.1', '1.2.5', '3.1.6', '1.2.4', '1.3.8', '10.2.1', '10.1.2', '6.1.1', '8.1.4', '2.1.2', '1.6.8']


for i in range(1):
    reaction_dataset_pretrain = reaction.Reactions(dataset_path[i], chained=True, cut=1e5)
    print(len(reaction_dataset_pretrain.reactions_split))
    with open(f'/root/jupyter/DAG_Transformer/eval_forpaper/class_1k/data/test_set_5k/org_reaction_data.pkl', 'wb') as f:
        pickle.dump({'reactions': reaction_dataset_pretrain.reactions_split, 'label': reaction_dataset_pretrain.label, 'split': reaction_dataset_pretrain.split}, f)
    reaction_train_index = [ind for ind, s in enumerate(reaction_dataset_pretrain.split) if s == 'train']
    reaction_test_index = [ind for ind, s in enumerate(reaction_dataset_pretrain.split) if s == 'test']
    with open(f'/root/jupyter/DAG_Transformer/eval_forpaper/class_1k/data/test_set_5k/org_reaction_data_train.pkl', 'wb') as f:
        pickle.dump({'reactions': [reaction_dataset_pretrain.reactions_split[i] for i in reaction_test_index], 'label': [label_set.index(reaction_dataset_pretrain.label[i]) for i in reaction_test_index]}, f)
    with open(f'/root/jupyter/DAG_Transformer/eval_forpaper/class_1k/data/test_set_5k/org_reaction_data_test.pkl', 'wb') as f:
        pickle.dump({'reactions': [reaction_dataset_pretrain.reactions_split[i] for i in reaction_train_index], 'label': [label_set.index(reaction_dataset_pretrain.label[i]) for i in reaction_train_index]}, f)
'''
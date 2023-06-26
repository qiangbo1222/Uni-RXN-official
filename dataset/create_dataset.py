import os
import pickle

import mega_graph
import networkx as nx
import reaction
from rdkit import Chem

name = ["train", "valid", "test"]
dataset_path = [
    f"dataset/raw/Jin_USPTO_1product_{i}.txt"
    for i in name
]

def revert_assemble_chain(chain):
    chain = chain[1:]
    chain.reverse()
    #packed into pairs
    chain = [(chain[i], chain[i+1][:-1]) for i in range(0, len(chain), 2)]
    return chain

def clean_empty(chain):
    for i, react in enumerate(chain):
        if len(react[1][0]) + len(react[1][1]) == 0:
            return False
    return True


def preprocessing(_dataset_path, save_path):
    """
    create the reaction paths dataset for the generative model
    args:
        _dataset_path: path to the raw USPTO MIT dataset
        save_path: path to save the reaction graph and the reaction paths dataset
    """    
    chained_graph = mega_graph.create_graph_chained(reaction.Reactions(_dataset_path, chained=True, cut=200))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    nx.write_gpickle(chained_graph, os.path.join(save_path, 'chained_graph.gpickle'))

    chains_dataset = mega_graph.extract_chain_from_graph(chained_graph, 7, save_path, QED_hold=0.5)
    chains_dataset = [revert_assemble_chain(chain) for chain in chains_dataset]
    chains_dataset = [chain for chain in chains_dataset if clean_empty(chain)]
    with open(os.path.join(save_path, "chains_dataset.pkl"), "wb") as f:
        pickle.dump(chains_dataset, f)
    print(f"chains_dataset: {len(chains_dataset)} chains") 


if __name__ == "__main__":
    preprocessing(dataset_path, "dataset/data/reaction_graph")
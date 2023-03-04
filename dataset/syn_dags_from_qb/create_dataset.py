import os
import pickle

import networkx as nx
#import chem_utils
from rdkit import Chem

import mega_graph
import reaction

name = ["train", "valid", "test"]
dataset_path = [
    f"/root/jupyter/DAG_Transformer/DAG_Transformer/dataset/raw/from_ref/Jin_USPTO_1product_{i}.txt"
    for i in name
]
commercial_path = "/root/jupyter/DAG_Transformer/dataset/raw/WuXi Building blocks list SDF.sdf"

building_block_degree = 10

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

def preprocessing(_dataset_path, _commercial_path, save_path, _building_block_degree=25):
    
    #chained_graph = mega_graph.create_graph_chained(reaction.Reactions(_dataset_path, chained=True, cut=200))
    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)
    #nx.write_gpickle(chained_graph, os.path.join(save_path, 'chained_graph.gpickle'))
    chained_graph = nx.read_gpickle(os.path.join(save_path, 'chained_graph.gpickle'))

    chains_dataset = mega_graph.extract_chain_from_graph(chained_graph, 7, save_path, QED_hold=0.5)
    chains_dataset = [revert_assemble_chain(chain) for chain in chains_dataset]
    chains_dataset = [chain for chain in chains_dataset if clean_empty(chain)]
    with open(os.path.join(save_path, "chaines_dataset.pkl"), "wb") as f:
        pickle.dump(chains_dataset, f)
    print(f"chains_dataset: {len(chains_dataset)} chains") 



    #free graph preprocess    
    '''
    free_graph = mega_graph.create_graph_free(reaction.Reactions(_dataset_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "free_graph.pkl"), "wb") as f:
        pickle.dump(free_graph, f)
    
    with open(os.path.join(save_path, "free_graph.pkl"), "rb") as f:
        free_graph = pickle.load(f)
    #suppl = Chem.SDMolSupplier(_commercial_path)
    #kangde_BB = set([Chem.MolToSmiles(mol) for mol in suppl if mol])
    degree_dict = dict(free_graph.out_degree())
    degree_list = sorted(degree_dict.items(), key=lambda item:item[1], reverse=True)
    high_degree_set = set([i[0] for i in degree_list if i[1] >= _building_block_degree])
    #DAG_BB_set = set(high_degree_set & kangde_BB)
    DAG_BB_set = high_degree_set
    with open(os.path.join(save_path, "building_blocks.pkl"), "wb") as f:
        pickle.dump(DAG_BB_set, f)
    tree_dataset = mega_graph.extract_tree_from_graph(free_graph, DAG_BB_set, save_path)
    #with open(os.path.join(save_path, 'DAG_dataset.pkl'), 'wb') as f:
    #    pickle.dump(tree_dataset, f)
    '''


preprocessing(
    dataset_path, commercial_path,  "/root/jupyter/DAG_Transformer/DAG_Transformer/dataset/syn_dag_from_qb/data/reaction_graph"
)

import os
import pickle
import random

import chem_utils_for_reactant as chem_utils
import networkx as nx
import reaction
import tqdm
from rdkit import Chem
from rdkit.Chem import QED


def create_graph_chained(reaction_dataset: reaction.Reactions):
    """
    create graph nodes and edges from the reaction dataset
    nodes: reaction, reactant, product
    edges: rectant -> reaction -> product
    args:
        reaction_dataset: reaction.Reactions object
    """
    graph_chained = nx.DiGraph()
    mol_seen = set()
    for reaction in tqdm.tqdm(reaction_dataset, desc="building chained graph with reactions"):
        
        reaction_rep = ('.'.join([chem_utils.can_smiles(reaction[0][1][i][0]) for i in range(len(reaction[0][1]))]), chem_utils.can_smiles(reaction[1]), reaction[3])
        graph_chained.add_node(reaction_rep)
        node_r = chem_utils.can_smiles(reaction[0][0][0])
        if node_r not in mol_seen:
            graph_chained.add_node(node_r)
            mol_seen.add(node_r)
        node_p = chem_utils.can_smiles(reaction[2])
        if node_p not in mol_seen:
            graph_chained.add_node(node_p)
            mol_seen.add(node_p)
        graph_chained.add_edge(node_r, reaction_rep)
        graph_chained.add_edge(reaction_rep, node_p)
    return graph_chained

def bfs_walk(graph_chained, root, depth):
    """
    function to walk the graph with BFS
    args:
        graph_chained: nx.DiGraph object
        root: root node of the walk
        depth: depth of the walk
    """
    bfs_chains = []
    seen_node = set([root])
    for d in range(depth - 1):
        seen_node_next = set()
        for node in seen_node:
            for pred in graph_chained.predecessors(node):
                if d == 0:
                    bfs_chains.append([node, pred])
                    seen_node_next.add(pred)
                else:
                    for i, chain in enumerate(bfs_chains):
                        if chain[-1] == node:
                            bfs_chains[i].append(pred)
                            seen_node_next.add(pred)
        seen_node = seen_node_next
    chains_len = [len(chain) for chain in bfs_chains]
    for l in chains_len:
        if l % 2 == 0:
            print('odd length chain')
    return bfs_chains


def extract_chain_from_graph(graph_chained, depth:list, save_path, QED_hold=0.5):
    """
    extract chains from the chained graph
    args:
        graph_chained: nx.DiGraph object
        depth: max depth of the walk to extract the chains
        save_path: path to save the extracted chains
        QED_hold: QED score threshold for the start node of the chain
    """
    
    start_node = []
    # find the start node with QED score filter
    for node in tqdm.tqdm(graph_chained.nodes(),  desc='finding start node'):
        if isinstance(node, str):
            if QED.qed(Chem.MolFromSmiles(node)) >= QED_hold:
                start_node.append(node)
    with open(os.path.join(save_path, 'start_chain_node.pkl'), 'wb') as f:
        pickle.dump(start_node, f)

    with open(os.path.join(save_path, 'start_chain_node.pkl'), 'rb') as f:
        start_node = pickle.load(f)
    print(f'{len(start_node)} start node is saved')
    chains_dataset = []

    for root in tqdm.tqdm(start_node):
        walk_path = bfs_walk(graph_chained, root, depth)
        if len(walk_path) > 1:
            for path in walk_path:
                if path not in chains_dataset:
                    chains_dataset.append(path)
    print(f'run extraction. get {len(chains_dataset)} chains to ultize')
    mean_len = sum([len(chain) for chain in chains_dataset])/len(chains_dataset)
    print(f'average length of the chains is {mean_len}')
    return chains_dataset
    


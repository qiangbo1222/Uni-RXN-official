import os
import pickle
import random

import networkx as nx
import tqdm
from rdkit import Chem
from rdkit.Chem import QED

import chem_utils_for_reactant as chem_utils
import reaction


def create_graph_free(reaction_dataset: reaction.Reactions):
    garph_free = nx.DiGraph()
    mol_seen = set()
    for reaction in tqdm.tqdm(
        reaction_dataset, desc="building free graph with reactions"
    ):
        # add reaction node
        reaction_rep = (reaction[3], reaction[1])
        garph_free.add_node(reaction_rep)
        # add the reactant to the graph
        for reactant in reaction[0]:
            node = chem_utils.can_smiles(reactant)
            if node not in mol_seen:
                garph_free.add_node(node)
                mol_seen.add(node)
            garph_free.add_edge(node, reaction_rep)
        # add the product to the graph
        node = chem_utils.can_smiles(str(reaction[2]))
        if node not in mol_seen:
            garph_free.add_node(node)
            mol_seen.add(node)
        # link the product and the reactant with a direct edge
        garph_free.add_edge(reaction_rep, node)
    return garph_free


def create_graph_baseon_BB(reaction_dataset: reaction.Reactions, building_blocks: set):
    """ABANDONED BECAUSE INNER BUG WHEN REACTION CANNOT BE ADD IF THE ORDER IS WRONG
         B + C = X  X + B = Y, GOING THROUGH FOWARD IS OK, BUT IF IT WAS GOING THROUGH BACKWARD BOTH CAN NOT BE ADD TO THE GRAPH
    graph_BB = nx.DiGraph()
    mol_seen = building_blocks
    # initial the building blocks as starting nodes
    graph_BB.add_node_from(
        it
        for it in zip(
            [bb for bb in building_blocks],
            [{"label": "end point"} for lab in range(len(building_blocks))],
        )
    )
    reaction_waiting_list = set()
    reaction_pool = set([reaction for reaction in reaction_dataset])
    counter = 0
    while len(reaction_pool) != 0:
        counter += 1
        print(f'\r going through reactions at {counter}th loop' )
        for _ in tqdm.tqdm(
            range(len(reaction_pool)), desc="building graph based on building blocks"
        ):
            reaction = reaction_pool.pop()
            reactants = set(can_smiles(reactant) for reactant in reaction[0])
            reagants = set(can_smiles(reagant) for reagant in reaction[1])
            product = can_smiles(reaction[2])
            # if the reaction start with building blocks, the reaction is added
            if (
                reactants.issubset(mol_seen)
                and product not in mol_seen
            ):
                graph_BB.add_node_from([reaction[3], ], [{'label': 'reaction'}, ])
                for reactant in reactants:
                    graph_BB.add_edge(reactant, reaction[3])
                graph_BB.add_node([product, ], [{'label': 'intermediate or product'}])
                graph_BB.add_edge(reaction[3], product)
                #update the pool of seen mol
                mol_seen.add(product)
    """

    pass

def create_graph_chained(reaction_dataset: reaction.Reactions):
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

    '''
    agent_start = root
    #TODO debug and remove random choice
    next_step_list = [node for node in graph_chained.predecessors(agent_start)]
    if len(next_step_list) == 0 or depth == 1:
        return [root, ]
    else:
        chain = [root, ]
        next_step = random.choice(next_step_list)
        for new in random_walk(graph_chained, next_step, depth-1):
            chain.append(new)
        return chain
    '''

def extract_chain_from_graph(graph_chained, depth:list, save_path, QED_hold=0.5, run_time=1):
    
    start_node = []
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
    #for time in range(run_time):
    for root in tqdm.tqdm(start_node):
        walk_path = bfs_walk(graph_chained, root, depth)
        if len(walk_path) > 1:
            for path in walk_path:
                if path not in chains_dataset:
                    chains_dataset.append(path)
    print(f'run extraction for {run_time} times. get {len(chains_dataset)} chains to ultize')
    mean_len = sum([len(chain) for chain in chains_dataset])/len(chains_dataset)
    print(f'average length of the chains is {mean_len}')
    return chains_dataset
    


def recursive_sample_tree(
    graph_free: nx.DiGraph(), product_root: str, building_blocks: list
):
    """
    f       for T in targets:
                            for B in building blocks:
                                    compute(T, B) shortest path (inf if T, B is not connected)
                            randomly choose x B with the x shortest path
                            for B_ in x chosen Bs:
                                    find the reaction’s other reactants which is not on the path
                                    make them T and redo this loop
    """
    tree = nx.DiGraph()
    short_path_dic = {
        key: nx.shortest_path(graph_free, source=key, target=product_root)
        for key in building_blocks
        if nx.has_path(graph_free, key, product_root)
    }
    if len(short_path_dic) == 0:
        return "cannot be synthesis with BB"

    short_path_dic = sorted(
        short_path_dic.items(), key=lambda item: len(item[1]), reverse=False
    )
    exit_flag = False
    for ind in range(len(short_path_dic)):
        one_way_path = list(short_path_dic[ind][1])
        one_way_path.reverse()
        for c, node in enumerate(one_way_path):
            if c == 0:
                tree.add_node(node, role="product")
                ancestor_node = node
            elif c % 2 == 1:
                reagants = node[0]
                tree.add_node(reagants, role="reagants")
                tree.add_edge(ancestor_node, reagants)
                ancestor_node = reagants
            elif c % 2 == 2 and c != len(one_way_path) - 1:
                tree.add_node(node, role="intermediate")
                tree.add_edge(ancestor_node, node)
                ancestor_node = node
            else:
                tree.add_node(node, role="building_block")
                tree.add_edge(ancestor_node, node)

        for i in range(1, len(one_way_path), 2):
            neighbor_node = [
                edge[0]
                for edge in graph_free.in_edges((one_way_path[i],))
                if edge[0] != one_way_path[i + 1]
            ]
            if len(neighbor_node) > 0:
                for sub_root in neighbor_node:
                    neighbor_graph = recursive_sample_tree(
                        graph_free, sub_root, building_blocks
                    )
                    if neighbor_graph != "cannot be synthesis with BB":
                        tree = nx.disjoint_union(tree, neighbor_graph)
                        tree.add_edge(one_way_path[i], sub_root)
                    else:
                        exit_flag = True
            else:
                return tree
            if exit_flag == True:
                break
        return tree


def extract_tree_from_graph(graph: nx.DiGraph(), building_blocks: list, save_path, QED_hold=0.5):
    # choose the okay root products
    
    root_products = []
    for node in tqdm.tqdm(graph.nodes(), desc='finding okay root products'):
        if isinstance(node, str) and QED.qed(Chem.MolFromSmiles(node)) > QED_hold and node not in building_blocks:
            root_products.append(node)
    print(f"{len(root_products)} products is able prepared to generate tree")
    with open(os.path.join(save_path, 'root_products.pkl'), 'wb') as f:
        pickle.dump(root_products, f)
    '''
    with open(os.path.join(save_path, 'root_products.pkl'), 'rb') as f:
        root_products = pickle.load(f)
    tree_dataset = []
    for root in tqdm.tqdm(root_products, desc="sampling trees"):
        tree = recursive_sample_tree(graph, root, building_blocks)
        if not isinstance(tree, str):
            tree_dataset.append(tree.reverse())
            nx.draw(tree_dataset[-1])
    print(f"generate the dataset size of {len(tree_dataset)}")
    '''
    return tree_dataset


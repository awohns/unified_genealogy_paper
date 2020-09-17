import multiprocessing
import tskit
import json
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import collections

ts = tskit.load("merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic.trees")

node_ages = ts.tables.nodes.time[:]
deleted_trees = [tree.index for tree in ts.trees() if tree.parent(0) == -1]  

metadata = ts.tables.nodes.metadata[:]
metadata_offset = ts.tables.nodes.metadata_offset[:]
for index, met in enumerate(tskit.unpack_bytes(metadata, metadata_offset)):
    if index not in ts.samples():
        try:
            node_ages[index] = json.loads(met.decode())["mn"]
        except json.decoder.JSONDecodeError:
            raise ValueError(
                    "Tree Sequence must be dated to use unconstrained=True")
pop_nodes = ts.tables.nodes.population[ts.samples()]
pop_nodes = [np.where(pop_nodes == pop.id)[0] for pop in ts.populations()]
rand_nodes = list()
for nodes in pop_nodes:
    if len(nodes) > 20:
        rand_nodes.append(np.random.choice(nodes, 20, replace=False))
    else:
        rand_nodes.append(nodes)


def get_pairwise_tmrca_pops(ts):
    pop_names = [json.loads(pop.metadata)["name"] for pop in ts.populations()]
    if input == "1kg_sgdp_hgdp":
        pop_name_suffixes = pop_names[0:26]
        for pop in pop_names[26:156]:
            pop_name_suffixes.append(pop + "_SGDP")
        for pop in pop_names[156:]:
            pop_name_suffixes.append(pop + "_HGDP")
        pop_names = pop_name_suffixes
    tmrca_df = pd.DataFrame(columns=pop_names, index=pop_names)
    pop_rows = list()
    combos = list(itertools.combinations_with_replacement(np.arange(0, ts.num_populations),2))
    weights = np.array([tree.span for tree in ts.trees() if tree.index not in deleted_trees])
    with multiprocessing.Pool(processes=10) as pool: 
        for avg_tmrca, combo in tqdm(pool.imap_unordered(get_avg_tmrca, combos), total=len(combos)):
            tmrca_df.loc[pop_names[combo[0]], pop_names[combo[1]]] = np.exp(np.average(np.log(avg_tmrca), weights=weights))
    return tmrca_df

def get_avg_tmrca(combo):
    avg_tmrca = []
    pop_0 = combo[0]
    pop_1 = combo[1]
    pop_0_nodes = rand_nodes[pop_0]
    if pop_0 != pop_1:
        pop_1_nodes = rand_nodes[pop_1]
        node_combos = [(x, y) for x in pop_0_nodes for y in pop_1_nodes]
    elif pop_0 == pop_1:
        node_combos = list(itertools.combinations(pop_0_nodes, 2))
    avg_tmrca = np.zeros((ts.num_trees - len(deleted_trees), len(node_combos)))

    t_index = 0
    for tree in ts.trees(): 
        if tree.index not in deleted_trees:
            tree_tmrcas = []
            for index, (node_0, node_1) in enumerate(node_combos):
                avg_tmrca[t_index, index] = node_ages[tree.mrca(node_0, node_1)]
            t_index += 1
    avg_tmrca = np.exp(np.mean(np.log(avg_tmrca), axis=1))
    return avg_tmrca, combo


df = get_pairwise_tmrca_pops(ts)
df.to_csv("all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic.tmrcas")

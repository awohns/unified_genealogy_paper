import multiprocessing
import tskit
import json
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import collections

def get_pairwise_tmrca_pops(
    ts_name, max_pop_nodes, hist_nbins=30, hist_min_gens=1000, return_full_data=False
):
    """
    max_pop_nodes gives the maximum number of sample nodes per pop to use
    hist_nbins determines the number of bins used to save the histogram data,
    hist_min_gens gives a lower cutoff for the histogram bins, as there is usually
    very little in the lowest (logged) bins
    if return_full_data is True, also return the full dataset of weights (which may be
    huge, as it is ~ num_unique_times * n_pops * n_pops /2
    """
    ts = tskit.load(ts_name)
    deleted_trees = [tree.index for tree in ts.trees() if tree.parent(0) == -1]  
    node_ages = np.zeros_like(ts.tables.nodes.time[:])
    metadata = ts.tables.nodes.metadata[:]
    metadata_offset = ts.tables.nodes.metadata_offset[:]
    try:
        for index, met in enumerate(tskit.unpack_bytes(metadata, metadata_offset)):
            if index not in ts.samples():
                try:
                    # Get unconstrained node age if available
                    node_ages[index] = json.loads(met.decode())["mn"]
                except json.decoder.JSONDecodeError:
                    raise ValueError(
                            "Tree Sequence must be dated to use unconstrained=True")
        print("Using tsdate unconstrained node times")
    except KeyError:
        print("Using standard ts node times")
        node_ages[:] = ts.tables.nodes.time[:]
    unique_times, time_index = np.unique(node_ages, return_inverse=True)
    with np.errstate(divide='ignore'):
        log_unique_times = np.log(unique_times)

    # Make a random selection of up to 10 samples from each population
    np.random.seed(123)
    pop_nodes = ts.tables.nodes.population[ts.samples()]
    nodes_for_pop = {}
    for pop in ts.populations():
        metadata = json.loads(pop.metadata)
        key = metadata["name"]
        # Hack to distinguish SGDP from HGDP (all uppercase) pop names
        if 'region' in metadata and not metadata['region'].isupper():
            key += " (SGDP)" 
        assert key not in nodes_for_pop  # Check for duplicate names
        nodes = np.where(pop_nodes == pop.id)[0]
        if len(nodes) > max_pop_nodes:
            nodes_for_pop[key] = np.random.choice(nodes, max_pop_nodes, replace=False)
        else:
            nodes_for_pop[key] = nodes
    
    # Make all combinations of populations
    pop_names = list(nodes_for_pop.keys())
    tmrca_df = pd.DataFrame(columns=pop_names, index=pop_names)
    combos = itertools.combinations_with_replacement(np.arange(0, len(pop_names)), 2)
    combo_map = {c: i for i, c in enumerate(combos)}
    func_params = zip(
        combo_map.keys(),
        itertools.repeat(time_index),
        itertools.repeat(list(nodes_for_pop.values())),
        itertools.repeat(ts_name),
        itertools.repeat(deleted_trees),
    )
    data = np.zeros((len(combo_map), len(unique_times)), dtype=np.float)
    with multiprocessing.Pool(processes=64) as pool: 
        for tmrca_weight, combo in tqdm(
            pool.imap_unordered(get_tmrca_weights, func_params), total=len(combo_map)
        ):
            popA = pop_names[combo[0]]
            popB = pop_names[combo[1]]
            keep = (tmrca_weight != 0)  # Deal with log_unique_times[0] == -inf
            mean_log_age = np.sum(log_unique_times[keep] * tmrca_weight[keep])
            mean_log_age /= np.sum(tmrca_weight) # Normalise
            tmrca_df.loc[popA, popB] = np.exp(mean_log_age)
            data[combo_map[combo], :] = tmrca_weight
    bins, hist_data = make_histogram_data(
        log_unique_times, data, hist_nbins, hist_min_gens)
    named_combos = [None] * len(combo_map)
    for combo, i in combo_map.items():
        named_combos[i] = (pop_names[combo[0]], pop_names[combo[1]])
    named_combos = np.array(named_combos)
    if return_full_data:
        return tmrca_df, bins, hist_data, named_combos, data
    else:
        return tmrca_df, bins, hist_data, named_combos

def make_histogram_data(log_unique_times, data, hist_nbins, hist_min_gens):
    """
    NB: this can also be called on the (saved) full data matrix, if histograms need
    re-calculating with different bin widths etc.
    """
    av_weight = np.mean(data, axis=0)
    keep = (av_weight != 0)
    #Make common breaks for histograms
    _, bins = np.histogram(
        log_unique_times[keep],
        weights=av_weight[keep],
        bins=hist_nbins,
        range=[np.log(hist_min_gens), max(log_unique_times)],
        density=True)
    hist_data = np.zeros((data.shape[0], hist_nbins), dtype=np.float32)
    for i, row in enumerate(data):
        hist_data[i, :], _ = np.histogram(
            log_unique_times[keep],
            weights=row[keep],
            bins=bins,
            density=True,
        )
    return bins, hist_data

def get_tmrca_weights(params):
    combo, time_index, rand_nodes, ts_name, deleted_trees = params
    ts = tskit.load(ts_name)
    pop_0 = combo[0]
    pop_1 = combo[1]
    num_unique_times = max(time_index) + 1
    pop_0_nodes = rand_nodes[pop_0]
    if pop_0 != pop_1:
        pop_1_nodes = rand_nodes[pop_1]
        node_combos = [(x, y) for x in pop_0_nodes for y in pop_1_nodes]
    elif pop_0 == pop_1:
        node_combos = list(itertools.combinations(pop_0_nodes, 2))
    # Return the weights 
    tmrca_weight = np.zeros(num_unique_times, dtype=np.float)

    for tree in ts.trees(): 
        if tree.index not in deleted_trees:
            for index, (node_0, node_1) in enumerate(node_combos):
                tmrca_weight[time_index[tree.mrca(node_0, node_1)]] += tree.span
    return tmrca_weight, combo

if __name__ == '__main__':
    fn = "all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic"
    ts_name = fn + ".trees"
    max_pop_nodes = 20
    tmrca_df, bins, histdata, combos = get_pairwise_tmrca_pops(ts_name, max_pop_nodes)
    outfn = fn + f".{max_pop_nodes}nodes.tmrcas"
    print(f"Writing mean MRCAs to {outfn}.csv")
    tmrca_df.to_csv(outfn + ".csv")
    print(f"Writing bins and MRCA histogram distributions to {outfn}.npz")
    np.savez_compressed(outfn + ".npz", bins=bins, histdata=histdata, combos=combos)

import multiprocessing
import tskit
import json
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import collections

def get_pairwise_tmrca_pops(
    ts, hist_nbins=30, hist_min_gens=1000, return_full_data=False
):
    """
    hist_nbins determines the number of bins used to save the histogram data,
    hist_min_gens gives a lower cutoff for the histogram bins, as there is usually
    very little in the lowest (logged) bins
    if return_full_data is True, also return the full dataset of weights (which may be
    huge, as it is ~ num_unique_times * n_pops * n_pops /2
    """
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
    pop_nodes = [np.where(pop_nodes == pop.id)[0] for pop in ts.populations()]
    rand_nodes = list()
    for nodes in pop_nodes:
        if len(nodes) > 20:
            rand_nodes.append(np.random.choice(nodes, 20, replace=False))
        else:
            rand_nodes.append(nodes)
    
    # Make all combinations of populations
    pop_names = [json.loads(pop.metadata)["name"] for pop in ts.populations()]
    tmrca_df = pd.DataFrame(columns=pop_names, index=pop_names)
    combos = itertools.combinations_with_replacement(np.arange(0, ts.num_populations), 2)
    combo_map = {c: i for i, c in enumerate(combos)}
    func_params = zip(
        combo_map.keys(),
        itertools.repeat(time_index),
        itertools.repeat(rand_nodes),
        itertools.repeat(ts),
        itertools.repeat(deleted_trees),
    )
    data = np.zeros((len(combo_map), len(unique_times)), dtype=np.float)
    with multiprocessing.Pool(processes=2) as pool: 
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
    if return_full_data:
        return tmrca_df, bins, hist_data, data
    else:
        return tmrca_df, bins, hist_data

def make_histogram_data(log_unique_times, data, hist_nbins, hist_min_gens):
    """
    NB: this can also be called on the (saved) full data matrix, if histograms need
    re-calculating with different bin widths etc.
    """
    av_weight = np.mean(data, axis=0)
    keep == (av_weight != 0)
    #Make common breaks for histograms
    _, bins = np.histogram(
        log_unique_times[keep],
        weights=av_weight[keep],
        bins=hist_nbins,
        range=[np.log(hist_min_gens), max(log_unique_times)],
        density=True)
    hist_data = np.zeros((len(combo_map), hist_nbins), dtype=np.float32)
    for i, row in enumerate(data):
        hist_data[i, :] = np.histogram(
            log_unique_times[keep],
            weights=row[keep],
            bins=bins,
            density=True,
        )
    return bins, hist_data

def get_tmrca_weights(params):
    combo, time_index, rand_nodes, ts, deleted_trees = params
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
    file = "all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic"
    ts = tskit.load(file + ".trees")
    
    tmrca_df, bins, hist_data = get_pairwise_tmrca_pops(ts)
    outfile = file + ".tmrcas"
    print(f"Writing mean MRCAs to {outfile}.csv")
    tmrca_df.to_csv(outfile + ".csv")
    print(f"Writing bins and MRCA histogram distributions to {outfile}.npz")
    np.savez_compressed(outfile + ".npz", bins=bins, hist_data=hist_data)

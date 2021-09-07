import multiprocessing
import tskit
import json
import numpy as np
import pandas as pd
import itertools
import logging
import argparse
import collections
import os

from tqdm import tqdm


TmrcaData = collections.namedtuple("TMRCA_data", ["means", "histogram", "raw_data"])
HistData = collections.namedtuple("Hist_data", ["bin_edges", "data", "rownames"])


def get_pairwise_tmrca_pops(
    ts_name,
    max_pop_nodes,
    hist_nbins=30,
    hist_min_gens=1000,
    num_processes=1,
    restrict_populations=None,
    return_raw_data=False,
):
    """
    Get the mean tMRCA and a histogram of tMRCA times for pairs of populations from a
    tree sequence.

    :param int max_pop_nodes: The maximum number of sample nodes per pop to use. This
        number of samples (or lower, for small populations) will be taken at random from
        each population as a set of representative samples for which to construct
        pairwise statistics
    :param int hist_nbins: The number of bins used to save the histogram data. Bins will
        be spaced out evenly on a log scale.
    :param float hist_min_gens: A lower cutoff for the histogram bins, as there is
        usually very little in the lowest (logged) bins
    :param int num_processes: The number of CPUs to run in parallel on the calculation.
    :param list restrict_populations: A list of population IDs or names giving the
        populations among which to calculate pairwise distances. If ``None`` (default)
        then use all the populations defined in the tree sequence.
    :param bool return_raw_data is True, also return the full dataset of weights (which
        may be huge, as it is ~ num_unique_times * n_pops * n_pops /2

    :return: a TmrcaData object containing a dataframe of the mean values for each
        pair, a HistData object with the histogram data, and (if return_full_data is
        ``True``) a potentially huge numpy array of weights of pairs X unique_times
    :rtype: TmrcaData
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
                        "Tree Sequence must be dated to use unconstrained=True"
                    )
        logging.info("Using tsdate unconstrained node times")
    except ValueError:
        logging.info("Using standard ts node times")
        node_ages[:] = ts.tables.nodes.time[:]
    unique_times, time_index = np.unique(node_ages, return_inverse=True)
    with np.errstate(divide="ignore"):
        log_unique_times = np.log(unique_times)

    # Make a random selection of up to 10 samples from each population
    np.random.seed(123)
    pop_nodes = ts.tables.nodes.population[ts.samples()]
    nodes_for_pop = {}
    if restrict_populations is None:
        pops = [pop.id for pop in ts.populations()]
    else:
        # Convert any named populations to population ids
        name2id = {json.loads(pop.metadata)["name"]: pop.id for pop in ts.populations()}
        pops = [int(p) if p.isdigit() else name2id[p] for p in restrict_populations]
    for pop_id in pops:
        metadata = json.loads(ts.population(pop_id).metadata)
        key = metadata["name"]
        # Hack to distinguish SGDP from HGDP (all uppercase) pop names
        if "region" in metadata and not metadata["region"].isupper():
            key += " (SGDP)"
        assert key not in nodes_for_pop  # Check for duplicate names
        nodes = np.where(pop_nodes == pop_id)[0]
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
    with multiprocessing.Pool(processes=num_processes) as pool:
        for tmrca_weight, combo in tqdm(
            pool.imap_unordered(get_tmrca_weights, func_params), total=len(combo_map)
        ):
            popA = pop_names[combo[0]]
            popB = pop_names[combo[1]]
            keep = tmrca_weight != 0  # Deal with log_unique_times[0] == -inf
            mean_log_age = np.sum(log_unique_times[keep] * tmrca_weight[keep])
            mean_log_age /= np.sum(tmrca_weight)  # Normalise
            tmrca_df.loc[popA, popB] = np.exp(mean_log_age)
            data[combo_map[combo], :] = tmrca_weight
    bins, hist_data = make_histogram_data(
        log_unique_times, data, hist_nbins, hist_min_gens
    )
    named_combos = [None] * len(combo_map)
    for combo, i in combo_map.items():
        named_combos[i] = (pop_names[combo[0]], pop_names[combo[1]])
    hist = HistData(bins, hist_data, np.array(named_combos))
    if return_raw_data is False:
        data = None
    return TmrcaData(means=tmrca_df, histogram=hist, raw_data=(log_unique_times, data))


def make_histogram_data(log_unique_times, data, hist_nbins, hist_min_gens):
    """
    Return an tuple of (bin_edges, array), where the array is of size
    number_of_pairs x n_bins.

    .. note::
        This can also be called on the (saved) full data matrix, if histograms need
        re-calculating with different bin widths etc.
    """
    av_weight = np.mean(data, axis=0)
    keep = av_weight != 0
    # Make common breaks for histograms
    _, bins = np.histogram(
        log_unique_times[keep],
        weights=av_weight[keep],
        bins=hist_nbins,
        range=[np.log(hist_min_gens), max(log_unique_times)],
        density=True,
    )
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


def save_tmrcas(
    ts_file, max_pop_nodes, populations=None, num_processes=1, save_raw_data=False
):
    if not ts_file.endswith(".trees"):
        raise valueError("Tree sequence must end with '.trees'")
    fn = ts_file[: -len(".trees")]
    tMRCAS = get_pairwise_tmrca_pops(
        ts_file,
        max_pop_nodes,
        restrict_populations=populations,
        num_processes=num_processes,
        return_raw_data=save_raw_data,
    )
    popstring = "all" if populations is None else "+".join(populations)
    outfn = (
        os.path.join("data", os.path.basename(fn))
        + f".{max_pop_nodes}nodes_{popstring}.tmrcas"
    )
    logging.info(f"Writing mean MRCAs to {outfn}.csv")
    tMRCAS.means.to_csv(outfn + ".csv")
    logging.info(f"Writing bins and MRCA histogram distributions to {outfn}.npz")
    hist = tMRCAS.histogram
    np.savez_compressed(
        outfn + ".npz", bins=hist.bin_edges, histdata=hist.data, combos=hist.rownames
    )
    if save_raw_data:
        logging.info(f"Saving raw data to {outfn}_RAW.npz")
        np.savez_compressed(outfn + "_RAW.npz", *tMRCAS.raw_data)


def main(args):
    if args.verbosity == 0:
        logging.basicConfig(level=logging.WARNING)
    elif args.verbosity == 1:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    elif args.verbosity >= 2:
        logging.basicConfig(level=logging.DEBUG)

    save_tmrcas(
        args.tree_sequence,
        args.max_pop_nodes,
        args.populations,
        args.num_processes,
        args.save_raw_data,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate pairwise mean tMRCAs from a tree sequence. "
        "This can be quite time-consuming if there are many populations"
    )
    parser.add_argument("tree_sequence")
    parser.add_argument(
        "--max_pop_nodes",
        "-m",
        type=int,
        default=20,
        help="The maximum number of samples nodes to compare per population",
    )
    parser.add_argument(
        "--populations",
        "-P",
        nargs="*",
        default=None,
        help="Restrict pairwise calculations to particular populations. "
        " If None, use all populations in the tree sequence.",
    )
    parser.add_argument(
        "--num_processes",
        "-p",
        type=int,
        default=64,
        help="The number of CPUs to use in the calculation",
    )
    parser.add_argument(
        "--save_raw_data",
        action="store_true",
        help="Also save the (potentially huge) raw data file",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        action="count",
        default=0,
        help="verbosity: output extra non-essential info",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

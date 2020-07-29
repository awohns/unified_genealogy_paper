#!/bin/python
import argparse
import tsdate
import tsinfer
import tskit
import pickle
import numpy as np

"""
Script to date a tree sequence
"""

def simplify_out_ancients(ts, samples_fn, output_fn):
    samples = tsinfer.load(samples_fn)
    ancient = np.where(samples.individuals_time[:] != 0)[0]
    modern_only_ts = ts.simplify(
            samples=np.where(~np.isin(ts.tables.nodes.individual[:][ts.samples()], ancient))[0], filter_sites=False)
    modern_only_ts.dump(output_fn + ".modern.trees")
    return modern_only_ts

def trim_tree_sequence(ts, output_fn, gap_size=1000000):
    sites = ts.tables.sites.position[:]
    first_pos = sites[0]
    last_pos = sites[-1]
    ts_trimmed = ts.keep_intervals([[first_pos, last_pos + 1]], simplify=False)
    sites = ts_trimmed.tables.sites.position[:]
    gaps = np.argsort(sites[1:] - sites[:-1])
    delete_intervals = []
    for gap in gaps:
        gap_start = sites[gap] + 1
        gap_end = sites[gap + 1]
        if gap_end - gap_start > gap_size:
            print("Gap Size", start, end, "Snipping topology from ", start, end)
            delete_intervals.append([gap_start, gap_end])
    ts_trimmed = ts.delete_intervals(delete_intervals)
    ts_trimmed = ts_trimmed.simplify(filter_sites=False)
    ts_trimmed.dump(output_fn + ".trimmed.trees")
    return ts_trimmed


def run(input_ts, output_fn, Ne, mutation_rate, no_constrain, simplify_ancients, num_threads):
    ts = tskit.load(input_ts)
    if simplify_ancients is not None:
        ts = simplify_out_ancients(ts, simplify_ancients, output_fn)
        output_fn = output_fn + ".modern"
    else:
        ts = ts.simplify(filter_sites=False)
    ts_trimmed = trim_tree_sequence(ts, output_fn)
    assert ts.num_sites == ts_trimmed.num_sites
    prior = tsdate.build_prior_grid(ts_trimmed, approximate_priors=True, progress=True, timepoints=50)
    mn_post, posterior, timepoints, eps, nonfixed_nodes = tsdate.get_dates(ts_trimmed, Ne, mutation_rate=mutation_rate, priors=prior,
            num_threads=num_threads, method='inside_outside', outside_normalize=True,
                    ignore_oldest_root=False, progress=True, cache_inside=False)
    pickle.dump(mn_post * 2 * Ne, open(output_fn + '.dates.p', 'wb'))
    if not no_constrain:
        constrained= tsdate.constrain_ages_topo(ts_trimmed, mn_post, 1e-6, progress=True)
        #constrained = tsdate.constrain_ages_topo(ts_trimmed, mn_post, prior.timepoints, eps, nonfixed_nodes, progress=True)
        tables = ts.dump_tables()
        tables.nodes.time = constrained * 2 * Ne
        tables.sort()
        tables.tree_sequence().dump(output_fn + ".dated.trees")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_ts", default=None,
        help="A tskit tree sequence to date ending in '.trees'. If filename contains chrNN"
            " where 'NN' is a number, assume this is a human tree sequence and cut out the"
            " centromere and the material before first site and after last.")
    parser.add_argument("ne", default=10000, type=float,
        help="Estimated Ne for dating.")
    parser.add_argument("mut_rate", default=1e-8, type=float,
        help="Estimated mutation rate")
    parser.add_argument("-t", "--num_threads", type=int, default=0,
        help="The number of threads to use in inference")
    parser.add_argument("--no-constrain", action="store_true",
        help="Don't constrain ages or return dated tree sequence")
    parser.add_argument("--simplify-ancients", type=str, default=None,
        help="Sampledata file to determine ancient individuals")
    args = parser.parse_args()
    prefix = args.input_ts[0:-len(".trees")]
    run(args.input_ts, prefix, args.ne, args.mut_rate, args.no_constrain, args.simplify_ancients, args.num_threads)

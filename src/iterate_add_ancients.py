#!/bin/python
import argparse
import csv
import re
import pickle

import tsdate
import tskit
import tsinfer
import msprime
import numpy as np
import iteration

def run(input_ts_prefix, samples, num_threads, output_fn, progress):
    """
    Script to take a dated tree sequence inferred from modern samples and add ancients as ancestors
    at the correct times.
    """
    ts = tskit.load(input_ts_prefix + ".dated.trees").simplify()
    centromere = run_snip_centromere(ts, "centromeres.csv", "chr20")
    dates = pickle.load(open(input_ts_prefix + ".dates.p", "rb"))
    sampledata = tsinfer.load(samples)
    other_sites =  ~np.isin(sampledata.sites_position[:], ts.tables.sites.position)
    if np.any(other_sites):
        deleted_sampledata = sampledata.delete(sites=other_sites)
        deleted_sampledata_copy = deleted_sampledata.copy(output_fn + ".consistent_ancients.samples")
        deleted_sampledata_copy.finalise()
        print(deleted_sampledata_copy.num_sites, sampledata.num_sites, ts.num_sites)
        samples, rho, prefix, _ = setup_sample_file(output_fn + ".consistent_ancients.samples", "../yanwong/tsinfer-benchmarking/recomb-hg38/genetic_map_GRCh38_chr20.txt")
    else:
        samples, rho, prefix, _ = setup_sample_file(samples, "../yanwong/tsinfer-benchmarking/recomb-hg38/genetic_map_GRCh38_chr20.txt")
    base_rec_prob = np.quantile(rho, 0.5)
    sample_data_constrained, merged_age = iteration.get_ancient_constraints(samples, dates, ts, output_fn, centromere)
    iteration.tsinfer_second_pass(sample_data_constrained, rho, base_rec_prob * 0.1, base_rec_prob * 0.1, 13, output_fn, num_threads, progress)


def physical_to_genetic(recombination_map, input_physical_positions):
    map_pos = recombination_map.get_positions()
    map_rates = recombination_map.get_rates()
    map_genetic_positions = np.insert(np.cumsum(np.diff(map_pos) * map_rates[:-1]), 0, 0)
    return np.interp(input_physical_positions, map_pos, map_genetic_positions)


def setup_sample_file(sample_fn, genetic_map):
    """ 
    Return a Thousand Genomes Project sample data file, the
    corresponding recombination rate array, a prefix to use for files, and None
    """
    if not sample_fn.endswith(".samples"):
        raise ValueError("Sample data file must end with '.samples'")
    sd = tsinfer.load(sample_fn)
    inference_pos = sd.sites_position[:][sd.sites_inference[:]]

    match = re.search(r'(chr\d+)', sample_fn)
    if match or genetic_map is not None:
        if map is not None:
            chr_map = msprime.RecombinationMap.read_hapmap(genetic_map)
        else:
            chr = match.group(1)
            print(f"Using {chr} from HapMapII_GRCh37 for the recombination map")
            genetic_map = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
            if not genetic_map.is_cached():
                genetic_map.download()
            chr_map = genetic_map.get_chromosome_map(chr)
        inference_distances = physical_to_genetic(chr_map, inference_pos)
        d = np.diff(inference_distances)
        rho = np.concatenate(([0.0], d)) 
    else:
        inference_distances = inference_pos
        d = np.diff(inference_distances)
        rho = np.concatenate(
            ([0.0], d/sd.sequence_length))

    return sd, rho, sample_fn[:-len(".samples")], None


def run_snip_centromere(ts, centromeres, chrom):
    with open(centromeres) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["chrom"] == chrom:
                start = int(row["start"])
                end = int(row["end"])
                break
        else:
            raise ValueError("Did not find row")
    position = ts.tables.sites.position
    s_index = np.searchsorted(position, start)
    e_index = np.searchsorted(position, end)
    # We have a bunch of sites within the centromere. Get the largest
    # distance between these and call these the start and end. Probably
    # pointless having the centromere coordinates as input in the first place,
    # since we're just searching for the largest gap anyway. However, it can
    # be useful in UKBB, since it's perfectly possible that the largest
    # gap between sites isn't in the centromere.
    X = position[s_index : e_index + 1]
    j = np.argmax(X[1:] - X[:-1])
    real_start = X[j] + 1
    real_end = X[j + 1]
    return (real_start, real_end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_ts", default=None,
        help="A tskit tree sequence to date ending in '.trees'. If filename contains chrNN"
            " where 'NN' is a number, assume this is a human tree sequence and cut out the"
            " centromere and the material before first site and after last.")
    parser.add_argument("ancient_samples", default=None,
        help="File containing ancient samples.")
    parser.add_argument("-t", "--num_threads", type=int, default=0,
        help="The number of threads to use in inference")
    parser.add_argument("output_fn", default=None,
        help="Prefix for output files.")
    parser.add_argument('-p', '--progress', action='store_true',
        help="Show progress bar.")
    args = parser.parse_args()
    run(args.input_ts, args.ancient_samples, args.num_threads, args.output_fn, args.progress)

"""
    print(args.chrom)
Various utilities for manipulating tree sequences and running tsinfer.
"""
import argparse
import subprocess
import time
import collections
import json
import sys
import csv
import itertools
import os.path
import re
from functools import reduce

import tskit
import tsinfer
import daiquiri
import numpy as np
import pandas as pd
import tqdm
import humanize
import cyvcf2


def run_simplify(args):
    ts = tskit.load(args.input)
    ts = ts.simplify()
    ts.dump(args.output)


def run_augment(sample_data, ancestors_ts, subset, num_threads):
    progress_monitor = tsinfer.cli.ProgressMonitor(enabled=True, augment_ancestors=True)
    return tsinfer.augment_ancestors(
        sample_data,
        ancestors_ts,
        subset,
        num_threads=num_threads,
        progress_monitor=progress_monitor,
    )


def run_match_samples(sample_data, ancestors_ts, num_threads):
    progress_monitor = tsinfer.cli.ProgressMonitor(enabled=True, match_samples=True)
    return tsinfer.match_samples(
        sample_data,
        ancestors_ts,
        num_threads=num_threads,
        simplify=False,
        progress_monitor=progress_monitor,
    )


def run_sequential_augment(args):

    base = ".".join(args.input.split(".")[:-1])

    sample_data = tsinfer.load(args.input)
    num_samples = sample_data.num_samples
    ancestors_ts = tskit.load(base + ".ancestors.trees")

    # Compute the total samples required.
    n = 2
    total = 0
    while n < num_samples // 4:
        total += n
        n *= 2

    np.random.seed(args.seed)
    samples = np.random.choice(np.arange(num_samples), size=total, replace=False)
    np.save(base + ".augmented_samples.npy", samples)

    n = 2
    j = 0
    while n < num_samples // 4:
        augmented_file = base + ".augmented_{}.ancestors.trees".format(n)
        final_file = base + ".augmented_{}.nosimplify.trees".format(n)
        subset = samples[j : j + n]
        subset.sort()
        ancestors_ts = run_augment(sample_data, ancestors_ts, subset, args.num_threads)
        ancestors_ts.dump(augmented_file)
        j += n
        n *= 2

    final_ts = run_match_samples(sample_data, ancestors_ts, args.num_threads)
    final_ts.dump(final_file)


def run_benchmark_tskit(args):

    before = time.perf_counter()
    ts = tskit.load(args.input)
    duration = time.perf_counter() - before
    print("Loaded in {:.2f}s".format(duration))

    print("num_nodes = ", ts.num_nodes)
    print("num_edges = ", ts.num_edges)
    print("num_trees = ", ts.num_trees)
    print("size = ", humanize.naturalsize(os.path.getsize(args.input), binary=True))

    before = time.perf_counter()
    j = 0
    for tree in ts.trees(sample_counts=False):
        j += 1
    assert j == ts.num_trees
    duration = time.perf_counter() - before
    print("Iterated over trees in {:.2f}s".format(duration))

    before = time.perf_counter()
    num_variants = 0
    # As of msprime 0.6.1, it's a little bit more efficient to specify the full
    # samples and use the tree traversal based decoding algorithm than the full
    # sample-lists for UKBB trees. This'll be fixed in the future.
    for var in ts.variants(samples=ts.samples()):
        if num_variants == args.num_variants:
            break
        num_variants += 1
    duration = time.perf_counter() - before
    total_genotypes = (ts.num_samples * num_variants) / 10 ** 6
    print(
        "Iterated over {} variants in {:.2f}s @ {:.2f} M genotypes/s".format(
            num_variants, duration, total_genotypes / duration
        )
    )


def run_benchmark_vcf(args):

    before = time.perf_counter()
    records = cyvcf2.VCF(args.input)
    duration = time.perf_counter() - before
    print("Read BCF header in {:.2f} seconds".format(duration))
    before = time.perf_counter()
    count = 0
    for record in records:
        count += 1
    duration = time.perf_counter() - before
    print("Read {} VCF records in {:.2f} seconds".format(count, duration))


def get_augmented_samples(tables):
    # Shortcut. Iterating over all the IDs is very slow here.
    # Note that we don't necessarily recover all of the samples that were
    # augmented here because they might have been simplified out.
    # return np.load("ukbb_chr20.augmented_samples.npy")
    nodes = tables.nodes
    ids = np.where(nodes.flags == tsinfer.NODE_IS_SAMPLE_ANCESTOR)[0]
    sample_ids = np.zeros(len(ids), dtype=int)
    for j, node_id in enumerate(tqdm.tqdm(ids)):
        offset = nodes.metadata_offset[node_id : node_id + 2]
        buff = bytearray(nodes.metadata[offset[0] : offset[1]])
        md = json.loads(buff.decode())
        sample_ids[j] = md["sample"]
    return sample_ids


def run_compute_ukbb_gnn(args):
    ts = tskit.load(args.input)
    tables = ts.tables
    before = time.time()
    augmented_samples = set(get_augmented_samples(tables))
    duration = time.time() - before
    print("Got augmented:", len(augmented_samples), "in ", duration)

    reference_sets_map = collections.defaultdict(list)

    ind_metadata = [None for _ in range(ts.num_individuals)]
    all_samples = []
    for ind in ts.individuals():
        md = json.loads(ind.metadata.decode())
        ind_metadata[ind.id] = md
        for node in ind.nodes:
            if node not in augmented_samples:
                reference_sets_map[md["CentreName"]].append(node)
                all_samples.append(node)
    reference_set_names = list(reference_sets_map.keys())
    reference_sets = [reference_sets_map[key] for key in reference_set_names]

    cols = {
        "centre": [
            ind_metadata[ts.node(u).individual]["CentreName"] for u in all_samples
        ],
        "sample_id": [
            ind_metadata[ts.node(u).individual]["SampleID"] for u in all_samples
        ],
        "ethnicity": [
            ind_metadata[ts.node(u).individual]["Ethnicity"] for u in all_samples
        ],
    }
    print("Computing GNNs for ", len(all_samples), "samples")
    before = time.time()
    A = ts.genealogical_nearest_neighbours(
        all_samples, reference_sets, num_threads=args.num_threads
    )
    duration = time.time() - before
    print("Done in {:.2f} mins".format(duration / 60))

    for j, name in enumerate(reference_set_names):
        cols[name] = A[:, j]
    df = pd.DataFrame(cols)
    df.to_csv(args.output)


def run_compute_1kg_gnn(args):
    ts = tskit.load(args.input)

    population_name = []
    region_name = []

    for population in ts.populations():
        md = json.loads(population.metadata.decode())
        name = md["name"]
        population_name.append(name)
        if "super_population" in md:
            region_name.append(md["super_population"])
        elif "region" in md:
            region_name.append(md["region"])
        elif "name" in md:
            region_name.append(md["name"])

    population = []
    region = []
    individual = []
    for j, u in enumerate(ts.samples()):
        node = ts.node(u)
        ind = json.loads(ts.individual(node.individual).metadata.decode())
        if "individual_id" in ind:
            individual.append(ind["individual_id"])
        elif "name" in ind:
            individual.append(ind["name"])
        population.append(population_name[node.population])
        region.append(region_name[node.population])

    sample_sets = [ts.samples(pop) for pop in range(ts.num_populations)]
    print("Computing GNNs")
    before = time.time()
    A = ts.genealogical_nearest_neighbours(
        ts.samples(), sample_sets, num_threads=args.num_threads
    )
    duration = time.time() - before
    print("Done in {:.2f} mins".format(duration / 60))

    cols = {population_name[j]: A[:, j] for j in range(ts.num_populations)}
    cols["population"] = population
    cols["region"] = region
    cols["individual"] = individual
    df = pd.DataFrame(cols)
    df.to_csv(args.output)


def run_compute_sgdp_gnn(args):
    ts = tskit.load(args.input)

    population_name = []
    region_name = []

    for population in ts.populations():
        md = json.loads(population.metadata.decode())
        name = md["name"]
        population_name.append(name)
        region_name.append(md["region"])

    population = []
    region = []
    individual = []
    for j, u in enumerate(ts.samples()):
        node = ts.node(u)
        ind = json.loads(ts.individual(node.individual).metadata.decode())
        individual.append(ind["sgdp_id"])
        population.append(population_name[node.population])
        region.append(region_name[node.population])

    sample_sets = [ts.samples(pop) for pop in range(ts.num_populations)]
    print("Computing GNNs")
    before = time.time()
    A = ts.genealogical_nearest_neighbours(
        ts.samples(), sample_sets, num_threads=args.num_threads
    )
    duration = time.time() - before
    print("Done in {:.2f} mins".format(duration / 60))

    cols = {population_name[j]: A[:, j] for j in range(ts.num_populations)}
    cols["population"] = population
    cols["region"] = region
    cols["individual"] = individual
    df = pd.DataFrame(cols)
    df.to_csv(args.output)


def run_compute_hgdp_gnn(args):
    ts = tskit.load(args.input)

    population_name = []
    region_name = []

    for population in ts.populations():
        md = json.loads(population.metadata.decode())
        name = md["name"]
        population_name.append(name)
        region_name.append(md["region"])

    population = []
    region = []
    individual = []
    for j, u in enumerate(ts.samples()):
        node = ts.node(u)
        ind = json.loads(ts.individual(node.individual).metadata.decode())
        individual.append(ind["sample"])
        population.append(population_name[node.population])
        region.append(region_name[node.population])

    sample_sets = [ts.samples(pop) for pop in range(ts.num_populations)]
    print("Computing GNNs")
    before = time.time()
    A = ts.genealogical_nearest_neighbours(
        ts.samples(), sample_sets, num_threads=args.num_threads
    )
    duration = time.time() - before
    print("Done in {:.2f} mins".format(duration / 60))

    cols = {population_name[j]: A[:, j] for j in range(ts.num_populations)}
    cols["population"] = population
    cols["region"] = region
    cols["individual"] = individual
    df = pd.DataFrame(cols)
    df.to_csv(args.output)


def run_snip_centromere(args):
    with open(args.centromeres) as csvfile:
        reader = csv.DictReader(csvfile)
        match = re.search(r'(chr\d+)', args.chrom)
        chrom = match.group(1)
        for row in reader:
            if row["chrom"] == chrom:
                start = int(row["start"])
                end = int(row["end"])
                break
        else:
            raise ValueError("Did not find row")
    ts = tskit.load(args.input)
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
    if len(X) > 0:
        j = np.argmax(X[1:] - X[:-1])
        real_start = X[j] + 1
        real_end = X[j + 1]
        print("Centromere at", start, end, "Snipping topology from ", real_start, real_end)
        snipped_ts = ts.delete_intervals([[real_start, real_end]])
        snipped_ts.dump(args.output)
    else:
        print("No gap detected")
        ts.dump(args.output)


def run_combine_sampledata(args):
    """
    Combine a list of SampleData files. 
    Use tskit.MISSING_DATA where data does not appear in one file.
    Assume all individuals are diploid.
    Similar to a database outer join. 
    """

    def make_site_df(samples, file_index):
        ancestral_alleles = [allele[0] for allele in samples.sites_alleles[:]]
        derived_alleles = [
            allele[1] if len(allele) > 1 else np.nan
            for allele in samples.sites_alleles[:]
        ]
        df = pd.DataFrame(
            {
                str(file_index) + " Site ID": np.arange(0, samples.num_sites),
                "Position": samples.sites_position[:],
                "Ancestral Allele": ancestral_alleles,
                "Derived Allele": derived_alleles,
            }
        )
        df = df.astype({str(file_index) + " Site ID": "int32"})
        return df

    sampledata_files = list()
    sampledata_pos_dfs = list()
    sequence_length = None
    total_samples = 0

    # Load all the sampledata files into a list
    for index, fn in enumerate(args.input_sampledata):
        fn = fn.rstrip("\n")
        cur_file = tsinfer.load(fn)
        sampledata_files.append(cur_file)
        total_samples += sampledata_files[index].num_samples
        if sequence_length:
            assert sequence_length == sampledata_files[index].sequence_length
        else:
            sequence_length = sampledata_files[index].sequence_length
        # Create a dataframe of sites from this sampledata file
        sampledata_pos_dfs.append(make_site_df(cur_file, index))

    # Create merged sampledata file
    with tsinfer.SampleData(
        path=args.output, num_flush_threads=2, sequence_length=sequence_length
    ) as samples:
        population_id_map = {}
        # Add populations from each sampledata file
        for sampledata in sampledata_files:
            for pop in sampledata.populations_metadata[:]:
                pop_id = samples.add_population(pop)
                population_id_map[pop["name"]] = pop_id
        for sampledata in sampledata_files:
            for indiv in sampledata.individuals():
                samples.add_individual(
                    metadata=indiv.metadata,
                    population=population_id_map[
                        sampledata.populations_metadata[:][
                            sampledata.samples_population[indiv.id * 2]
                        ]["name"]
                    ],
                    time=indiv.time,
                    ploidy=2,
                )
        for index, df in enumerate(sampledata_pos_dfs):
            if index == 0:
                combined_pos_df = df
                continue
            else:
                combined_pos_df = pd.merge(
                    combined_pos_df,
                    df,
                    how="outer",
                    suffixes=("", index),
                    on=["Position", "Ancestral Allele"],
                )
                combined_pos_df["Derived Allele"] = combined_pos_df[
                    "Derived Allele"
                ].fillna(combined_pos_df["Derived Allele" + str(index)])
                derived_allele_exists = ~pd.isna(
                    combined_pos_df["Derived Allele" + str(index)]
                )
                # multiallelic = combined_pos_df["Derived Allele"] != combined_pos_df["Derived Allele" + str(index)]

        combined_pos_df = combined_pos_df.sort_values(by=["Position"])
        num_sites = combined_pos_df.shape[0]
        num_files = len(sampledata_files)
        combined_genos = np.full(
            (num_sites, total_samples), tskit.MISSING_DATA, dtype=np.int8
        )

        combined_alleles = np.full((num_sites, 2), "")
        combined_alleles = combined_pos_df[
            ["Ancestral Allele", "Derived Allele"]
        ].to_numpy()
        combined_metadata = np.full((num_sites, num_files), {})
        cur_sample = 0
        for index, sampledata in enumerate(sampledata_files):
            cur_file_sites = combined_pos_df[[str(index) + " Site ID"]]
            pos_bool = np.where(~np.isnan(cur_file_sites))[0]
            num_samps = sampledata.num_samples
            if index != 0:
                # Mark sites as missing anywhere they have a "new" derived allele
                derived_allele_exists = ~pd.isna(
                    combined_pos_df["Derived Allele" + str(index)]
                )
                multiallelic = (
                    combined_pos_df["Derived Allele"]
                    != combined_pos_df["Derived Allele" + str(index)]
                )
                multiallelic = np.logical_and(derived_allele_exists, multiallelic)
                genos = sampledata.sites_genotypes[:]
                multiallelic_site_id = cur_file_sites[multiallelic].astype(int)
                multi_genos = genos[multiallelic_site_id]
                multiallelic_derived = np.where(multi_genos == 1)
                multi_genos[multiallelic_derived] = tskit.MISSING_DATA
                genos[multiallelic_site_id] = multi_genos
                combined_genos[pos_bool, cur_sample : cur_sample + num_samps] = genos
            elif index == 0:
                combined_genos[
                    pos_bool, cur_sample : cur_sample + num_samps
                ] = sampledata.sites_genotypes[:]
            combined_metadata[pos_bool, index] = sampledata.sites_metadata[:]
            cur_sample += num_samps

        for row in tqdm.tqdm(
            zip(
                combined_pos_df[["Position"]].values,
                combined_genos,
                combined_alleles,
                combined_metadata,
            ),
            total=np.sum(combined_pos_df.shape[0]),
        ):
            samples.add_site(
                position=float(row[0]),
                genotypes=row[1],
                alleles=list(row[2]),
                metadata=row[3][0],
            )


def make_sampledata_compatible(args):
    """
    Make a list of sampledata files compatible with the first file.
    """
    sampledata_files = list()
    new_names = list()

    # Load all the sampledata files into a list
    print("Subset sites with {} sampledata files".format(len(args.input_sampledata) - 1))
    for index, fn in enumerate(args.input_sampledata):
        fn = fn.rstrip("\n")
        if index == 0:
            target_sd = tsinfer.load(fn)
            print("Loaded First sampledata file")
            continue
        cur_sd = tsinfer.load(fn)
        print("Loaded sampledata file # {}".format(index))
        keep_sites = np.where(
            np.isin(cur_sd.sites_position[:], target_sd.sites_position[:])
        )[0]
        print("Subsetting to {} sites".format(len(keep_sites)))
        small_cur_sd = cur_sd.subset(sites=keep_sites)
        print("Done with subset")
        newname = fn[: -len(".samples")] + ".subset.samples"
        small_cur_sd_copy = small_cur_sd.copy(newname)
        small_cur_sd_copy.finalise()
        sampledata_files.append(small_cur_sd_copy)
        print("Subsetted to {} sites from {}. Output can be found at {}.".format(
            len(keep_sites), fn, newname))


def run_merge_sampledata(args):
    """
    Merge a list of SampleData files. Only add variants appearing in all sampledata files.
    """
    sampledata_files = list()
    sequence_length = None
    total_samples = 0

    with open(args.input, "r") as sampledata_fn:
        for index, fn in enumerate(sampledata_fn):
            fn = fn.rstrip("\n")
            sampledata_files.append(tsinfer.load(fn))
            total_samples += sampledata_files[index].num_samples
            if sequence_length:
                assert sequence_length == sampledata_files[index].sequence_length
            else:
                sequence_length = sampledata_files[index].sequence_length

    with tsinfer.SampleData(
        path=args.output, num_flush_threads=2, sequence_length=sequence_length
    ) as samples:
        population_id_map = {}
        for sampledata in sampledata_files:
            for pop in sampledata.populations_metadata[:]:
                pop_id = samples.add_population({"name": pop["name"]})
                population_id_map[pop["name"]] = pop_id
        for sampledata in sampledata_files:
            for indiv in sampledata.individuals():
                samples.add_individual(
                    metadata=indiv.metadata,
                    population=population_id_map[
                        sampledata.populations_metadata[:][
                            sampledata.samples_population[indiv.id * 2]
                        ]["name"]
                    ],
                    ploidy=2,
                )
        merged_pos = np.sort(
            list(
                set.intersection(
                    *map(
                        set,
                        [
                            sampledata.sites_position[:]
                            for sampledata in sampledata_files
                        ],
                    )
                )
            )
        )
        pos_bool = np.isin(sampledata_files[0].sites_position[:], merged_pos)
        anc_alleles = np.array(
            [allele[0] for allele in sampledata_files[0].sites_alleles[:][pos_bool]]
        )
        deriv_alleles = np.array(
            [
                allele[1] if len(allele) > 1 else ""
                for allele in sampledata_files[0].sites_alleles[:][pos_bool]
            ]
        )
        biallelic_sites = np.full_like(merged_pos, True, dtype=bool)
        for sampledata in sampledata_files[1:]:
            pos_bool = np.isin(sampledata.sites_position[:], merged_pos)
            # Assert ancestral alleles match between the sampledata files
            assert np.array_equal(
                anc_alleles,
                [allele[0] for allele in sampledata.sites_alleles[:][pos_bool]],
            )
            cur_derived = np.array(
                [
                    allele[1] if len(allele) > 1 else ""
                    for allele in sampledata.sites_alleles[:][pos_bool]
                ]
            )
            no_deriv_bool = np.logical_or(
                np.array(deriv_alleles == ""), np.array(cur_derived == "")
            )
            deriv_alleles[no_deriv_bool] = cur_derived[no_deriv_bool]
            # If alt alleles conflict
            biallelic_sites = np.logical_and(
                biallelic_sites, deriv_alleles == cur_derived
            )
        merged_pos = merged_pos[biallelic_sites]
        num_sites = len(merged_pos)
        num_files = len(sampledata_files)
        combined_genos = np.full(
            (num_sites, total_samples), tskit.MISSING_DATA, dtype=np.int8
        )
        combined_metadata = np.full((num_sites, num_files), {})
        cur_sample = 0
        for index, sampledata in enumerate(sampledata_files):
            pos_bool = np.isin(sampledata.sites_position[:], merged_pos)
            num_samps = sampledata.num_samples
            combined_genos[
                :, cur_sample : cur_sample + num_samps
            ] = sampledata.sites_genotypes[:][pos_bool]
            combined_metadata[:, index] = sampledata.sites_metadata[:][pos_bool]
            cur_sample += num_samps

        for row in tqdm.tqdm(
            zip(
                merged_pos,
                combined_genos,
                anc_alleles,
                deriv_alleles,
                combined_metadata,
            ),
            total=np.sum(biallelic_sites),
        ):
            samples.add_site(
                position=row[0],
                genotypes=row[1],
                alleles=[row[2], row[3]],
                metadata=row[4][0],
            )

def add_indiv_times(args):
    """
    Takes samples 'age' in metadata and 
    """
    samples = tsinfer.load(args.input)
    times = samples.individuals_time[:]
    copy = samples.copy(args.output)
    for indiv in samples.individuals():
        if "age" in indiv.metadata:
            times[indiv.id] = int(indiv.metadata["age"])

    copy.individuals_time[:] = times
    copy.finalise()


def merge_sampledata_files(args):
    samples = []
    for cur_sample in args.input_sampledata:
        samples.append(tsinfer.load(cur_sample))
    merged_samples = samples[0]
    for other_samples in samples[1:]:
        intersect_sites = np.isin(merged_samples.sites_position[:], other_samples.sites_position[:])
        other_intersect_sites = np.where(np.isin(other_samples.sites_position[:], merged_samples.sites_position[:]))[0]
        other_samples_metadata = other_samples.sites_metadata[:]
        for site_index, site_metadata in zip(other_intersect_sites, merged_samples.sites_metadata[:][intersect_sites]):
            other_samples_metadata[site_index] = site_metadata
        other_samples_copy = other_samples.copy()
        other_samples_copy.sites_metadata[:] = other_samples_metadata
        other_samples_copy.finalise()
        merged_samples = merged_samples.merge(other_samples_copy)
    merged_copy = merged_samples.copy(args.output)
    merged_copy.finalise()


def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "command"

    subparser = subparsers.add_parser("simplify")
    subparser.add_argument("input", type=str, help="Input tree sequence")
    subparser.add_argument("output", type=str, help="Input tree sequence")
    subparser.set_defaults(func=run_simplify)

    subparser = subparsers.add_parser("sequential-augment")
    subparser.add_argument("input", type=str, help="Input tree sequence")
    subparser.add_argument("--num-threads", type=int, default=0)
    subparser.add_argument("--seed", type=int, default=1)
    subparser.set_defaults(func=run_sequential_augment)

    subparser = subparsers.add_parser("benchmark-tskit")
    subparser.add_argument("input", type=str, help="Input tree sequence")
    subparser.add_argument(
        "--num-variants",
        type=int,
        default=None,
        help="Number of variants to benchmark genotypes decoding performance on",
    )
    subparser.set_defaults(func=run_benchmark_tskit)

    subparser = subparsers.add_parser("benchmark-vcf")
    subparser.add_argument("input", type=str, help="Input VCF")
    subparser.add_argument(
        "--num-variants",
        type=int,
        default=None,
        help="Number of variants to benchmark genotypes decoding performance on",
    )
    subparser.set_defaults(func=run_benchmark_vcf)

    subparser = subparsers.add_parser("compute-ukbb-gnn")
    subparser.add_argument("input", type=str, help="Input tree sequence")
    subparser.add_argument("output", type=str, help="Filename to write CSV to.")
    subparser.add_argument("--num-threads", type=int, default=16)
    subparser.set_defaults(func=run_compute_ukbb_gnn)

    subparser = subparsers.add_parser("compute-1kg-gnn")
    subparser.add_argument("input", type=str, help="Input tree sequence")
    subparser.add_argument("output", type=str, help="Filename to write CSV to.")
    subparser.add_argument("--num-threads", type=int, default=16)
    subparser.set_defaults(func=run_compute_1kg_gnn)

    subparser = subparsers.add_parser("compute-sgdp-gnn")
    subparser.add_argument("input", type=str, help="Input tree sequence")
    subparser.add_argument("output", type=str, help="Filename to write CSV to.")
    subparser.add_argument("--num-threads", type=int, default=16)
    subparser.set_defaults(func=run_compute_sgdp_gnn)

    subparser = subparsers.add_parser("compute-hgdp-gnn")
    subparser.add_argument("input", type=str, help="Input tree sequence")
    subparser.add_argument("output", type=str, help="Filename to write CSV to.")
    subparser.add_argument("--num-threads", type=int, default=16)
    subparser.set_defaults(func=run_compute_hgdp_gnn)

    subparser = subparsers.add_parser("snip-centromere")
    subparser.add_argument("input", type=str, help="Input tree sequence")
    subparser.add_argument("output", type=str, help="Output tree sequence")
    subparser.add_argument("chrom", type=str, help="Chromosome name")
    subparser.add_argument(
        "centromeres", type=str, help="CSV file containing centromere coordinates."
    )
    subparser.set_defaults(func=run_snip_centromere)

    subparser = subparsers.add_parser("combine-sampledata")
    subparser.add_argument(
        "--input-sampledata",
        type=str,
        nargs='+',
        help="Input sample files to merge.",
        required=True
    )
    subparser.add_argument("--output", type=str, required=True, help="Output combined sample data file")
    subparser.set_defaults(func=run_combine_sampledata)

    subparser = subparsers.add_parser("merge-sampledata")
    subparser.add_argument(
        "input",
        type=str,
        help="Input sample files to merge. \
            The path to each SampleData file should be a row in a text file.",
    )
    subparser.add_argument("output", type=str, help="Output merged sample data file")
    subparser.set_defaults(func=run_merge_sampledata)

    subparser = subparsers.add_parser("make-sampledata-compatible")
    subparser.add_argument(
        "--input-sampledata",
        type=str,
        nargs='+',
        help="Input sample files to merge.",
        required=True
    )
    subparser.set_defaults(func=make_sampledata_compatible)

    subparser = subparsers.add_parser("output-indiv-times")
    subparser.add_argument(
        "input",
        type=str,
        help="Add individuals times to sampledata file.",
    )
    subparser.add_argument(
        "output",
        type=str,
        help="Add individuals times to sampledata file.",
    )
    subparser.set_defaults(func=add_indiv_times)

    subparser = subparsers.add_parser("merge-sampledata-files")
    subparser.add_argument(
        "--input-sampledata",
        type=str,
        nargs='+',
        help="Input sample files to merge.",
        required=True
    )
    subparser.add_argument(
        "--output",
        type=str,
        required=True
    )
    subparser.set_defaults(func=merge_sampledata_files)


    daiquiri.setup(level="INFO")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

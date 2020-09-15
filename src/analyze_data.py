"""
Analyses real data in all-data directory. Outputs csvs for
plotting in plot.py
"""
import argparse
import os.path
import json
import collections

import numpy as np
import pandas as pd

import sys

import tskit
import tsinfer
import tqdm

import argparse
import pandas as pd
import numpy as np
import allel
import pickle
import os.path
from os import path
import tsinfer
import tskit
import sys
from tqdm import tqdm

import tsdate

data_prefix = "all-data"

def get_relate_tgp_age_df():
#    if path.exists("all-data/1kg_chr20_relate_mutation_ages_geometric.csv"):
#        relate_ages = pd.read_csv("all-data/1kg_chr20_relate_mutation_ages_geometric.csv", index_col=0)
#    else:
    relate_ts = tskit.load("/home/jk/large_files/relate/relate_chr20_metdata.trees")
    relate_mut_ages, relate_mut_upper_bound, mut_ids = get_mut_ages(relate_ts,
            unconstrained=False, geometric=False)
    relate_frequencies = get_site_frequencies(relate_ts)
    relate_ages = pd.DataFrame(
        {
            "position": relate_ts.tables.sites.position[
                relate_ts.tables.mutations.site
            ][mut_ids],
            "relate_age": relate_mut_ages,
            "relate_upper_bound": relate_mut_upper_bound,
            "relate_ancestral_allele": np.array(tskit.unpack_strings(relate_ts.tables.sites.ancestral_state,
                relate_ts.tables.sites.ancestral_state_offset))[relate_ts.tables.mutations.site][mut_ids],
            "relate_derived_allele": np.array(tskit.unpack_strings(relate_ts.tables.mutations.derived_state,
                relate_ts.tables.mutations.derived_state_offset))[mut_ids],
            "relate_frequency": relate_frequencies
        }
    )
    relate_ages.to_csv("all-data/1kg_chr20_relate_mutation_ages_geometric.csv")
    return relate_ages

def get_geva_tgp_age_df():
    if path.exists("all-data/1kg_chr20_geva_mutation_ages.csv"):
        geva_ages = pd.read_csv("all-data/1kg_chr20_geva_mutation_ages.csv", index_col=0)
    else:
        geva = pd.read_csv(
            "/home/wilderwohns/tsinfer_geva/atlas.chr20.csv.gz",
            delimiter=",",
            skipinitialspace=True,
            skiprows=3,
        )
        geva_tgp = geva[geva["DataSource"] == "TGP"]
        geva_tgp_consistent = geva_tgp[geva_tgp["AlleleAnc"] == geva_tgp["AlleleRef"]]
        geva_ages = geva_tgp_consistent[["Position", "AgeMean_Jnt", "AgeCI95Upper_Jnt", "AlleleRef", "AlleleAlt"]]
        geva_ages.to_csv("all-data/1kg_chr20_geva_mutation_ages.csv")
    return geva_ages

def get_tsdate_tgp_age_df():
    if path.exists("all-data/tsdate_ages_1kg_chr20.iter.dated.binned_ma0.1_ms0.1_NNone_p16.simplified.dated.insideoutside.trees"):
        tsdate_ages = pd.read_csv("all-data/tsdate_ages_1kg_chr20.iter.dated.binned_ma0.1_ms0.1_NNone_p16.simplified.dated.insideoutside.trees", index_col=0)

    else:

        tgp_chr20 = tskit.load(
                "all-data/1kg_chr20.iter.dated.binned_ma0.1_ms0.1_NNone_p16.simplified.dated.insideoutside.trees")
        
        posterior_mut_ages, posterior_upper_bound, oldest_mut_nodes = get_mut_ages(
            tgp_chr20, unconstrained=False
        )
        site_frequencies = get_site_frequencies(tgp_chr20)
        tsdate_ages = pd.DataFrame(
            {
                "position": tgp_chr20.tables.sites.position,
                "tsdate_age": posterior_mut_ages,
                "tsdate_upper_bound": posterior_upper_bound,
                "tsdate_frequency": site_frequencies,
                "tsdate_ancestral_allele": np.array(tskit.unpack_strings(tgp_chr20.tables.sites.ancestral_state,
                    tgp_chr20.tables.sites.ancestral_state_offset)),
                "tsdate_derived_allele": np.array(tskit.unpack_strings(tgp_chr20.tables.mutations.derived_state,
                    tgp_chr20.tables.mutations.derived_state_offset))[oldest_mut_nodes]
            }
        )
        tsdate_ages.to_csv(
                "all-data/tsdate_ages_1kg_chr20.iter.dated.binned_ma0.1_ms0.1_NNone_p16.simplified.dated.insideoutside.trees")
    return tsdate_ages


def tgp_date_estimates():
    """
    Produce comparable set of mutations from tgp
    """
    tsdate_ages = get_tsdate_tgp_age_df()
    geva = get_geva_tgp_age_df()
    merged = pd.merge(tsdate_ages,
            geva, 
            left_on=["position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
            right_on=["Position", "AlleleRef", "AlleleAlt"])
    relate_ages = get_relate_tgp_age_df()
    merged = pd.merge(merged, relate_ages,
            left_on=["Position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
            right_on=["position", "relate_ancestral_allele", "relate_derived_allele"])
    merged = merged[np.abs(merged["tsdate_frequency"] - merged["relate_frequency"]) < 0.5]
    merged = merged.drop(columns=["position_x", "position_y"])
    merged.to_csv("all-data/tgp_mutations.csv")

def get_site_frequencies(ts):
    """
    Calculate frequency of each site and return numpy 1d array of len num_mutations 
    with frequency as values. This assumes that if there are multiple mutations at a 
    site they are recurrent.
    """
    site_freq = np.zeros(ts.num_sites)
    for var in tqdm(ts.variants(), total=ts.num_sites, desc="Get mutation frequencies"):
        site_freq[var.site.id] = np.sum(var.genotypes) / ts.num_samples
    return site_freq 

def get_mut_ages(ts, unconstrained=True, ignore_sample_muts=False, geometric=True):
    mut_ages = np.zeros(ts.num_sites)
    mut_upper_bounds = np.zeros(ts.num_sites)
    node_ages = ts.tables.nodes.time
    oldest_mut_ids = np.zeros(ts.num_sites)
    if unconstrained:
        metadata = ts.tables.nodes.metadata[:]
        metadata_offset = ts.tables.nodes.metadata_offset[:]
        for index, met in enumerate(tskit.unpack_bytes(metadata, metadata_offset)):
            if index not in ts.samples():
                node_ages[index] = json.loads(met.decode())["mn"]
    if ignore_sample_muts:
        mutations_table = ts.tables.mutations
        unique_sites = np.unique(ts.tables.mutations.site, return_counts=True)
        unique_sites = unique_sites[0][unique_sites[1] > 1]
        no_samp_muts = ~np.logical_and(np.isin(mutations_table.site, unique_sites),
                np.isin(mutations_table.node, ts.samples()))
    for tree in tqdm(ts.trees(), total=ts.num_trees, desc="Finding mutation ages"):
        for site in tree.sites():
            for mut in site.mutations:
                parent_age = node_ages[tree.parent(mut.node)]
                if geometric:
                    age = np.sqrt(node_ages[mut.node] * parent_age)
                else:
                    age = (node_ages[mut.node] + parent_age) / 2
                if mut_ages[site.id] < age:
                    mut_upper_bounds[site.id] = parent_age
                    mut_ages[site.id] = age
                    oldest_mut_ids[site.id] = mut.id
    return mut_ages, mut_upper_bounds, oldest_mut_ids.astype(int)


def get_ancient_constraints_tgp():
    if path.exists("all-data/1kg_ancients_only_chr20.samples"):
        ancient_samples = tsinfer.load("all-data/1kg_ancients_only_chr20.samples")
    else:
        ancient_samples = tsinfer.load("all-data/1kg_ancients_chr20.samples")
        print("Subsetting SampleData file to only keep ancient samples")
        ancient_indiv_ids = np.where(ancient_samples.individuals_time[:] != 0)[0]
        ancient_sample_ids = np.where(ancient_samples.individuals_time[:][ancient_samples.samples_individual] != 0)[0]
        ancient_genos = ancient_samples.sites_genotypes[:]
        ancient_sites = np.where(np.any(ancient_genos[:, ancient_sample_ids] == 1, axis=1))[0]
        ancient_samples = ancient_samples.subset(individuals=ancient_indiv_ids,
                sites=ancient_sites)
        copy = ancient_samples.copy("all-data/1kg_ancients_only_chr20.samples")
        copy.finalise() 
        print("Subsetted to {} samples and {} sites".format(
            ancient_samples.num_samples, ancient_samples.num_sites))
    genotypes = ancient_samples.sites_genotypes[:]
    positions = ancient_samples.sites_position[:]
    alleles = ancient_samples.sites_alleles[:]
    min_site_times = ancient_samples.min_site_times(individuals_only=True)
    lower_bound = [
            (pos, allele[0], allele[1], age, np.sum(geno[geno==1]))
            for pos, allele, age, geno in zip(positions, alleles, min_site_times, genotypes)
    ]
    constraint_df = pd.DataFrame(lower_bound, columns=["Position", "Reference Allele",
        "Alternative Allele", "Ancient Bound", "Number of Ancients"])
    constraint_df = constraint_df.astype({"Position": "int64", "Ancient Bound": "float64", "Number of Ancients": "int32"})
    constraint_df = constraint_df[constraint_df["Ancient Bound"] != 0]
    #constraint_df = constraint_df.set_index("Position")
    constraint_df.to_csv("all-data/ancient_constraints.csv")
    try:
        #tgp_mut_ests = pd.read_csv("all-data/tgp_mutations_allmethods_new_snipped_allsites.csv", index_col=0)
        tgp_mut_ests = pd.read_csv("all-data/tgp_mutations.csv", index_col=0)
    except:
        raise ValueError("tgp_mutations.csv does not exist. Must run tgp_dates first")
    tgp_muts_constraints = pd.merge(
        tgp_mut_ests, constraint_df, how="left",
        left_on=["Position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
        right_on=["Position", "Reference Allele", "Alternative Allele"])
    tgp_muts_constraints.to_csv("all-data/tgp_muts_constraints.csv")

def get_recurrent_mutations(ts):
    """
    Get number of mutations per site.
    """
    mutations_sites = ts.tables.mutations.site
    muts_per_site = np.unique(mutations_sites, return_counts=True)[1]
    sites_by_muts = np.unique(muts_per_site, return_counts=True)[1]

    # Exclude mutations above samples, this is simplier as there are no singletons
    mutations_sites = mutations_sites[~np.isin(ts.tables.mutations.node, ts.samples())]
    muts_per_site = np.unique(mutations_sites, return_counts=True)[1]
    sites_by_muts_nosamples = np.unique(muts_per_site, return_counts=True)[1]
    # Exclude mutations above two samples
    muts_non_sample_edge = np.where(~np.isin(ts.tables.mutations.node, ts.samples()))[0]
    muts_non_sample_edge_set = set(muts_non_sample_edge)
    mut_sites_nodouble = list()
    for tree in ts.trees():
        for site in tree.sites():
            for mutation in site.mutations:
                if tree.num_samples(mutation.node) > 2:
                    mut_sites_nodouble.append(mutation.site)
    unique_sites_counts_nodouble = np.unique(np.array(mut_sites_nodouble), return_counts=True)[1]
    sites_by_muts_nodouble = np.unique(unique_sites_counts_nodouble, return_counts=True)[1]

    # Tips below mutations at sites with two mutations
    two_mutations_id = set(muts_non_sample_edge[np.where(np.unique(mutations_sites, return_counts=True)[1] == 2)[0]])
    num_samples_muts = list()
    for tree in ts.trees():
        for site in tree.sites():
            for mutation in site.mutations:
                if mutation.id in two_mutations_id:
                    num_samples_muts.append(tree.num_samples(mutation.node))

    return sites_by_muts, sites_by_muts_nosamples, sites_by_muts_nodouble, num_samples_muts

def get_mutations_by_sample():
    """
    See number of mutations above each sample
    """


def get_tgp_recurrent_mutations():
    #filename = os.path.join(data_prefix, "1kg_chr20_ma0.1_ms0.01_p13.simplify.trees")
    filename = os.path.join(
            data_prefix,
            "1kg_chr20.iter.dated.binned_ma0.1_ms0.1_NNone_p16.simplified.dated.insideoutside.trees")

    ts = tskit.load(filename)
    recurrent_counts, recurrent_counts_nosamples, sites_by_muts_nodouble, recurrent_counts_two_muts = get_recurrent_mutations(ts)
    df = pd.DataFrame(recurrent_counts, columns=["recurrent_counts"])
    df.to_csv("data/1kg_chr20_ma0.1_ms0.1_p16.recurrent_counts.csv")
    df = pd.DataFrame(recurrent_counts_nosamples, columns=["recurrent_counts_nosamples"])
    df.to_csv("data/1kg_chr20_ma0.1_ms0.1_p16.recurrent_counts_nosamples.csv")
    df = pd.DataFrame(sites_by_muts_nodouble, columns=["recurrent_counts_nodouble"])
    df.to_csv("data/1kg_chr20_ma0.1_ms0.1_p16.recurrent_counts_nodouble.csv")

    df = pd.DataFrame(recurrent_counts_two_muts, columns=["recurrent_counts_two_muts"])
    df.to_csv("data/1kg_chr20_ma0.1_ms0.1_p16.recurrent_counts_nosamples_two_muts.csv")

def get_hgdp_recurrent_mutations():
    filename = os.path.join(data_prefix, "hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.trees")

    ts = tskit.load(filename)
    recurrent_counts, recurrent_counts_nosamples, sites_by_muts_nodouble, recurrent_counts_two_muts = get_recurrent_mutations(ts)
    df = pd.DataFrame(recurrent_counts, columns=["recurrent_counts"])
    df.to_csv("data/hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.recurrent_counts.csv")
    df = pd.DataFrame(recurrent_counts_nosamples, columns=["recurrent_counts_nosamples"])
    df.to_csv("data/hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.recurrent_counts_nosamples.csv")
    df = pd.DataFrame(sites_by_muts_nodouble, columns=["recurrent_counts_nodouble"])
    df.to_csv("data/hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.recurrent_counts_nodouble.csv")

    df = pd.DataFrame(recurrent_counts_two_muts, columns=["recurrent_counts_two_muts"])
    df.to_csv("data/hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.recurrent_counts_nosamples_two_muts.csv")

def get_sgdp_recurrent_mutations():
    filename = os.path.join(data_prefix, "sgdp_chr20.tsinferred.trees")

    ts = tskit.load(filename)
    recurrent_counts, recurrent_counts_nosamples, sites_by_muts_nodouble, recurrent_counts_two_muts = get_recurrent_mutations(ts)
    df = pd.DataFrame(recurrent_counts, columns=["recurrent_counts"])
    df.to_csv("data/sgdp_chr20.tsinferred.recurrent_counts.csv")
    df = pd.DataFrame(recurrent_counts_nosamples, columns=["recurrent_counts_nosamples"])
    df.to_csv("data/sgdp_chr20.tsinferred.recurrent_counts_nosamples.csv")
    df = pd.DataFrame(sites_by_muts_nodouble, columns=["recurrent_counts_nodouble"])
    df.to_csv("data/sgdp_chr20.tsinferred.recurrent_counts_nodouble.csv")

    df = pd.DataFrame(recurrent_counts_two_muts, columns=["recurrent_counts_two_muts"])
    df.to_csv("data/sgdp_chr20.tsinferred.recurrent_counts_nosamples_two_muts.csv")

def min_site_times_ancients():
    samples = tsinfer.load("all-data/1kg_ancients_noreich_chr20.samples")
    min_times = samples.min_site_times(individuals_only=True)
    df = pd.DataFrame(np.unique(min_times, return_counts=True))
    df.to_csv("data/1kg_ancients_chr20_min_site_times.csv")


def main():
    name_map = {
        "recurrent_mutations_tgp": get_tgp_recurrent_mutations,
        "recurrent_mutations_hgdp": get_hgdp_recurrent_mutations,
        "recurrent_mutations_sgdp": get_sgdp_recurrent_mutations,
        "tgp_dates": tgp_date_estimates,
        "ancient_constraints": get_ancient_constraints_tgp,
        "min_site_times_ancients": min_site_times_ancients
    }

    parser = argparse.ArgumentParser(
        description="Process the human data and make data files for plotting.")
    parser.add_argument(
        "name", type=str, help="figure name", choices=list(name_map.keys()))

    args = parser.parse_args()
    name_map[args.name]()


if __name__ == "__main__":
    main()

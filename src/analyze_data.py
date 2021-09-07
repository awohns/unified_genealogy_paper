"""
Analyses real data in all-data and data directories. Outputs csvs for
plotting in plot.py
"""
import argparse
import collections
import csv
import os.path
import json
import itertools
import operator
import pickle
import subprocess

import numpy as np
import pandas as pd
import pyreadr

import tskit
import tsinfer
import tsdate

from tqdm import tqdm

import utility
import tmrcas

data_prefix = "all-data"


def get_ancient_proxy_nodes(ts):
    ancient_proxy_nodes = {}
    ancient_proxy_nodes["Altai"] = np.where(ts.tables.nodes.time == 4400.04)[0]
    ancient_proxy_nodes["Chagyrskaya"] = np.where(ts.tables.nodes.time == 3200.04)[0]
    ancient_proxy_nodes["Denisovan"] = np.where(ts.tables.nodes.time == 2556.04)[0]
    ancient_proxy_nodes["Vindija"] = np.where(ts.tables.nodes.time == 2000.04)[0]
    ancient_proxy_nodes["Afanasievo"] = np.concatenate(
        [
            np.where(ts.tables.nodes.time == 183.60)[0],
            np.where(ts.tables.nodes.time == 184.60)[0],
        ]
    )
    ancient_pop_indices = {}
    ancient_pop_names = ["Altai", "Chagyrskaya", "Denisovan", "Vindija", "Afanasievo"]
    for index, pop in enumerate(ts.populations()):
        name = json.loads(pop.metadata.decode())["name"]
        if name in ancient_pop_names:
            ancient_pop_indices[index] = ancient_proxy_nodes[name]

    pop_nodes = ts.tables.nodes.population
    # Double check proxy nodes are parents of ancient samples for whole tree sequence
    for ancient_pop, ancient_proxy_node in ancient_pop_indices.items():
        for node in np.where(pop_nodes == ancient_pop)[0]:
            parent_node = -9
            for tree in tqdm(ts.trees()):
                cur_parent = tree.parent(node)
                if parent_node == -9 and cur_parent != -1:  # First tree with parent
                    assert cur_parent in ancient_proxy_node
                    parent_node = cur_parent
                elif cur_parent != -1:  # Subsequent tree with parent
                    assert parent_node == cur_parent, cur_parent
                elif cur_parent == -1:  # Parent is -1, we've reached a snipped tree
                    parent_node = -9
                else:
                    raise ValueError("Incorrect relationship between proxy and ancient")
    return (
        ancient_proxy_nodes["Altai"],
        ancient_proxy_nodes["Chagyrskaya"],
        ancient_proxy_nodes["Denisovan"],
        ancient_proxy_nodes["Vindija"],
        ancient_proxy_nodes["Afanasievo"],
    )


def get_relate_tgp_age_df():
    if os.path.exists("data/1kg_chr20_relate_mutation_ages_all_pops.csv"):
        relate_ages = pd.read_csv(
            "data/1kg_chr20_relate_mutation_ages_all_pops.csv", index_col=0
        )
    else:
        relate_ages = None
        for region in tqdm(["AFR", "AMR", "EAS", "EUR", "SAS"], desc="Region"):
            path = "data/allele_ages_" + region
            for f in tqdm(os.listdir(path), desc="Population"):
                pop = f[-10:-6]
                keep_columns = [
                    "CHR",
                    "BP",
                    "ID",
                    "ancestral/derived",
                    "DAF",
                    "est" + pop,
                    "upper_age" + pop,
                ]
                df = pyreadr.read_r(os.path.join(path, f))["allele_ages"]
                df = df[df["CHR"] == 20]
                df["est" + pop] = (df["lower_age"] + df["upper_age"]) * 0.5
                df.rename(columns={"upper_age": "upper_age" + pop}, inplace=True)
                df = df[keep_columns]
                if relate_ages is None:
                    relate_ages = df
                else:
                    relate_ages = relate_ages.merge(
                        df,
                        how="outer",
                        on=["CHR", "BP", "ID", "ancestral/derived"],
                        suffixes=["", "_" + f[-9:-6]],
                    )
        daf_cols = [c for c in relate_ages.columns if "DAF" in c]
        relate_ages["relate_daf_sum"] = relate_ages[daf_cols].sum(axis=1)
        relate_ages["relate_ancestral_allele"] = relate_ages["ancestral/derived"].str[0]
        relate_ages["relate_derived_allele"] = relate_ages["ancestral/derived"].str[2]
        relate_ages.to_csv("data/1kg_chr20_relate_mutation_ages_all_pops.csv")
    return relate_ages


def get_geva_tgp_age_df():
    if os.path.exists("data/1kg_chr20_geva_mutation_ages.csv"):
        geva_ages = pd.read_csv("data/1kg_chr20_geva_mutation_ages.csv", index_col=0)
    else:
        geva = pd.read_csv(
            "data/geva_ages.csv.gz",
            delimiter=",",
            skipinitialspace=True,
            skiprows=3,
        )
        geva_tgp = geva[geva["DataSource"] == "TGP"]
        geva_tgp_consistent = geva_tgp[geva_tgp["AlleleAnc"] == geva_tgp["AlleleRef"]]
        geva_ages = geva_tgp_consistent[
            ["Position", "AgeMean_Jnt", "AgeCI95Upper_Jnt", "AlleleRef", "AlleleAlt"]
        ]
        geva_ages.to_csv("data/1kg_chr20_geva_mutation_ages.csv")
    return geva_ages


def get_tsdate_tgp_age_df():
    if os.path.exists("data/1kg_chr20_tsdate_mutation_ages.csv"):
        tsdate_ages = pd.read_csv(
            "data/1kg_chr20_tsdate_mutation_ages.csv", index_col=0
        )
    else:
        tgp_chr20 = tskit.load("all-data/1kg_chr20.dated.trees")
        posterior_mut_ages, posterior_upper_bound, oldest_mut_nodes = get_mut_ages(
            tgp_chr20, unconstrained=False, geometric=False
        )
        keep_sites = ~np.isnan(posterior_mut_ages)
        site_frequencies = get_site_frequencies(tgp_chr20)[keep_sites]
        tsdate_ages = pd.DataFrame(
            {
                "Position": tgp_chr20.tables.sites.position[keep_sites],
                "tsdate_age": posterior_mut_ages[keep_sites],
                "tsdate_upper_bound": posterior_upper_bound[keep_sites],
                "tsdate_frequency": site_frequencies,
                "tsdate_ancestral_allele": np.array(
                    tskit.unpack_strings(
                        tgp_chr20.tables.sites.ancestral_state,
                        tgp_chr20.tables.sites.ancestral_state_offset,
                    )
                )[keep_sites],
                "tsdate_derived_allele": np.array(
                    tskit.unpack_strings(
                        tgp_chr20.tables.mutations.derived_state,
                        tgp_chr20.tables.mutations.derived_state_offset,
                    )
                )[oldest_mut_nodes[keep_sites]],
            }
        )
        tsdate_ages.to_csv("data/1kg_chr20_tsdate_mutation_ages.csv")
    return tsdate_ages


def tgp_date_estimates(args):
    """
    Produce comparable set of mutations from tgp
    """
    tsdate_ages = get_tsdate_tgp_age_df()
    geva = get_geva_tgp_age_df()
    merged = pd.merge(
        tsdate_ages,
        geva,
        left_on=["Position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
        right_on=["Position", "AlleleRef", "AlleleAlt"],
    )
    relate_ages = get_relate_tgp_age_df()
    merged = pd.merge(
        merged,
        relate_ages,
        left_on=["Position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
        right_on=["BP", "relate_ancestral_allele", "relate_derived_allele"],
    )
    # In Relate, quoted number of haplotypes in 1000G is 4956.
    # We check that the frequency of the variants in tsdate and Relate are both < 0.5
    merged = merged[
        np.abs(merged["tsdate_frequency"] - (merged["relate_daf_sum"] / 4956)) < 0.5
    ]
    merged.to_csv("data/tgp_mutations.csv")


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
    # Get age of oldest mutations associated with site, ignoring muts below oldest root
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
        unique_sites = np.unique(ts.tables.mutations.site, return_counts=True)
        unique_sites = unique_sites[0][unique_sites[1] > 1]
    for tree in tqdm(ts.trees(), total=ts.num_trees, desc="Finding mutation ages"):
        for site in tree.sites():
            for mut in site.mutations:
                parent_age = node_ages[tree.parent(mut.node)]
                if geometric:
                    age = np.sqrt(node_ages[mut.node] * parent_age)
                else:
                    age = (node_ages[mut.node] + parent_age) / 2
                if mut_ages[site.id] < age:
                    if tree.parent(mut.node) == ts.num_nodes - 1:
                        mut_upper_bounds[site.id] = np.nan
                        mut_ages[site.id] = np.nan
                        oldest_mut_ids[site.id] = np.nan
                    else:
                        mut_upper_bounds[site.id] = parent_age
                        mut_ages[site.id] = age
                        oldest_mut_ids[site.id] = mut.id
    return mut_ages, mut_upper_bounds, oldest_mut_ids.astype(int)


def calc_ancient_constraints_tgp(args):
    if os.path.exists("all-data/all_ancients_chr" + args.chrom + ".samples"):
        ancient_samples = tsinfer.load(
            "all-data/all_ancients_chr" + args.chrom + ".samples"
        )
    else:
        raise FileNotFoundError(
            "Must create all_ancients_chr"
            + args.chrom
            + ".samples using all-data/Makefile"
        )
    genotypes = ancient_samples.sites_genotypes[:]
    positions = ancient_samples.sites_position[:]
    alleles = ancient_samples.sites_alleles[:]
    min_site_times = ancient_samples.min_site_times(individuals_only=True)
    lower_bound = [
        (pos, allele[0], allele[1], age, np.sum(geno[geno == 1]))
        for pos, allele, age, geno in zip(positions, alleles, min_site_times, genotypes)
    ]
    constraint_df = pd.DataFrame(
        lower_bound,
        columns=[
            "Position",
            "Reference Allele",
            "Alternative Allele",
            "Ancient Bound",
            "Number of Ancients",
        ],
    )
    constraint_df = constraint_df.astype(
        {"Position": "int64", "Ancient Bound": "float64", "Number of Ancients": "int32"}
    )
    constraint_df = constraint_df[constraint_df["Ancient Bound"] != 0]
    constraint_df.to_csv("data/ancient_constraints.csv")
    try:
        tgp_mut_ests = pd.read_csv("data/tgp_mutations.csv", index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(
            "tgp_mutations.csv does not exist. Must run tgp_dates first"
        )
    tgp_muts_constraints = pd.merge(
        tgp_mut_ests,
        constraint_df,
        how="left",
        left_on=["Position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
        right_on=["Position", "Reference Allele", "Alternative Allele"],
    )
    tgp_muts_constraints.to_csv("data/tgp_muts_constraints.csv")


def get_unified_recurrent_mutations(ts):
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
    mut_sites_nodouble = list()
    for tree in ts.trees():
        for site in tree.sites():
            for mutation in site.mutations:
                if tree.num_samples(mutation.node) > 2:
                    mut_sites_nodouble.append(mutation.site)
    unique_sites_counts_nodouble = np.unique(
        np.array(mut_sites_nodouble), return_counts=True
    )[1]
    sites_by_muts_nodouble = np.unique(
        unique_sites_counts_nodouble, return_counts=True
    )[1]

    # Tips below mutations at sites with two mutations
    two_mutations_id = set(
        muts_non_sample_edge[
            np.where(np.unique(mutations_sites, return_counts=True)[1] == 2)[0]
        ]
    )
    num_samples_muts = list()
    for tree in ts.trees():
        for site in tree.sites():
            for mutation in site.mutations:
                if mutation.id in two_mutations_id:
                    num_samples_muts.append(tree.num_samples(mutation.node))

    return (
        sites_by_muts,
        sites_by_muts_nosamples,
        sites_by_muts_nodouble,
        num_samples_muts,
    )


def calc_unified_recurrent_mutations(args):
    """
    Get recurrent mutations from the unified tree sequence
    """
    ts = tskit.load(
        "all-data/hgdp_1kg_sgdp_high_cov_ancients_chr" + args.chrom + ".dated.trees"
    )

    (
        recurrent_counts,
        recurrent_counts_nosamples,
        sites_by_muts_nodouble,
        recurrent_counts_two_muts,
    ) = get_unified_recurrent_mutations(ts)
    df = pd.DataFrame(recurrent_counts, columns=["recurrent_counts"])
    df.to_csv("data/unified_chr" + args.chrom + ".recurrent_counts.csv")
    df = pd.DataFrame(
        recurrent_counts_nosamples, columns=["recurrent_counts_nosamples"]
    )
    df.to_csv("data/unified_chr" + args.chrom + ".recurrent_counts_nosamples.csv")
    df = pd.DataFrame(sites_by_muts_nodouble, columns=["recurrent_counts_nodouble"])
    df.to_csv("data/unified_chr" + args.chrom + ".recurrent_counts_nodouble.csv")

    df = pd.DataFrame(recurrent_counts_two_muts, columns=["recurrent_counts_two_muts"])
    df.to_csv(
        "data/unified_chr" + args.chrom + ".recurrent_counts_nosamples_two_muts.csv"
    )

def calc_simulated_recurrent_mutations(args):
    """
    Look at the simulated OOA data as well as some of the real TGP/HGDP data
    used to test mismatch, and get distributions of numbers of mutations at a site
    """
    sim = f"OutOfAfrica_3G09_chr{args.chrom}_n1500_seed1"
    orig_sample_data = tsinfer.load(f"data/{sim}.samples")
    ts = tskit.load(f"data/{sim}.trees")
    err_sample_data = tsinfer.load(f"data/{sim}_ae0.01.samples")
    inferred_ts = tskit.load(f"data/{sim}_ae0.01_rma1_rms1_pNone.trees")
    hgdp_ts = tskit.load("data/hgdp_chr20_1000000-1100000_rma1_rms1_pNone.trees")
    tgp_ts = tskit.load("data/1kg_chr20_1000000-1100000_rma1_rms1_pNone.trees")

    # Change if you want to see the effect of removing mutations above n tips
    ntips_below_err_muts = 3

    data = {}

    assert orig_sample_data.num_sites == err_sample_data.num_sites

    
    data['true_trees'] = {
        'nmuts': np.zeros(ts.num_sites, dtype=int),
        'rm_nmuts': np.zeros((ts.num_sites, ntips_below_err_muts), dtype=int),
    }
    
    tree_iter = ts.trees()
    tree = next(tree_iter)
    for v_err, v in zip(err_sample_data.variants(), ts.variants()):
        while v_err.site.position >= tree.interval.right:
            tree = next(tree_iter)
        anc_state, muts = tree.map_mutations(
            genotypes=v_err.genotypes,
            alleles=v_err.alleles,
            ancestral_state=v_err.site.ancestral_state,
        )
        num_mutations = len(muts)
        data['true_trees']['nmuts'][v_err.site.id] = num_mutations
        for n in range(ntips_below_err_muts):
            if num_mutations < 2:
                data['true_trees']['rm_nmuts'][v_err.site.id, n] = num_mutations
            else:
                mutation_parents = set([tskit.NULL])
                to_delete = []
                for m in muts:
                    if tree.num_samples(m.node) > n+1:
                        mutation_parents.add(m.parent)
                        data['true_trees']['rm_nmuts'][v_err.site.id, n] += 1
                    else:
                        to_delete.append(m)
                # Here we could loop over to_delete and collect information about
                # whether deleting this mutation corrects an error in v.genotypes
                # e.g. for mut in [m for m in to_delete if m.parent in mutation_parents]:

    data['inf_trees'] = {
        'nmuts': np.zeros(inferred_ts.num_sites, dtype=int),
        'rm_nmuts': np.zeros((inferred_ts.num_sites, ntips_below_err_muts), dtype=int),
    }    
    
    assert inferred_ts.num_sites == ts.num_sites
    
    vars_iter = ts.variants()
    err_vars_iter = inferred_ts.variants()
    correctly_changed = np.zeros(ntips_below_err_muts)
    incorrectly_changed = np.zeros(ntips_below_err_muts)
    unexpected = np.zeros(ntips_below_err_muts)
    derived_states = [m.derived_state for m in inferred_ts.mutations()]
    samples = inferred_ts.samples()
    
    for tree in inferred_ts.trees():
        for site in tree.sites():
            v = next(vars_iter)
            v_err = next(err_vars_iter)
            muts = site.mutations
            num_mutations = len(muts)
            data['inf_trees']['nmuts'][site.id] = num_mutations
            for n in range(ntips_below_err_muts):
                if num_mutations < 2:
                    data['inf_trees']['rm_nmuts'][site.id, n] = num_mutations
                else:
                    mutation_parents = set([tskit.NULL])
                    to_delete = []
                    for m in muts:
                        if tree.num_samples(m.node) > n+1:
                            mutation_parents.add(m.parent)
                            data['inf_trees']['rm_nmuts'][site.id, n] += 1
                        else:
                            to_delete.append(m)
                    for mut in [m for m in to_delete if m.parent in mutation_parents]:
                        if mut.parent == tskit.NULL:
                            state = site.ancestral_state
                        else:
                            state = derived_states[mut.parent]
                        changed = np.isin(samples, [s for s in tree.samples(mut.node)])
                        for g, ge in zip(v.genotypes[changed], v_err.genotypes[changed]):
                            if v.alleles[g] == state and v_err.alleles[ge] != state:
                                correctly_changed[n] += 1
                            elif (
                                v.alleles[g] == mut.derived_state
                                and v.alleles[g] != state
                            ):
                                incorrectly_changed[n] += 1
                            else:
                                # this can happen with a mutation over the root
                                if v.alleles[g] == v_err.alleles[ge]:
                                    unexpected[n] += 1
                                else:
                                    raise ValueError(
                                        f"Unexpected mutation {mut} at site {v.site}: "
                                        f"state changed to {state}, was originally "
                                        f"{v.alleles[g]} inferred as {v_err.alleles[ge]}"
                                    )
    for i, percent_correct in enumerate(
        correctly_changed / (correctly_changed + incorrectly_changed) * 100
    ):
        print(
            percent_correct,
            "% of mutations above",
            i+1,
            "tip(s) in inferred simulation correctly identified as erroneous"
        )
                        

    data['hgdp'] = {
        'nmuts': np.zeros(hgdp_ts.num_sites, dtype=int),
        'rm_nmuts': np.zeros((hgdp_ts.num_sites, ntips_below_err_muts), dtype=int),
    }
    
    for tree in hgdp_ts.trees():
        for site in tree.sites():
            muts = site.mutations
            num_mutations = len(muts)
            data['hgdp']['nmuts'][site.id] = num_mutations
            for n in range(ntips_below_err_muts):
                if num_mutations < 2:
                    data['hgdp']['rm_nmuts'][site.id, n] = num_mutations
                else:
                    data['hgdp']['rm_nmuts'][site.id, n] = len(
                        [m for m in muts if tree.num_samples(m.node) > n+1])
        
    data['tgp'] = {
        'nmuts': np.zeros(tgp_ts.num_sites, dtype=int),
        'rm_nmuts': np.zeros((tgp_ts.num_sites, ntips_below_err_muts), dtype=int),
    }
    
    for tree in tgp_ts.trees():
        for site in tree.sites():
            muts = site.mutations
            num_mutations = len(muts)
            data['tgp']['nmuts'][site.id] = num_mutations
            for n in range(ntips_below_err_muts):
                if num_mutations < 2:
                    data['tgp']['rm_nmuts'][site.id, n] = num_mutations
                else:
                    data['tgp']['rm_nmuts'][site.id, n] = len(
                        [m for m in muts if tree.num_samples(m.node) > n+1])

    num_rows = max([v['nmuts'].max() for v in data.values()]) + 1

    tabulated_data = np.zeros((num_rows, 4 * (1 + ntips_below_err_muts)))
    header = []

    column = 0
    for k, v in data.items():
        for i, site_muts in enumerate(itertools.chain([v['nmuts']], v['rm_nmuts'].T)):
            header += [k + ("_all" if i==0 else f"_{i}_tips_err")]
            bincount = np.bincount(site_muts)/len(site_muts)
            tabulated_data[np.arange(len(bincount)), column] = bincount
            column += 1
     
    np.savetxt(
        f"data/muts_per_site_chr{args.chrom}.csv",
        tabulated_data,
        delimiter=",",
        newline="\n",
        header=",".join(header),
        comments="",
    )

class AncestralGeography:
    def __init__(self, ts):
        self.ts = ts
        self.fixed_nodes = set(ts.samples())
        self.locations = np.zeros((self.ts.num_nodes, 2))

    def get_parent_age(self, edge):
        times = self.ts.tables.nodes.time[:]
        return times[operator.attrgetter("parent")(edge)]

    def edges_by_parent_age_asc(self):
        return itertools.groupby(self.ts.edges(), self.get_parent_age)

    def edges_by_parent_asc(self):
        """
        Return an itertools.groupby object of edges grouped by parent in ascending order
        of the time of the parent. Since tree sequence properties guarantee that edges
        are listed in nondecreasing order of parent time
        (https://tskit.readthedocs.io/en/latest/data-model.html#edge-requirements)
        we can simply use the standard edge order
        """
        return itertools.groupby(self.ts.edges(), operator.attrgetter("parent"))

    def parents_in_epoch(parents):
        return itertools.groupby(parents, operator.attrgetter("parent"))

    def edge_span(self, edge):
        return edge.right - edge.left

    def average_edges(self, parent_edges):
        parent = parent_edges[0]
        edges = parent_edges[1]

        child_spanfracs = list()
        child_lats = list()
        child_longs = list()

        for edge in edges:
            child_spanfracs.append(self.edge_span(edge))
            child_lats.append(self.locations[edge.child][0])
            child_longs.append(self.locations[edge.child][1])
        val = utility.weighted_geographic_center(
            child_lats, child_longs, np.ones_like(len(child_lats))
        )
        return parent, val

    def get_ancestral_geography(self, pop_lats, pop_longs, show_progress=False):
        """
        Use dynamic programming to find approximate posterior to sample from
        """

        # Set lat and long for sample nodes
        print(self.ts.num_populations, self.ts.num_nodes)
        print(self.ts.num_individuals)
        if "name" in json.loads(self.ts.population(0).metadata):
            population_names = {
                pop.id: json.loads(pop.metadata)["name"]
                for pop in self.ts.populations()
            }
        elif "id" in json.loads(self.ts.population(0).metadata):
            population_names = {
                pop.id: json.loads(pop.metadata)["id"] for pop in self.ts.populations()
            }
        else:
            raise ValueError("Population metadata encoded incorrectly")
        if self.ts.num_individuals > 0:
            for indiv in self.ts.individuals():
                if len(indiv.location) == 0:
                    for node in indiv.nodes:
                        self.locations[node] = (
                            pop_lats[population_names[self.ts.node(node).population]],
                            pop_longs[population_names[self.ts.node(node).population]],
                        )
                else:
                    for node in indiv.nodes:
                        self.locations[node] = (indiv.location[0], indiv.location[1])
        else:
            for node in self.ts.nodes():
                self.locations[node.id] = (
                    pop_lats[population_names[node.population]],
                    pop_longs[population_names[node.population]],
                )
        # Iterate through the nodes via groupby on parent node
        for parent_edges in tqdm(
            self.edges_by_parent_asc(),
            total=self.ts.num_nodes,
            disable=not show_progress,
        ):
            if parent_edges[0] not in self.fixed_nodes:
                parent, val = self.average_edges(parent_edges)
                self.locations[parent] = val
        return self.locations


def calc_ancestral_geographies(args):
    """
    Calculate ancestral geographies for use in Figure 4 and Supplementary Video
    """

    tgp_hgdp_sgdp_ancients = tskit.load(
        "all-data/hgdp_1kg_sgdp_high_cov_ancients_chr" + args.chrom + ".dated.trees"
    )
    # Remove 1000 Genomes populations
    hgdp_sgdp_ancients = tgp_hgdp_sgdp_ancients.simplify(
        np.where(
            ~np.isin(
                tgp_hgdp_sgdp_ancients.tables.nodes.population[
                    tgp_hgdp_sgdp_ancients.samples()
                ],
                np.arange(54, 80),
            )
        )[0]
    )
    hgdp_sgdp_ancients.dump("all-data/hgdp_sgdp_high_cov_ancients_chr20.dated.trees")

    pop_lats = {}
    pop_longs = {}
    pop_lats["Afanasievo"] = 49.0000
    pop_longs["Afanasievo"] = 89.0000
    pop_lats["Altai"] = 51.3975
    pop_longs["Altai"] = 84.67611111
    pop_lats["Vindija"] = 46.299167
    pop_longs["Vindija"] = 16.070556
    pop_lats["Chagyrskaya"] = 51.442497
    pop_longs["Chagyrskaya"] = 83.154522
    pop_lats["Denisovan"] = 51.3975
    pop_longs["Denisovan"] = 84.67611111
    hgdp_sgdp_ancients_geo = AncestralGeography(hgdp_sgdp_ancients)
    ancestor_coordinates = hgdp_sgdp_ancients_geo.get_ancestral_geography(
        pop_lats, pop_longs, show_progress=True
    )
    np.savetxt(
        "data/hgdp_sgdp_ancients_ancestor_coordinates_chr" + args.chrom + ".csv",
        ancestor_coordinates,
    )


def average_population_ancestors_geography(args):
    """
    Find the average position of ancestors of each population at given time slices.
    Used to plot Figure 4b
    """
    try:
        tgp_hgdp_sgdp_ancestor_locations = np.loadtxt(
            "data/hgdp_sgdp_ancients_ancestor_coordinates_chr" + args.chrom + ".csv"
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Must run 'hgdp_sgdp_ancients_ancestral_geography' first to infer ancestral geography"
        )
    ts = tskit.load(
        "all-data/hgdp_1kg_sgdp_high_cov_ancients_chr" + args.chrom + ".dated.trees"
    )
    ts = ts.simplify(
        np.where(~np.isin(ts.tables.nodes.population[ts.samples()], np.arange(54, 80)))[
            0
        ]
    )

    reference_sets = []
    population_names = []
    for pop in ts.populations():
        reference_sets.append(
            np.where(ts.tables.nodes.population == pop.id)[0].astype(np.int32)
        )
        name = json.loads(pop.metadata.decode())["name"]
        population_names.append(name)
    descendants = ts.mean_descendants(reference_sets)

    avg_lat_lists = list()
    avg_long_lists = list()
    locations = tgp_hgdp_sgdp_ancestor_locations
    times = ts.tables.nodes.time[:]
    time_windows = np.concatenate(
        [np.array([0]), np.logspace(3.5, np.log(np.max(times)), num=40, base=2.718)]
    )
    time_slices_child = list()
    time_slices_parent = list()
    for time in time_windows:
        time_slices_child.append(
            ts.tables.edges.child[
                np.where(
                    np.logical_and(
                        times[ts.tables.edges.child] <= time,
                        times[ts.tables.edges.parent] > time,
                    )
                )[0]
            ]
        )
        time_slices_parent.append(
            ts.tables.edges.parent[
                np.where(
                    np.logical_and(
                        times[ts.tables.edges.child] <= time,
                        times[ts.tables.edges.parent] > time,
                    )
                )[0]
            ]
        )
    num_ancestral_lineages = list()
    for population in tqdm(np.arange(0, 55)):
        avg_lat = list()
        avg_long = list()
        cur_ancestral_lineages = list()
        for i, time in enumerate(time_windows):
            time_slice_child = time_slices_child[i]
            time_slice_parent = time_slices_parent[i]
            ancestral_lineages = np.logical_and(
                descendants[time_slice_child, population] != 0,
                descendants[time_slice_parent, population] != 0,
            )

            cur_ancestral_lineages.append(np.sum(ancestral_lineages))

            time_slice_child = time_slice_child[ancestral_lineages]
            time_slice_parent = time_slice_parent[ancestral_lineages]
            edge_lengths = times[time_slice_parent] - times[time_slice_child]
            weight_parent = 1 - ((times[time_slice_parent] - time) / edge_lengths)
            weight_child = 1 - ((time - times[time_slice_parent]) / edge_lengths)
            if len(time_slice_child) != 0 and len(time_slice_parent) != 0:
                lat_arr = np.vstack(
                    [
                        locations[time_slice_parent][:, 0],
                        locations[time_slice_child][:, 0],
                    ]
                ).T
                long_arr = np.vstack(
                    [
                        locations[time_slice_parent][:, 1],
                        locations[time_slice_child][:, 1],
                    ]
                ).T
                weights = np.vstack([weight_parent, weight_child]).T
                lats, longs = utility.vectorized_weighted_geographic_center(
                    lat_arr, long_arr, weights
                )
                avg_coord = utility.weighted_geographic_center(
                    lats,
                    longs,
                    np.mean(
                        [
                            descendants[time_slice_child, population],
                            descendants[time_slice_parent, population],
                        ],
                        axis=0,
                    ),
                )
                avg_lat.append(avg_coord[0])
                avg_long.append(avg_coord[1])

        num_ancestral_lineages.append(cur_ancestral_lineages)

        avg_lat_lists.append(avg_lat)
        avg_long_lists.append(avg_long)

    with open(
        "data/avg_pop_ancestral_location_LATS_chr" + args.chrom + ".csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerows(avg_lat_lists)
    with open(
        "data/avg_pop_ancestral_location_LONGS_chr" + args.chrom + ".csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerows(avg_long_lists)
    with open(
        "data/num_ancestral_lineages_chr" + args.chrom + ".csv", "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerows(num_ancestral_lineages)


# Simplified region labelling for ancient figures
region_remapping = {
    "Africa": "Africa",
    "AFRICA": "Africa",
    "AMERICA": "Americas",
    "EAST_ASIA": "East Asia",
    "EastAsia": "East Asia",
    "EUROPE": "West Eurasia",
    "OCEANIA": "Oceania",
    "Oceania": "Oceania",
    "MIDDLE_EAST": "West Eurasia",
    "CentralAsiaSiberia": "Central Asia/Siberia",
    "CENTRAL_SOUTH_ASIA": "South Asia",
    "SouthAsia": "South Asia",
    "WestEurasia": "West Eurasia",
    "America": "Americas",
    "EUR": "West Eurasia",
    "AMR": "Americas",
    "EAS": "East Asia",
    "AFR": "Africa",
    "SAS": "South Asia",
    "Ancients": "Ancients",
    "Afanasievo": "Ancients",
    "Max Planck": "Ancients",
}


def calc_unified_reference_sets(args):
    """
    Returns the reference sets, regions, population names, and reference set maps for
    the unified genealogy
    """
    ts = tskit.load(
        "all-data/hgdp_1kg_sgdp_high_cov_ancients_chr" + args.chrom + ".dated.trees"
    )
    pop_names = []
    reference_sets = list()
    regions = list()
    regions_reference_sets = collections.defaultdict(list)
    pop_names = list()
    for index, pop in enumerate(ts.populations()):
        pop_nodes = np.where(ts.tables.nodes.population == pop.id)[0].astype(np.int32)
        if np.sum(pop_nodes) > 0:
            reference_sets.append(pop_nodes)
            name = json.loads(pop.metadata.decode())["name"]
            if index < 54:
                pop_names.append(name)
                region = region_remapping[json.loads(pop.metadata.decode())["region"]]
                regions_reference_sets[region].extend(pop_nodes)
            if index >= 54 and index < 80:
                pop_names.append(name)
                region = region_remapping[
                    json.loads(pop.metadata.decode())["super_population"]
                ]
                regions_reference_sets[region].extend(pop_nodes)
            if index >= 80 and index < 210:
                pop_names.append(name)
                region = region_remapping[json.loads(pop.metadata.decode())["region"]]
                regions_reference_sets[region].extend(pop_nodes)
            if index >= 210:
                pop_names.append(name)
                region = "Ancients"
                regions_reference_sets[region].extend(pop_nodes)
            regions.append(region)
    pickle.dump(reference_sets, open("data/unified_ts_reference_sets.p", "wb"))
    region_nodes = [nodes for _, nodes in regions_reference_sets.items()]
    pickle.dump(region_nodes, open("data/unified_ts_regions_reference_sets.p", "wb"))
    regions = np.array(regions)
    np.savetxt("data/unified_ts_regions.csv", regions, delimiter=",", fmt="%s")
    pop_names = np.array(pop_names)
    np.savetxt("data/unified_ts_pop_names.csv", pop_names, delimiter=",", fmt="%s")
    ref_set_map = np.zeros(np.max(ts.samples()) + 1, dtype=int)
    for index, ref_set in enumerate(reference_sets):
        for node in ref_set:
            ref_set_map[node] = index
    np.savetxt("data/unified_ts_reference_set_map.csv", ref_set_map, delimiter=",")


def calc_ancient_descendants(args):
    """
    Calculate genomic descent statistic for proxy nodes in the unified genealogy
    """
    ts = tskit.load(
        "all-data/hgdp_1kg_sgdp_high_cov_ancients_chr"
        + args.chrom
        + "_dated.binned.nosimplify.trees"
    )
    site_pos = ts.tables.sites.position
    ts = ts.keep_intervals([[site_pos[0], site_pos[-1]]], simplify=False).trim()
    # We have to find the reference sets here rather than loading previously created ones because we're using
    # the unsimplified tree sequence
    populations_reference_sets = list()
    regions_reference_sets = collections.defaultdict(list)
    pop_names = list()
    for pop in ts.populations():
        pop_nodes = np.where(ts.tables.nodes.population == pop.id)[0].astype(np.int32)
        metadata = json.loads(pop.metadata.decode())
        if len(pop_nodes) > 0:
            if "region" in metadata:
                regions_reference_sets[region_remapping[metadata["region"]]].extend(
                    pop_nodes
                )
            elif "super_population" in metadata:
                regions_reference_sets[
                    region_remapping[metadata["super_population"]]
                ].extend(pop_nodes)
            populations_reference_sets.append(pop_nodes)
            pop_names.append(metadata["name"])
    region_nodes = [nodes for _, nodes in regions_reference_sets.items()]

    reference_sets_list = {
        "ancient_descendants": populations_reference_sets,
        "regions_ancient_descendants": region_nodes,
    }
    (
        altai_proxy,
        chagyrskaya_proxy,
        denisovan_proxy,
        vindija_proxy,
        afanasievo_proxy,
    ) = get_ancient_proxy_nodes(ts)
    nodes = np.concatenate(
        [
            altai_proxy,
            chagyrskaya_proxy,
            denisovan_proxy,
            vindija_proxy,
            afanasievo_proxy,
        ]
    )
    for (output_name, reference_sets), names in zip(
        reference_sets_list.items(), [pop_names, regions_reference_sets.keys()]
    ):
        descendants = ts.mean_descendants(reference_sets)
        reference_set_lens = np.array([len(ref_set) for ref_set in reference_sets])
        normalised_descendants = (
            descendants / np.array(reference_set_lens)[np.newaxis, :]
        )
        descendants = pd.DataFrame(
            normalised_descendants[nodes],
            index=[
                "Altai",
                "Altai",
                "Altai",
                "Altai",
                "Chagyrskaya",
                "Chagyrskaya",
                "Chagyrskaya",
                "Chagyrskaya",
                "Denisovan",
                "Denisovan",
                "Denisovan",
                "Denisovan",
                "Vindija",
                "Vindija",
                "Vindija",
                "Vindija",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
                "Afanasievo",
            ],
            columns=names,
        )
        descendants.to_csv(
            "data/unified_ts_chr" + args.chrom + "_" + output_name + ".csv"
        )


def find_descent(ts, proxy_nodes, descent_cutoff, exclude_pop, ref_set_map, pop_names):
    """
    Get genomic locations of descent from given ancestral nodes in 1Kb chunks
    Chunks are binary: 0 indicates no descent from proxy nodes, 1 indicates descent
    No differentation is given between descent from both proxy nodes or descent from
    only one.
    """

    # Array indicating where indivdual descends from ancient sample
    descendants_arr = np.zeros(
        (ts.num_samples, int(ts.get_sequence_length() / 1000)), dtype=int
    )
    for focal_node in proxy_nodes:
        for tree in tqdm(ts.trees()):
            stack = list(tree.children(focal_node))
            while len(stack) != 0:
                cur_node = stack.pop()
                if cur_node in ts.samples():
                    descendants_arr[
                        cur_node,
                        int(np.round(tree.interval[0] / 1000)) : int(
                            np.round(tree.interval[1] / 1000)
                        ),
                    ] = 1
                for child in tree.children(cur_node):
                    stack.append(child)
    # Sum descent from ancient per sample
    sample_desc_sum = np.sum(descendants_arr, axis=1)

    # Note samples which descend from the ancient sample for a span > than the cutoff
    high_descendants = ts.samples()[
        np.where(np.sum(descendants_arr[ts.samples()], axis=1) > descent_cutoff)[0]
    ]
    high_descendants = high_descendants[
        pop_names[ref_set_map[high_descendants]] != exclude_pop
    ]
    descendants_arr = descendants_arr[high_descendants]
    # Construct a dataframe of correlation coefficients between descendants
    corrcoef_df = pd.DataFrame(
        np.corrcoef(descendants_arr),
        index=pop_names[ref_set_map[high_descendants]],
    )
    return descendants_arr.astype(int), corrcoef_df, high_descendants, sample_desc_sum


def calc_ancient_descent_haplotypes(args):
    """
    Finds haplotypes descending from the eight ancient individuals in the unified tree
    sequence
    """

    def save_descent_files(name, proxy, cutoff, exclude_pop, ref_set_map, pop_names):
        (descent_arr, corrcoef_df, descendants, sample_desc_sum) = find_descent(
            ts, proxy, cutoff, exclude_pop, ref_set_map, pop_names
        )
        np.savetxt(
            "data/unified_ts_chr" + args.chrom + "_" + name + "_descent_arr.csv",
            descent_arr,
            fmt="%i",
            delimiter=",",
        )
        np.savetxt(
            "data/unified_ts_chr" + args.chrom + "_" + name + "_descendants.csv",
            descendants,
        )
        corrcoef_df.to_csv(
            "data/unified_ts_chr" + args.chrom + "_" + name + "_corrcoef_df.csv"
        )
        np.savetxt(
            "data/unified_ts_chr" + args.chrom + "_" + name + "_sample_desc_sum.csv",
            sample_desc_sum,
            fmt="%i",
            delimiter=",",
        )

    ts = tskit.load(
        "all-data/hgdp_1kg_sgdp_high_cov_ancients_chr"
        + args.chrom
        + "_dated.binned.nosimplify.trees"
    )
    ts = tsdate.preprocess_ts(ts, **{"keep_unary": True})
    (
        altai_proxy,
        chagyrskaya_proxy,
        denisovan_proxy,
        vindija_proxy,
        afanasievo_proxy,
    ) = get_ancient_proxy_nodes(ts)

    # We have to find the reference sets here rather than loading previously created ones because we're using
    # the preprocessed tree sequence
    populations_reference_sets = list()
    regions_reference_sets = collections.defaultdict(list)
    pop_names = list()
    for pop in ts.populations():
        pop_nodes = np.where(ts.tables.nodes.population == pop.id)[0].astype(np.int32)
        metadata = json.loads(pop.metadata.decode())
        if len(pop_nodes) > 0:
            if "region" in metadata:
                regions_reference_sets[region_remapping[metadata["region"]]].extend(
                    pop_nodes
                )
            elif "super_population" in metadata:
                regions_reference_sets[
                    region_remapping[metadata["super_population"]]
                ].extend(pop_nodes)
            populations_reference_sets.append(pop_nodes)
            pop_names.append(metadata["name"])
    pop_names = np.array(pop_names)
    ref_set_map = np.zeros(np.max(ts.samples()) + 1, dtype=int)
    for index, ref_set in enumerate(populations_reference_sets):
        for node in ref_set:
            ref_set_map[node] = index

    save_descent_files(
        "afanasievo", afanasievo_proxy, 100, "Afanasievo", ref_set_map, pop_names
    )
    save_descent_files("vindija", vindija_proxy, 100, "Vindija", ref_set_map, pop_names)
    save_descent_files(
        "denisovan", denisovan_proxy, 100, "Denisovan", ref_set_map, pop_names
    )
    save_descent_files(
        "chagyrskaya", chagyrskaya_proxy, 100, "Chagyrskaya", ref_set_map, pop_names
    )
    save_descent_files("altai", altai_proxy, 100, "Altai", ref_set_map, pop_names)


def calc_archaic_relationships(args):
    """
    Determine relationships between archaic individuals and younger samples
    NOTE: these relationships are different than genomic descent (normalised
    mean_descendants). Genomic descent shows how much of a reference set
    descends from an ancestor. These relationships indicate the proportion of ancestors
    genome inherited by younger samples.
    """
    ts = tskit.load(
        "all-data/hgdp_1kg_sgdp_high_cov_ancients_dated_chr"
        + args.chrom
        + ".binned.nosimplify.trees"
    )

    tables = ts.tables
    inds = tskit.unpack_bytes(
        tables.individuals.metadata, tables.individuals.metadata_offset
    )
    individual_map = {}
    for index, ind in enumerate(inds):
        ind = json.loads(ind)
        if "name" in ind:
            individual_map[ind["name"]] = index
    altai = ts.individual(individual_map["AltaiNeandertal"]).nodes
    chagyrskaya = ts.individual(individual_map["Chagyrskaya-Phalanx"]).nodes
    denisovan = ts.individual(individual_map["Denisova"]).nodes
    vindija = ts.individual(individual_map["Vindija33.19"]).nodes
    (
        altai_proxy,
        chagyrskaya_proxy,
        denisovan_proxy,
        vindija_proxy,
        afanasievo_proxy,
    ) = get_ancient_proxy_nodes(ts)
    nonarchaic = ts.samples()[:-8]

    # Descent from Vindija: straightforward descent in nonarchaic samples
    # v=vindija, d=denisovan, c=chagyrskaya, a=altai, m=modern (nonarchaic)
    v_descent = dict.fromkeys(["total_v_descent", "v_m"], 0)

    for tree in tqdm(ts.trees(), desc="Vindija Descent"):
        for node in vindija_proxy:
            leaves = list(tree.leaves(node))
            if len(leaves) > 1:
                v_descent["total_v_descent"] += tree.span
                if len(np.intersect1d(leaves, nonarchaic)) > 0:
                    v_descent["v_m"] += tree.span
                else:
                    print(
                        ts.tables.nodes.population[leaves], leaves, vindija, denisovan
                    )
                    raise ValueError("Leaves must be younger than Vindija")
            elif len(leaves) == 1:
                assert leaves[0] in vindija_proxy or leaves[0] in vindija

    # Chagyrskaya
    arrows = ["c_v", "c_d", "c_m", "c_d_v", "c_d_m", "c_v_m"]
    c_descent = dict.fromkeys(["total_c_descent"] + arrows, 0)

    # Iterate over each tree in the tree sequence
    for tree in tqdm(ts.trees(), desc="Chagyrskaya Descent"):
        # Check both archaic chromosome copies
        for node in chagyrskaya_proxy:
            leaves = list(tree.leaves(node))
            # Only investigate tree if > 1 leaves (otherwise check child is archaic)
            if len(leaves) > 1:
                # Dictionary of booleans to track of relationships seen at this tree
                flags = dict.fromkeys(arrows, False)
                # If any descent, add to total
                c_descent["total_c_descent"] += tree.span
                for leaf in leaves:
                    # First check the c->d relationship
                    if leaf in denisovan:
                        flags["c_d"] = True
                    # Next check vindija, which could be c->v or d->v
                    elif leaf in vindija:
                        denisovan_descendant = False
                        parent = tree.parent(leaf)
                        while parent not in chagyrskaya_proxy:
                            if parent in denisovan_proxy:
                                denisovan_descendant = True
                            parent = tree.parent(parent)
                        # If denisovan is noted along path, then path is d->v
                        if denisovan_descendant is True:
                            flags["c_d_v"] = True
                        # If no denisovan seen on path, then path is c->v
                        else:
                            flags["c_v"] = True
                    # Enumerate possible paths if leaf is nonarchaic
                    elif leaf in nonarchaic:
                        vindija_descendant = False
                        denisovan_descendant = False
                        parent = tree.parent(leaf)
                        while parent not in chagyrskaya_proxy:
                            if parent in vindija_proxy:
                                vindija_descendant = True
                            elif parent in denisovan_proxy:
                                denisovan_descendant = True
                            parent = tree.parent(parent)
                        # If we see vindija, we assign relationship to v->m
                        if vindija_descendant:
                            flags["c_v_m"] = True
                        # If we don't see vindija, but do see denisovan, then it's d->m
                        elif (
                            vindija_descendant is False and denisovan_descendant is True
                        ):
                            flags["c_d_m"] = True
                        # If we don't see vindija or denisovan, it's c->m
                        elif (
                            vindija_descendant is False
                            and denisovan_descendant is False
                        ):
                            flags["c_m"] = True
                for path, flag in flags.items():
                    if flag is True:
                        c_descent[path] += tree.span
            else:
                assert leaves[0] in chagyrskaya or leaves[0] in chagyrskaya_proxy

    # Denisovan
    arrows = ["d_v", "d_m", "d_v_m"]
    d_descent = dict.fromkeys(["total_d_descent"] + arrows, 0)

    for tree in tqdm(ts.trees(), desc="Denisovan Descent"):
        for node in denisovan_proxy:
            leaves = list(tree.leaves(node))
            if len(leaves) > 1:
                flags = dict.fromkeys(arrows, False)
                d_descent["total_d_descent"] += tree.span
                for leaf in leaves:
                    if leaf in vindija:
                        flags["d_v"] = True
                    elif leaf in nonarchaic:
                        vindija_descendant = False
                        parent = tree.parent(leaf)
                        while parent not in denisovan_proxy:
                            if parent in vindija_proxy:
                                vindija_descendant = True
                            parent = tree.parent(parent)
                        if vindija_descendant is True:
                            flags["d_v_m"] = True
                        else:
                            flags["d_m"] = True
                for path, flag in flags.items():
                    if flag is True:
                        d_descent[path] += tree.span
            else:
                assert leaves[0] in denisovan or leaves[0] in denisovan_proxy

    # Altai
    arrows = [
        "a_d",
        "a_c",
        "a_v",
        "a_m",
        "a_c_v",
        "a_c_d",
        "a_c_m",
        "a_d_v",
        "a_d_m",
        "a_v_m",
    ]

    a_descent = dict.fromkeys(["total_a_descent"] + arrows, 0)
    for tree in tqdm(ts.trees(), desc="Altai Descent"):
        for node in altai_proxy:
            leaves = list(tree.leaves(node))
            if len(leaves) > 1:
                a_descent["total_a_descent"] += tree.span
                flags = dict.fromkeys(arrows, False)
                for leaf in leaves:
                    if leaf in chagyrskaya:
                        flags["a_c"] = True
                    elif leaf in denisovan:
                        flags["a_d"] = True
                        parent = tree.parent(leaf)
                        while parent not in altai_proxy:
                            if parent in chagyrskaya_proxy:
                                flags["a_c_d"] = True
                            parent = tree.parent(parent)
                    elif leaf in vindija:
                        denisovan_descendant = False
                        chagyrskaya_descendant = False
                        parent = tree.parent(leaf)
                        while parent not in altai_proxy:
                            if parent in denisovan_proxy:
                                denisovan_descendant = True
                            elif parent in chagyrskaya_proxy:
                                chagyrskaya_descendant = True
                            parent = tree.parent(parent)
                        if denisovan_descendant and chagyrskaya_descendant is False:
                            flags["a_d_v"] = True
                        elif chagyrskaya_descendant and denisovan_descendant is False:
                            flags["a_c_v"] = True
                        elif (
                            chagyrskaya_descendant is False
                            and denisovan_descendant is False
                        ):
                            flags["a_v"] = True
                    elif leaf in nonarchaic:
                        denisovan_descendant = False
                        chagyrskaya_descendant = False
                        vindija_descendant = False
                        parent = tree.parent(leaf)
                        while parent not in altai_proxy:
                            if parent in denisovan_proxy:
                                denisovan_descendant = True
                            elif parent in chagyrskaya_proxy:
                                chagyrskaya_descendant = True
                            elif parent in vindija_proxy:
                                vindija_descendant = True
                            parent = tree.parent(parent)
                        if vindija_descendant:
                            flags["a_v_m"] = True
                        elif denisovan_descendant and vindija_descendant is False:
                            flags["a_d_m"] = True
                        elif (
                            chagyrskaya_descendant
                            and denisovan_descendant is False
                            and vindija_descendant is False
                        ):
                            flags["a_c_m"] = True
                        elif (
                            chagyrskaya_descendant is False
                            and denisovan_descendant is False
                            and vindija_descendant is False
                        ):
                            flags["a_m"] = True
                for path, flag in flags.items():
                    if flag is True:
                        a_descent[path] += tree.span
            else:
                assert leaves[0] in altai or leaves[0] in altai_proxy

    with open("data/archaic_descent.txt", "w") as file:
        file.write(json.dumps(a_descent))
        file.write(json.dumps(c_descent))
        file.write(json.dumps(d_descent))
        file.write(json.dumps(v_descent))


def calc_tmrcas(args):
    ts_fn = os.path.join(
        data_prefix, "hgdp_1kg_sgdp_high_cov_ancients_chr" + args.chrom + ".dated.trees"
    )
    tmrcas.save_tmrcas(
        ts_fn, max_pop_nodes=20, num_processes=args.processes, save_raw_data=True
    )


def redate_delete_sites(args):
    chr20_ts_fn = os.path.join(
        data_prefix, "hgdp_1kg_sgdp_chr20_q.missing_binned.dated.trees"
    )
    # if not os.path.isfile(chr20_ts_fn):
    #    subprocess.call(["python", os.path.join(data_prefix, "tsutil.py"), "combine-chromosome",
    #                     os.path.join(data_prefix, "hgdp_1kg_sgdp_chr20_p.missing_binned.dated.trees"),
    #                      os.path.join(data_prefix, "hgdp_1kg_sgdp_chr20_q.missing_binned.dated.trees"), chr20_ts_fn])
    orig_dated = tskit.load(chr20_ts_fn)
    delete_sites = []
    for site in orig_dated.sites():
        mutations = site.mutations
        if len(mutations) > 100:
            delete_sites.append(site.id)

    deleted_ts = orig_dated.delete_sites(delete_sites)
    deleted_dated = tsdate.date(
        deleted_ts, Ne=10000, mutation_rate=1e-8, progress=True, ignore_oldest_root=True
    )
    deleted_dated.dump(
        os.path.join(
            data_prefix, "hgdp_1kg_sgdp_chr20_q.missing_binned.delete_100.dated.trees"
        )
    )
    comparable_sites = np.isin(
        orig_dated.tables.sites.position, deleted_dated.tables.sites.position
    )
    deleted_dated_site_times = tsdate.sites_time_from_ts(
        deleted_dated, unconstrained=True, node_selection="arithmetic"
    )
    orig_dated_site_times = tsdate.sites_time_from_ts(
        orig_dated, unconstrained=True, node_selection="arithmetic"
    )

    chr20_ages = pd.DataFrame(
        {
            "Position": orig_dated.tables.sites.position[comparable_sites],
            "original_age": orig_dated_site_times[comparable_sites],
            "deleted_age": deleted_dated_site_times,
        }
    )
    chr20_ages.to_csv("data/hgdp_tgp_sgdp_chr20_q.deleted_site_times.csv")


def main():
    name_map = {
        "unified_recurrent_mutations": calc_unified_recurrent_mutations,
        "simulated_recurrent_mutations": calc_simulated_recurrent_mutations,
        "tgp_dates": tgp_date_estimates,
        "ancient_constraints": calc_ancient_constraints_tgp,
        "hgdp_sgdp_ancients_ancestral_geography": calc_ancestral_geographies,
        "average_pop_ancestors_geography": average_population_ancestors_geography,
        "reference_sets": calc_unified_reference_sets,
        "archaic_relationships": calc_archaic_relationships,
        "ancient_descendants": calc_ancient_descendants,
        "ancient_descent_haplotypes": calc_ancient_descent_haplotypes,
        "tmrcas": calc_tmrcas,
        "redate_delete_sites": redate_delete_sites,
    }

    parser = argparse.ArgumentParser(
        description="Process the human data and make data files for plotting."
    )
    parser.add_argument(
        "name", type=str, help="figure name", choices=list(name_map.keys()) + ["all"]
    )
    parser.add_argument(
        "--chrom",
        type=str,
        default="20",
        help="The chromosome number to run analysis on",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=1,
        help="The number of CPUs to use in for some of the more intensive calculations",
    )

    args = parser.parse_args()
    if args.name == "all":
        for func_name, func in name_map.items():
            print("Running: " + func_name)
            func(args)
    else:
        name_map[args.name](args)


if __name__ == "__main__":
    main()

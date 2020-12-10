"""
Analyses real data in all-data directory. Outputs csvs for
plotting in plot.py
"""
import argparse
import csv
import os.path
import json
import itertools
import operator
import pickle

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
    altai_proxy = np.where(ts.tables.nodes.time == 4400.01)[0]
    chagyrskaya_proxy = np.where(ts.tables.nodes.time == 3200.01)[0]
    denisovan_proxy = np.where(ts.tables.nodes.time == 2556.01)[0]
    vindija_proxy = np.where(ts.tables.nodes.time == 2000.01)[0]
    afanasievo_proxy = np.where(ts.tables.nodes.time == 164.01)[0]
    return (
        altai_proxy,
        chagyrskaya_proxy,
        denisovan_proxy,
        vindija_proxy,
        afanasievo_proxy,
    )


def get_relate_tgp_age_df():
    if os.path.exists("data/1kg_chr20_relate_mutation_ages.csv"):
        relate_ages = pd.read_csv(
            "data/1kg_chr20_relate_mutation_ages.csv", index_col=0
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
        # age_cols = [c for c in relate_ages.columns if "mean_est" in c]
        # relate_ages["relate_avg_age"] = relate_ages[age_cols].mean(axis=1)
        # upper_age_cols = [c for c in relate_ages.columns if "upper_age" in c]
        # relate_ages["relate_upper_age_avg"] = relate_ages[upper_age_cols].mean(axis=1)
        daf_cols = [c for c in relate_ages.columns if "DAF" in c]
        relate_ages["relate_daf_sum"] = relate_ages[daf_cols].sum(axis=1)
        relate_ages["relate_ancestral_allele"] = relate_ages["ancestral/derived"].str[0]
        relate_ages["relate_derived_allele"] = relate_ages["ancestral/derived"].str[2]
        # relate_ages = relate_ages[["CHR", "BP", "ID", "relate_daf_sum", "relate_avg_age", "relate_upper_age_avg", "relate_ancestral_allele", "relate_derived_allele"]]
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
        # tgp_chr20 = tskit.load("all-data/1kg_chr20.dated.50slices.trees")
        tgp_chr20 = tskit.load("all-data/1kg_chr20.dated.trees")
        # posterior_mut_ages = tsdate.sites_time_from_ts(tgp_chr20, mutation_age="child")
        # posterior_upper_bound = tsdate.sites_time_from_ts(tgp_chr20, mutation_age="parent")
        posterior_mut_ages, posterior_upper_bound, oldest_mut_nodes = get_mut_ages(
            tgp_chr20, unconstrained=False
        )
        site_frequencies = get_site_frequencies(tgp_chr20)
        tsdate_ages = pd.DataFrame(
            {
                "Position": tgp_chr20.tables.sites.position,
                "tsdate_age": posterior_mut_ages,
                "tsdate_upper_bound": posterior_upper_bound,
                "tsdate_frequency": site_frequencies,
                "tsdate_ancestral_allele": np.array(
                    tskit.unpack_strings(
                        tgp_chr20.tables.sites.ancestral_state,
                        tgp_chr20.tables.sites.ancestral_state_offset,
                    )
                ),
                "tsdate_derived_allele": np.array(
                    tskit.unpack_strings(
                        tgp_chr20.tables.mutations.derived_state,
                        tgp_chr20.tables.mutations.derived_state_offset,
                    )
                )[oldest_mut_nodes],
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
    # merged = merged.drop(columns=["Position_x", "Position_y"])
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
        no_samp_muts = ~np.logical_and(
            np.isin(mutations_table.site, unique_sites),
            np.isin(mutations_table.node, ts.samples()),
        )
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


def get_ancient_constraints_tgp(args):
    if os.path.exists("all-data/all_ancients_chr20.samples"):
        ancient_samples = tsinfer.load("all-data/all_ancients_chr20.samples")
    else:
        raise FileNotFoundError(
            "Must create all_ancients_chr20.samples using all-data/Makefile"
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


def get_unified_recurrent_mutations(args):
    """
    Get recurrent mutations from the unified tree sequence
    """
    ts = tskit.load("all-data/hgdp_1kg_sgdp_high_cov_ancients_dated_chr20.trees")

    (
        recurrent_counts,
        recurrent_counts_nosamples,
        sites_by_muts_nodouble,
        recurrent_counts_two_muts,
    ) = get_recurrent_mutations(ts)
    df = pd.DataFrame(recurrent_counts, columns=["recurrent_counts"])
    df.to_csv("data/unified_chr20.recurrent_counts.csv")
    df = pd.DataFrame(
        recurrent_counts_nosamples, columns=["recurrent_counts_nosamples"]
    )
    df.to_csv("data/unified_chr20.recurrent_counts_nosamples.csv")
    df = pd.DataFrame(sites_by_muts_nodouble, columns=["recurrent_counts_nodouble"])
    df.to_csv("data/unified_chr20.recurrent_counts_nodouble.csv")

    df = pd.DataFrame(recurrent_counts_two_muts, columns=["recurrent_counts_two_muts"])
    df.to_csv("data/unified_chr20.recurrent_counts_nosamples_two_muts.csv")


def min_site_times_ancients(args):
    samples = tsinfer.load("all-data/1kg_ancients_noreich_chr20.samples")
    min_times = samples.min_site_times(individuals_only=True)
    df = pd.DataFrame(np.unique(min_times, return_counts=True))
    df.to_csv("data/1kg_ancients_chr20_min_site_times.csv")


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
        population_names = {
            pop.id: json.loads(pop.metadata)["name"] for pop in self.ts.populations()
        }
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


def find_ancestral_geographies(args):
    """
    Calculate ancestral geographies
    """

    tgp_hgdp_sgdp_ancients = tskit.load(
        "all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic.trees"
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
    np.savetxt("data/hgdp_sgdp_ancients_ancestor_coordinates.csv", ancestor_coordinates)


def average_population_ancestors_geography(args):
    """
    Find the average position of ancestors of each population at given time slices.
    Used to plot Figure 4b
    """
    try:
        tgp_hgdp_sgdp_ancestor_locations = np.loadtxt(
            "data/hgdp_sgdp_ancients_ancestor_coordinates.csv"
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Must run 'ancient_descendants' first to infer ancestral geography"
        )
    ts = tskit.load(
        """all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.
            binned.historic.trees"""
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
    time_windows_smaller = np.concatenate(
        [np.array([0]), np.logspace(3.5, 11, num=40, base=2.718)]
    )
    times = ts.tables.nodes.time[:]
    time_slices_child = list()
    time_slices_parent = list()
    for time in time_windows_smaller:
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
        for i, time in enumerate(time_windows_smaller):
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

    with open("data/avg_pop_ancestral_location_LATS.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(avg_lat_lists)
    with open("data/avg_pop_ancestral_location_LONGS.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(avg_long_lists)
    with open("data/num_ancestral_lineages.csv", "w", newline="") as f:
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


def get_unified_reference_sets(args):
    """
    Returns the reference sets, regions, population names, and reference set maps for
    the unified genealogy
    """
    ts = tskit.load("all-data/hgdp_1kg_sgdp_high_cov_ancients_dated_chr20.trees")
    pop_names = []
    reference_sets = list()
    regions = list()
    pop_names = list()
    for index, pop in enumerate(ts.populations()):
        pop_nodes = np.where(ts.tables.nodes.population == pop.id)[0].astype(np.int32)
        if np.sum(pop_nodes) > 0:
            reference_sets.append(pop_nodes)
            name = json.loads(pop.metadata.decode())["name"]
            if index < 54:
                pop_names.append(name)
                region = region_remapping[json.loads(pop.metadata.decode())["region"]]
            if index >= 54 and index < 80:
                pop_names.append(name)
                region = region_remapping[
                    json.loads(pop.metadata.decode())["super_population"]
                ]
            if index >= 80 and index < 210:
                pop_names.append(name)
                region = region_remapping[json.loads(pop.metadata.decode())["region"]]
            if index >= 210:
                pop_names.append(name)
                region = "Ancients"
            regions.append(region)

    pickle.dump(reference_sets, open("data/combined_ts_reference_sets.p", "wb"))
    regions = np.array(regions)
    np.savetxt("data/combined_ts_regions.csv", regions, delimiter=",", fmt="%s")
    pop_names = np.array(pop_names)
    np.savetxt("data/combined_ts_pop_names.csv", pop_names, delimiter=",", fmt="%s")
    ref_set_map = np.zeros(np.max(ts.samples()) + 1, dtype=int)
    for index, ref_set in enumerate(reference_sets):
        for node in ref_set:
            ref_set_map[node] = index
    np.savetxt("data/combined_ts_reference_set_map.csv", ref_set_map, delimiter=",")


def find_ancient_descendants(args):
    """
    Calculate genomic descent statistic for proxy nodes in the unified genealogy
    """
    # ts = tskit.load(
    #    "all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic.snipped.trees"
    ts = tskit.load(
        """all-data/hgdp_1kg_sgdp_high_cov_ancients_dated_chr20.binned.nosimplify.trees"""
    )

    reference_sets = list()
    for pop in ts.populations():
        pop_nodes = np.where(ts.tables.nodes.population == pop.id)[0].astype(np.int32)
        if np.sum(pop_nodes) > 0:
            reference_sets.append(pop_nodes)

    descendants = ts.mean_descendants(reference_sets)
    reference_set_lens = np.array([len(ref_set) for ref_set in reference_sets])
    normalised_descendants = descendants / np.array(reference_set_lens)[np.newaxis, :]
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
    descendants = pd.DataFrame(
        normalised_descendants[nodes],
        index=[
            "Altai",
            "Altai",
            "Chagyrskaya",
            "Chagyrskaya",
            "Denisovan",
            "Denisovan",
            "Vindija",
            "Vindija",
            "Afanasievo",
            "Afanasievo",
            "Afanasievo",
            "Afanasievo",
        ],
    )
    descendants.to_csv("data/combined_ts_ancient_descendants.csv")


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


def find_ancient_descent_haplotypes(args):
    """
    Finds haplotypes descending from the eight ancient individuals in the combined tree
    sequence
    """

    def save_descent_files(name, proxy, cutoff, exclude_pop, ref_set_map, pop_names):
        (descent_arr, corrcoef_df, descendants, sample_desc_sum) = find_descent(
            ts, proxy, cutoff, exclude_pop, ref_set_map, pop_names
        )
        np.savetxt(
            "data/combined_ts_" + name + "_descent_arr.csv",
            descent_arr,
            fmt="%i",
            delimiter=",",
        )
        np.savetxt("data/combined_ts_" + name + "_descendants.csv", descendants)
        corrcoef_df.to_csv("data/combined_ts_" + name + "_corrcoef_df.csv")
        np.savetxt(
            "data/combined_ts_" + name + "_sample_desc_sum.csv",
            sample_desc_sum,
            fmt="%i",
            delimiter=",",
        )

    ts = tskit.load(
        """all-data/hgdp_1kg_sgdp_high_cov_ancients_dated_chr20.binned.
            nosimplify.trees"""
    )
    ts = tsdate.preprocess_ts(ts, **{"keep_unary": True})
    (
        altai_proxy,
        chagyrskaya_proxy,
        denisovan_proxy,
        vindija_proxy,
        afanasievo_proxy,
    ) = get_ancient_proxy_nodes(ts)
    ref_set_map = np.loadtxt("data/combined_ts_reference_set_map.csv").astype(int)
    pop_names = np.genfromtxt("data/combined_ts_pop_names.csv", dtype="str")
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


def find_archaic_relationships(args):
    """
    Determine relationships between archaic individuals and younger samples
    NOTE: these relationships are different than genomic descent (normalised
    mean_descendants). Genomic descent shows how much of a reference set
    descends from an ancestor. These relationships indicate the proportion of ancestors
    genome inherited by younger samples.
    """
    ts = tskit.load(
        """all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.
        binned.historic.snipped.trees"""
    )
    tables = ts.tables
    altai_proxy = np.where(ts.tables.nodes.time == 4400.01)[0]
    chagyrskaya_proxy = np.where(ts.tables.nodes.time == 3200.01)[0]
    altai = np.where(tables.nodes.population == ts.num_populations - 1)[0]
    denisovan = np.where(tables.nodes.population == ts.num_populations - 4)[0]
    denisovan_proxy = np.where(ts.tables.nodes.time == 2556.01)[0]
    vindija = np.where(tables.nodes.population == ts.num_populations - 3)[0]
    vindija_proxy = np.where(ts.tables.nodes.time == 2000.01)[0]
    chagyrskaya = np.where(tables.nodes.population == ts.num_populations - 2)[0]
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


def get_tmrcas(args):
    ts_fn = os.path.join(
        data_prefix,
        "merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic.trees",
    )
    tmrcas.save_tmrcas(
        ts_fn, max_pop_nodes=20, num_processes=args.num_processes, save_raw_data=True
    )


def main():
    name_map = {
        "recurrent_mutations": get_unified_recurrent_mutations,
        "tgp_dates": tgp_date_estimates,
        "ancient_constraints": get_ancient_constraints_tgp,
        "min_site_times_ancients": min_site_times_ancients,
        "hgdp_sgdp_ancients_ancestral_geography": find_ancestral_geographies,
        "average_pop_ancestors_geography": average_population_ancestors_geography,
        "get_reference_sets": get_unified_reference_sets,
        "archaic_relationships": find_archaic_relationships,
        "ancient_descendants": find_ancient_descendants,
        "ancient_descent_haplotypes": find_ancient_descent_haplotypes,
        "all_mrcas": get_tmrcas,
    }

    parser = argparse.ArgumentParser(
        description="Process the human data and make data files for plotting."
    )
    parser.add_argument(
        "name", type=str, help="figure name", choices=list(name_map.keys())
    )
    parser.add_argument(
        "--num_processes",
        "-p",
        type=int,
        default=1,
        help="The number of CPUs to use in for some of the more intensive calculations",
    )

    args = parser.parse_args()
    name_map[args.name](args)


if __name__ == "__main__":
    main()

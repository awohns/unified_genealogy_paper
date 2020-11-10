"""
Analyses real data in all-data directory. Outputs csvs for
plotting in plot.py
"""
import argparse
import os.path
import json
import itertools
import operator
import pickle

import numpy as np
import pandas as pd

import tskit
import tsinfer
import tsdate

from tqdm import tqdm

import utility

data_prefix = "all-data"


def get_relate_tgp_age_df():
    #    if path.exists("all-data/1kg_chr20_relate_mutation_ages_geometric.csv"):
    #        relate_ages = pd.read_csv("all-data/1kg_chr20_relate_mutation_ages_geometric.csv", index_col=0)
    #    else:
    relate_ts = tskit.load("/home/jk/large_files/relate/relate_chr20_metdata.trees")
    relate_mut_ages, relate_mut_upper_bound, mut_ids = get_mut_ages(
        relate_ts, unconstrained=False, geometric=False
    )
    relate_frequencies = get_site_frequencies(relate_ts)
    relate_ages = pd.DataFrame(
        {
            "position": relate_ts.tables.sites.position[
                relate_ts.tables.mutations.site
            ][mut_ids],
            "relate_age": relate_mut_ages,
            "relate_upper_bound": relate_mut_upper_bound,
            "relate_ancestral_allele": np.array(
                tskit.unpack_strings(
                    relate_ts.tables.sites.ancestral_state,
                    relate_ts.tables.sites.ancestral_state_offset,
                )
            )[relate_ts.tables.mutations.site][mut_ids],
            "relate_derived_allele": np.array(
                tskit.unpack_strings(
                    relate_ts.tables.mutations.derived_state,
                    relate_ts.tables.mutations.derived_state_offset,
                )
            )[mut_ids],
            "relate_frequency": relate_frequencies,
        }
    )
    relate_ages.to_csv("all-data/1kg_chr20_relate_mutation_ages_geometric.csv")
    return relate_ages


def get_geva_tgp_age_df():
    if os.path.exists("all-data/1kg_chr20_geva_mutation_ages.csv"):
        geva_ages = pd.read_csv(
            "all-data/1kg_chr20_geva_mutation_ages.csv", index_col=0
        )
    else:
        geva = pd.read_csv(
            "/home/wilderwohns/tsinfer_geva/atlas.chr20.csv.gz",
            delimiter=",",
            skipinitialspace=True,
            skiprows=3,
        )
        geva_tgp = geva[geva["DataSource"] == "TGP"]
        geva_tgp_consistent = geva_tgp[geva_tgp["AlleleAnc"] == geva_tgp["AlleleRef"]]
        geva_ages = geva_tgp_consistent[
            ["Position", "AgeMean_Jnt", "AgeCI95Upper_Jnt", "AlleleRef", "AlleleAlt"]
        ]
        geva_ages.to_csv("all-data/1kg_chr20_geva_mutation_ages.csv")
    return geva_ages


def get_tsdate_tgp_age_df():
    if os.path.exists(
        """all-data/tsdate_ages_1kg_chr20.iter.dated.binned_ma0.1_ms0.1_NNone_p16.
            simplified.dated.insideoutside.trees"""
    ):
        tsdate_ages = pd.read_csv(
            """all-data/tsdate_ages_1kg_chr20.iter.dated.
                binned_ma0.1_ms0.1_NNone_p16.simplified.dated.insideoutside.trees""",
            index_col=0,
        )

    else:

        tgp_chr20 = tskit.load(
            """all-data/1kg_chr20.iter.dated.binned_ma0.1_ms0.1_NNone_p16.simplified.
            dated.insideoutside.trees"""
        )

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
        tsdate_ages.to_csv(
            """all-data/tsdate_ages_1kg_chr20.iter.dated.binned_ma0.1_ms0.1_NNone_p16.
            simplified.dated.insideoutside.trees"""
        )
    return tsdate_ages


def tgp_date_estimates():
    """
    Produce comparable set of mutations from tgp
    """
    tsdate_ages = get_tsdate_tgp_age_df()
    geva = get_geva_tgp_age_df()
    merged = pd.merge(
        tsdate_ages,
        geva,
        left_on=["position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
        right_on=["Position", "AlleleRef", "AlleleAlt"],
    )
    relate_ages = get_relate_tgp_age_df()
    merged = pd.merge(
        merged,
        relate_ages,
        left_on=["Position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
        right_on=["position", "relate_ancestral_allele", "relate_derived_allele"],
    )
    merged = merged[
        np.abs(merged["tsdate_frequency"] - merged["relate_frequency"]) < 0.5
    ]
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


def get_ancient_constraints_tgp():
    if os.path.exists("all-data/1kg_ancients_only_chr20.samples"):
        ancient_samples = tsinfer.load("all-data/1kg_ancients_only_chr20.samples")
    else:
        ancient_samples = tsinfer.load("all-data/1kg_ancients_chr20.samples")
        print("Subsetting SampleData file to only keep ancient samples")
        ancient_indiv_ids = np.where(ancient_samples.individuals_time[:] != 0)[0]
        ancient_sample_ids = np.where(
            ancient_samples.individuals_time[:][ancient_samples.samples_individual] != 0
        )[0]
        ancient_genos = ancient_samples.sites_genotypes[:]
        ancient_sites = np.where(
            np.any(ancient_genos[:, ancient_sample_ids] == 1, axis=1)
        )[0]
        ancient_samples = ancient_samples.subset(
            individuals=ancient_indiv_ids, sites=ancient_sites
        )
        copy = ancient_samples.copy("all-data/1kg_ancients_only_chr20.samples")
        copy.finalise()
        print(
            "Subsetted to {} samples and {} sites".format(
                ancient_samples.num_samples, ancient_samples.num_sites
            )
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
    constraint_df.to_csv("all-data/ancient_constraints.csv")
    try:
        tgp_mut_ests = pd.read_csv("all-data/tgp_mutations.csv", index_col=0)
    except FileNotFoundError:
        raise ValueError("tgp_mutations.csv does not exist. Must run tgp_dates first")
    tgp_muts_constraints = pd.merge(
        tgp_mut_ests,
        constraint_df,
        how="left",
        left_on=["Position", "tsdate_ancestral_allele", "tsdate_derived_allele"],
        right_on=["Position", "Reference Allele", "Alternative Allele"],
    )
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


def get_mutations_by_sample():
    """
    See number of mutations above each sample
    """


def get_tgp_recurrent_mutations():
    filename = os.path.join(
        data_prefix,
        """1kg_chr20.iter.dated.binned_ma0.1_ms0.1_NNone_p16.simplified.dated.
            insideoutside.trees""",
    )

    ts = tskit.load(filename)
    (
        recurrent_counts,
        recurrent_counts_nosamples,
        sites_by_muts_nodouble,
        recurrent_counts_two_muts,
    ) = get_recurrent_mutations(ts)
    df = pd.DataFrame(recurrent_counts, columns=["recurrent_counts"])
    df.to_csv("data/1kg_chr20_ma0.1_ms0.1_p16.recurrent_counts.csv")
    df = pd.DataFrame(
        recurrent_counts_nosamples, columns=["recurrent_counts_nosamples"]
    )
    df.to_csv("data/1kg_chr20_ma0.1_ms0.1_p16.recurrent_counts_nosamples.csv")
    df = pd.DataFrame(sites_by_muts_nodouble, columns=["recurrent_counts_nodouble"])
    df.to_csv("data/1kg_chr20_ma0.1_ms0.1_p16.recurrent_counts_nodouble.csv")

    df = pd.DataFrame(recurrent_counts_two_muts, columns=["recurrent_counts_two_muts"])
    df.to_csv("data/1kg_chr20_ma0.1_ms0.1_p16.recurrent_counts_nosamples_two_muts.csv")


def get_hgdp_recurrent_mutations():
    filename = os.path.join(
        data_prefix, "hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.trees"
    )

    ts = tskit.load(filename)
    (
        recurrent_counts,
        recurrent_counts_nosamples,
        sites_by_muts_nodouble,
        recurrent_counts_two_muts,
    ) = get_recurrent_mutations(ts)
    df = pd.DataFrame(recurrent_counts, columns=["recurrent_counts"])
    df.to_csv(
        """data/hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.
            recurrent_counts.csv"""
    )
    df = pd.DataFrame(
        recurrent_counts_nosamples, columns=["recurrent_counts_nosamples"]
    )
    df.to_csv(
        """data/hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.
            recurrent_counts_nosamples.csv"""
    )
    df = pd.DataFrame(sites_by_muts_nodouble, columns=["recurrent_counts_nodouble"])
    df.to_csv(
        """data/hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.
            recurrent_counts_nodouble.csv"""
    )

    df = pd.DataFrame(recurrent_counts_two_muts, columns=["recurrent_counts_two_muts"])
    df.to_csv(
        """data/hgdp_missing_data_chr20_ma0.5_ms0.05_p15.simplify.
            recurrent_counts_nosamples_two_muts.csv"""
    )


def get_sgdp_recurrent_mutations():
    filename = os.path.join(data_prefix, "sgdp_chr20.tsinferred.trees")

    ts = tskit.load(filename)
    (
        recurrent_counts,
        recurrent_counts_nosamples,
        sites_by_muts_nodouble,
        recurrent_counts_two_muts,
    ) = get_recurrent_mutations(ts)
    df = pd.DataFrame(recurrent_counts, columns=["recurrent_counts"])
    df.to_csv("data/sgdp_chr20.tsinferred.recurrent_counts.csv")
    df = pd.DataFrame(
        recurrent_counts_nosamples, columns=["recurrent_counts_nosamples"]
    )
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


def find_ancestral_geographies():
    """
    Calculate ancestral geographies
    """

    tgp_hgdp_sgdp_ancients = tskit.load(
        """all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.
            historic.trees"""
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
    np.savetxt(
        "all-data/hgdp_sgdp_ancients_ancestor_coordinates.csv", ancestor_coordinates
    )


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
    "CentralAsiaSiberia": "Central Asia-Siberia",
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


def find_ancient_descendants():
    ts = tskit.load(
        "all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic.snipped.trees"
    )
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
    descendants = ts.mean_descendants(reference_sets)
    # mean_descendants is "averaged over the portions of the genome for which the node
    # is ancestral to any sample", and we remove this normalisation in order to
    # find the proportion of the chromosome which descends from each ancestral node
    # of interest

#    def ancestral_anywhere(nodes, descendants):
#        ancestral_lengths = np.ones(ts.num_nodes)
#        for tree in ts.trees():
#            for node in nodes:
#                if len(list(tree.leaves(node))) > 1:
#                    ancestral_lengths[node] += tree.span
#        ancestral_lengths = ancestral_lengths / ts.get_sequence_length()
#        descendants = descendants * ancestral_lengths[:, np.newaxis]
#        return descendants
#
#    # Altai proxy nodes
    altai_proxy = np.where(ts.tables.nodes.time == 4400.01)[0]
    chagyrskaya_proxy = np.where(ts.tables.nodes.time == 3200.01)[0]
    denisovan_proxy = np.where(ts.tables.nodes.time == 2556.01)[0]
    vindija_proxy = np.where(ts.tables.nodes.time == 2000.01)[0]
    afanasievo_proxy = np.where(ts.tables.nodes.time == 164.01)[0]
    nodes = np.concatenate(
        [
            altai_proxy,
            chagyrskaya_proxy,
            denisovan_proxy,
            vindija_proxy,
            afanasievo_proxy,
        ]
    )
#    descendants = ancestral_anywhere(nodes, descendants)
    reference_set_lens = np.array([len(ref_set) for ref_set in reference_sets])
    normalised_descendants = (
        descendants / np.array(reference_set_lens)[np.newaxis, :]
    )
    np.savetxt(
        "data/combined_ts_ancient_descendants.csv",
        normalised_descendants[nodes],
        delimiter=",",
    )


def find_descent(ts, proxy_nodes, descent_cutoff, exclude_pop, ref_set_map, pop_names):
    """
    Get genomic locations of descent from given ancestral nodes in 1Kb chunks
    Chunks are binary: 0 indicates no descent from proxy nodes, 1 indicates descent
    No differentation is given between descent from both proxy nodes or descent from
    only one.
    """

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
    high_descendants = ts.samples()[
        np.where(np.sum(descendants_arr[ts.samples()], axis=1) > descent_cutoff)[0]
    ]
    high_descendants = high_descendants[
        pop_names[ref_set_map[high_descendants]] != exclude_pop
    ]
    corrcoef_df = pd.DataFrame(
        np.corrcoef(descendants_arr[high_descendants]),
        index=pop_names[ref_set_map[high_descendants]],
    )
    return descendants_arr[ts.samples()], corrcoef_df, high_descendants


def find_ancient_descent_haplotypes():
    """
    Finds patterns of descent from the eight ancient individuals in the combined tree
    sequence
    """
    ts = tskit.load(
        "all-data/hgdp_1kg_sgdp_high_cov_ancients_chr20.binned.dated.historic.nosimplify.trees"
    )
    ts = tsdate.preprocess_ts(ts, **{"keep_unary": True})
    altai_proxy = np.where(ts.tables.nodes.time == 4400.01)[0]
    chagyrskaya_proxy = np.where(ts.tables.nodes.time == 3200.01)[0]
    denisovan_proxy = np.where(ts.tables.nodes.time == 2556.01)[0]
    vindija_proxy = np.where(ts.tables.nodes.time == 2000.01)[0]
    afanasievo_proxy = np.where(ts.tables.nodes.time == 164.01)[0]
    ref_set_map = np.loadtxt("data/combined_ts_reference_set_map.csv").astype(int)
    pop_names = np.genfromtxt("data/combined_ts_pop_names.csv", dtype="str")
    (
        afanasievo_descent_arr,
        afanasievo_corrcoef_df,
        afanasievo_descendants,
    ) = find_descent(ts, afanasievo_proxy, 100, "Afanasievo", ref_set_map, pop_names)
    np.savetxt("data/combined_ts_afanasievo_descent_arr.csv", afanasievo_descent_arr)
    np.savetxt("data/combined_ts_afanasievo_descendants.csv", afanasievo_descendants)
    afanasievo_corrcoef_df.to_csv("data/combined_ts_afanasievo_corrcoef_df.csv")
    vindija_descent_arr, vindija_corrcoef_df, vindija_descendants = find_descent(
        ts, vindija_proxy, 100, "vindija", ref_set_map, pop_names
    )
    np.savetxt("data/combined_ts_vindija_descent_arr.csv", vindija_descent_arr)
    np.savetxt("data/combined_ts_vindija_descendants.csv", vindija_descendants)
    vindija_corrcoef_df.to_csv("data/combined_ts_vindija_corrcoef_df.csv")
    denisovan_descent_arr, denisovan_corrcoef_df, denisovan_descendants = find_descent(
        ts, denisovan_proxy, 100, "denisovan", ref_set_map, pop_names
    )
    np.savetxt("data/combined_ts_denisovan_descent_arr.csv", denisovan_descent_arr)
    np.savetxt("data/combined_ts_denisovan_descendants.csv", denisovan_descendants)
    denisovan_corrcoef_df.to_csv("data/combined_ts_denisovan_corrcoef_df.csv")
    (
        chagyrskaya_descent_arr,
        chagyrskaya_corrcoef_df,
        chagyrskaya_descendants,
    ) = find_descent(ts, chagyrskaya_proxy, 100, "chagyrskaya", ref_set_map, pop_names)
    np.savetxt("data/combined_ts_chagyrskaya_descent_arr.csv", chagyrskaya_descent_arr)
    np.savetxt("data/combined_ts_chagyrskaya_descendants.csv", chagyrskaya_descendants)
    chagyrskaya_corrcoef_df.to_csv("data/combined_ts_chagyrskaya_corrcoef_df.csv")

    altai_descent_arr, altai_corrcoef_df, altai_descendants = find_descent(
        ts, altai_proxy, 500, "altai", ref_set_map, pop_names
    )
    np.savetxt("data/combined_ts_altai_descent_arr.csv", altai_descent_arr)
    np.savetxt("data/combined_ts_altai_descendants.csv", altai_descendants)
    altai_corrcoef_df.to_csv("data/combined_ts_altai_corrcoef_df.csv")


def find_archaic_relationships():
    ts = tskit.load(
        "all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic.snipped.trees"
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


def main():
    name_map = {
        "recurrent_mutations_tgp": get_tgp_recurrent_mutations,
        "recurrent_mutations_hgdp": get_hgdp_recurrent_mutations,
        "recurrent_mutations_sgdp": get_sgdp_recurrent_mutations,
        "tgp_dates": tgp_date_estimates,
        "ancient_constraints": get_ancient_constraints_tgp,
        "min_site_times_ancients": min_site_times_ancients,
        "hgdp_sgdp_ancients_ancestral_geography": find_ancestral_geographies,
        "archaic_relationships": find_archaic_relationships,
        "ancient_descendants": find_ancient_descendants,
        "ancient_descent_haplotypes": find_ancient_descent_haplotypes,
    }

    parser = argparse.ArgumentParser(
        description="Process the human data and make data files for plotting."
    )
    parser.add_argument(
        "name", type=str, help="figure name", choices=list(name_map.keys())
    )

    args = parser.parse_args()
    name_map[args.name]()


if __name__ == "__main__":
    main()

"""
Various utilities for manipulating tree sequences and running tsinfer.
"""
import argparse
import csv
import re
import json

import tskit
import tsinfer
import tsdate
import daiquiri
import numpy as np


def run_simplify(args):
    ts = tskit.load(args.input)
    ts = ts.simplify()
    ts.dump(args.output)


def run_get_dated_samples(args):
    samples = tsinfer.load(args.samples)
    ts = tskit.load(args.ts)
    assert args.samples.endswith(".samples")
    prefix = args.samples[0 : -len(".samples")]
    copy = samples.copy(prefix + ".dated.samples")
    copy.sites_time[:] = tsdate.get_sites_time(ts)
    copy.finalise()


def make_sampledata_compatible(args):
    """
    Make a list of sampledata files compatible with the first file.
    """

    # Load all the sampledata files into a list
    print(
        "Subset sites with {} sampledata files".format(len(args.input_sampledata) - 1)
    )
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
        print(
            "Subsetted to {} sites from {}. Output can be found at {}.".format(
                len(keep_sites), fn, newname
            )
        )


def add_indiv_times(args):
    """
    Takes samples 'age' in metadata and add to individuals_time[:]
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
    for index, other_samples in enumerate(samples[1:]):
        print("Loaded sampledata file # {}".format(index))
        intersect_sites = np.isin(
            merged_samples.sites_position[:], other_samples.sites_position[:]
        )
        other_intersect_sites = np.where(
            np.isin(other_samples.sites_position[:], merged_samples.sites_position[:])
        )[0]
        other_samples_metadata = other_samples.sites_metadata[:]
        for site_index, site_metadata in zip(
            other_intersect_sites, merged_samples.sites_metadata[:][intersect_sites]
        ):
            other_samples_metadata[site_index] = site_metadata
        other_samples_copy = other_samples.copy()
        other_samples_copy.sites_metadata[:] = other_samples_metadata
        other_samples_copy.finalise()
        merged_samples = merged_samples.merge(other_samples_copy)
        print("Merged sampledata file # {}".format(index))
    merged_copy = merged_samples.copy(args.output)
    merged_copy.finalise()


def remove_moderns_reich(args):
    samples = tsinfer.load(args.input)
    ancients = samples.subset(individuals=np.where(samples.individuals_time[:] != 0)[0])
    genos = ancients.sites_genotypes[:]
    sites = np.where(np.sum(genos == 1, axis=1) != 0)[0]
    ancients_pruned = ancients.subset(sites=sites)
    copy = ancients_pruned.copy(args.output)
    copy.finalise()


def keep_with_offset(keep, data, offset):
    """
    Used when filtering _offset columns in tables
    """
    # We need the astype here for 32 bit machines
    lens = np.diff(offset).astype(np.int32)
    return (
        data[np.repeat(keep, lens)],
        np.concatenate(
            [
                np.array([0], dtype=offset.dtype),
                np.cumsum(lens[keep], dtype=offset.dtype),
            ]
        ),
    )


def get_provenance_dict(parameters=None):
    """
    Returns a dictionary encoding an execution of tskit conforming to the
    provenance schema.
    """
    document = {
        "schema_version": "1.0.0",
        "software": {"name": "tskit", "version": tskit.__version__},
        "parameters": parameters,
    }
    return document


def delete_site_mutations(tables, site_ids, record_provenance=True):
    """
    Remove the mutations at the specified sites entirely from the mutations table in
    this collection.
    :param list[int] site_ids: A list of site IDs specifying the sites whose
        mutations will be removed.
    :param bool record_provenance: If ``True``, add details of this operation
        to the provenance table in this TableCollection. (Default: ``True``).
    """
    keep_sites = np.ones(len(tables.sites), dtype=bool)
    site_ids = site_ids.astype(np.int32)
    if np.any(site_ids < 0) or np.any(site_ids >= len(tables.sites)):
        raise ValueError("Site ID out of bounds")
    keep_sites[site_ids] = 0
    keep_mutations = keep_sites[tables.mutations.site]
    new_ds, new_ds_offset = keep_with_offset(
        keep_mutations,
        tables.mutations.derived_state,
        tables.mutations.derived_state_offset,
    )
    new_md, new_md_offset = keep_with_offset(
        keep_mutations, tables.mutations.metadata, tables.mutations.metadata_offset
    )
    # Mutation numbers will change, so the parent references need altering
    mutation_map = np.cumsum(keep_mutations, dtype=tables.mutations.parent.dtype) - 1
    # Map parent == -1 to -1, and check this has worked (assumes tskit.NULL == -1)
    mutation_map = np.append(mutation_map, -1).astype(tables.mutations.parent.dtype)
    assert mutation_map[tskit.NULL] == tskit.NULL
    tables.mutations.set_columns(
        site=tables.mutations.site[keep_mutations],
        node=tables.mutations.node[keep_mutations],
        time=tables.mutations.time[keep_mutations],
        derived_state=new_ds,
        derived_state_offset=new_ds_offset,
        parent=mutation_map[tables.mutations.parent[keep_mutations]],
        metadata=new_md,
        metadata_offset=new_md_offset,
    )
    if record_provenance:
        # TODO replace with a version of https://github.com/tskit-dev/tskit/pull/243
        parameters = {"command": "delete_site_mutations", "TODO": "add parameters"}
        tables.provenances.add_row(record=json.dumps(get_provenance_dict(parameters)))
    return tables


def combined_ts_constrained_samples(args):
    modern_samples = tsinfer.load(args.modern)
    high_cov_samples = tsinfer.load(args.high_cov)
    all_ancient_samples = tsinfer.load(args.all_samples)
    dated_hgdp_1kg_sgdp_ts = tskit.load(args.dated_ts)
    sites_time = tsdate.sites_time_from_ts(dated_hgdp_1kg_sgdp_ts)
    # Only look at sites where the same alleles are found in ancients and moderns. This means the site must be biallelic in moderns and the derived allele is shared between moderns and ancients
    alleles_equal = np.full(high_cov_samples.num_sites, False, dtype=bool)
    for index, (modern_alleles, high_cov_alleles, all_ancient_alleles) in enumerate(
        zip(
            modern_samples.sites_alleles[:],
            high_cov_samples.sites_alleles[:],
            all_ancient_samples.sites_alleles[:],
        )
    ):
        modern_alleles = [i for i in modern_alleles if i]
        high_cov_alleles = [i for i in high_cov_alleles if i]
        all_ancient_alleles = [i for i in all_ancient_alleles if i]
        if modern_alleles == high_cov_alleles == all_ancient_alleles:
            alleles_equal[index] = True

    # Get the ancient bounds from sampledata file of all ancients
    all_ancient_samples_bound = all_ancient_samples.min_site_times(
        individuals_only=True
    )
    high_cov_samples_bound = high_cov_samples.min_site_times(individuals_only=True)
    # Assert that the all ancient samples files has same or older ancient bounds than only high cov
    assert np.all(all_ancient_samples_bound >= high_cov_samples_bound)
    print(
        "Number of ancient lower bounds (with multiallelic sites): ",
        np.sum(all_ancient_samples_bound != 0),
    )

    # Set time of non-biallelic sites to 0
    all_ancient_samples_bound[~alleles_equal] = 0
    # Constrain the estimated ages from tree sequence with ancient bounds
    constrained_sites_time = np.maximum(sites_time, all_ancient_samples_bound)
    # Add constrained times to sampledata file with moderns and high cov ancients
    dated_samples = tsdate.add_sampledata_times(
        high_cov_samples, constrained_sites_time
    )
    # Record number of constrained sites
    print("Total number of sites: ", sites_time.shape[0])
    print(
        "Number of ancient lower bounds: ",
        np.sum(all_ancient_samples_bound != 0),
    )
    print(
        "Number of corrected times: ", np.sum(dated_samples.sites_time[:] != sites_time)
    )
    high_cov_samples_copy = dated_samples.copy(args.output)
    high_cov_samples_copy.finalise()


def split_chromosome(args):
    match = re.search(r"(chr\d+)", args.chrom)
    if match is None:
        raise ValueError("chr must be in filename")
    chrom = match.group(1)
    with open(args.centromeres) as csvfile:
        reader = csv.DictReader(csvfile)
        centromere_positions = list()
        for row in reader:
            if row["chrom"] == chrom:
                centromere_positions.append(int(row["chromStart"]))
                centromere_positions.append(int(row["chromEnd"]))
    start = np.min(centromere_positions)
    end = np.max(centromere_positions)
    split_point = (start + end) / 2
    samples = tsinfer.load(args.input)
    position = samples.sites_position[:]
    print(f"Splitting at {split_point}")
    if args.arm == "p":
        keep_sites = np.where(position < split_point)[0]
        print(f"Keeping {keep_sites.shape[0]} sites")
        arm = samples.subset(sites=keep_sites)
        snipped_samples = arm.copy(path=args.output)
        snipped_samples.data.attrs["sequence_length"] = split_point
    elif args.arm == "q":
        keep_sites = np.where(position > split_point)[0]
        print(f"Keeping {keep_sites.shape[0]} sites")
        arm = samples.subset(sites=keep_sites)
        snipped_samples = arm.copy(path=args.output)
    snipped_samples.finalise()


def combine_chromosome_arms(args):
    """
    Splices two chromosome arms together to form a full chromosome
    """
    short_arm = tskit.load(args.p_arm)
    long_arm = tskit.load(args.q_arm)
    assert short_arm.num_samples == long_arm.num_samples
    # Remove material before first position and after last position
    short_arm = short_arm.keep_intervals(
        [
            [
                short_arm.tables.sites.position[0] - 1,
                short_arm.tables.sites.position[-1] + 1,
            ]
        ],
        simplify=False,
    )
    long_arm = long_arm.keep_intervals(
        [
            [
                long_arm.tables.sites.position[0] - 1,
                long_arm.tables.sites.position[-1] + 1,
            ]
        ],
        simplify=False,
    )
    short_tables = short_arm.dump_tables()
    long_tables = long_arm.dump_tables()
    assert np.array_equal(
        short_tables.individuals.metadata, long_tables.individuals.metadata
    )
    short_tables.sequence_length = long_arm.get_sequence_length()
    short_metadata = short_tables.nodes.metadata
    short_metadata_offset = short_tables.nodes.metadata_offset
    short_metadata = tskit.unpack_bytes(short_metadata, short_metadata_offset)

    long_metadata = long_tables.nodes.metadata
    long_metadata_offset = long_tables.nodes.metadata_offset
    long_metadata = tskit.unpack_bytes(long_metadata, long_metadata_offset)
    long_metadata = long_metadata[long_arm.num_samples :]
    combined_metadata = np.concatenate([short_metadata, long_metadata])
    metadata, metadata_offset = tskit.pack_bytes(combined_metadata)

    all_nodes_except_samples = ~np.isin(
        np.arange(long_arm.num_nodes), long_arm.samples()
    )
    short_tables.nodes.append_columns(
        long_tables.nodes.flags[all_nodes_except_samples],
        long_tables.nodes.time[all_nodes_except_samples],
        long_tables.nodes.population[all_nodes_except_samples],
    )
    short_tables.nodes.set_columns(
        flags=short_tables.nodes.flags,
        time=short_tables.nodes.time,
        population=short_tables.nodes.population,
        metadata=metadata,
        individual=short_tables.nodes.individual,
        metadata_offset=metadata_offset,
    )

    long_edges_parent = long_tables.edges.parent
    long_edges_child = long_tables.edges.child
    long_arm_sample_map = np.zeros(long_arm.num_nodes).astype(int)
    long_arm_sample_map[long_arm.samples()] = short_arm.samples()
    long_edges_parent[
        ~np.isin(long_edges_parent, long_arm.samples())
    ] = long_edges_parent[~np.isin(long_edges_parent, long_arm.samples())] + (
        short_arm.num_nodes
    )
    long_edges_parent[long_arm.tables.edges.parent > long_arm.samples()[-1]] = (
        long_edges_parent[long_arm.tables.edges.parent > long_arm.samples()[-1]]
        - long_arm.num_samples
    )
    long_edges_child[~np.isin(long_edges_child, long_arm.samples())] = long_edges_child[
        ~np.isin(long_edges_child, long_arm.samples())
    ] + (short_arm.num_nodes)
    long_edges_child[long_tables.edges.child > long_arm.samples()[-1]] = (
        long_edges_child[long_tables.edges.child > long_arm.samples()[-1]]
        - long_arm.num_samples
    )
    long_edges_child[
        np.isin(long_tables.edges.child, long_arm.samples())
    ] = long_arm_sample_map[
        long_tables.edges.child[np.isin(long_tables.edges.child, long_arm.samples())]
    ]
    short_tables.edges.append_columns(
        long_tables.edges.left,
        long_tables.edges.right,
        long_edges_parent,
        long_edges_child,
    )
    short_tables.sites.append_columns(
        long_tables.sites.position,
        long_tables.sites.ancestral_state,
        long_tables.sites.ancestral_state_offset,
    )
    long_mutations_node = long_tables.mutations.node
    long_mutations_node[
        ~np.isin(long_mutations_node, long_arm.samples())
    ] = long_mutations_node[~np.isin(long_mutations_node, long_arm.samples())] + (
        short_arm.num_nodes
    )
    long_mutations_node[long_tables.mutations.node > long_arm.samples()[-1]] = (
        long_mutations_node[long_tables.mutations.node > long_arm.samples()[-1]]
        - long_arm.num_samples
    )
    long_mutations_node[
        np.isin(long_tables.mutations.node, long_arm.samples())
    ] = long_arm_sample_map[
        long_tables.mutations.node[
            np.isin(long_tables.mutations.node, long_arm.samples())
        ]
    ]
    short_tables.mutations.append_columns(
        long_tables.mutations.site + short_arm.num_sites,
        long_mutations_node,
        long_tables.mutations.derived_state,
        long_tables.mutations.derived_state_offset,
    )

    short_tables.sort()
    combined = short_tables.tree_sequence()
    assert combined.num_nodes == (
        short_arm.num_nodes + long_arm.num_nodes - short_arm.num_samples
    )
    assert combined.num_sites == (short_arm.num_sites + long_arm.num_sites)
    assert combined.num_edges == (short_arm.num_edges + long_arm.num_edges)
    assert combined.num_mutations == (short_arm.num_mutations + long_arm.num_mutations)
    assert (
        combined.num_individuals
        == short_arm.num_individuals
        == long_arm.num_individuals
    )
    assert np.array_equal(
        np.sort(combined.tables.sites.position),
        np.concatenate(
            [short_arm.tables.sites.position, long_arm.tables.sites.position]
        ),
    )
    assert np.array_equal(
        np.sort(combined.tables.nodes.time[combined.tables.mutations.node]),
        np.sort(
            np.concatenate(
                [
                    short_arm.tables.nodes.time[short_arm.tables.mutations.node],
                    long_arm.tables.nodes.time[long_arm.tables.mutations.node],
                ]
            )
        ),
    )
    assert np.array_equal(
        combined.tables.individuals.metadata, long_tables.individuals.metadata
    )
    combined.dump(args.output)


def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "command"

    subparser = subparsers.add_parser("simplify")
    subparser.add_argument("input", type=str, help="Input tree sequence")
    subparser.add_argument("output", type=str, help="Input tree sequence")
    subparser.set_defaults(func=run_simplify)

    subparser = subparsers.add_parser("dated_samples")
    subparser.add_argument("samples", type=str, help="Input sampledata")
    subparser.add_argument("ts", type=str, help="Input dated tree sequence")
    subparser.set_defaults(func=run_get_dated_samples)

    subparser = subparsers.add_parser("make-sampledata-compatible")
    subparser.add_argument(
        "--input-sampledata",
        type=str,
        nargs="+",
        help="Input sample files to merge.",
        required=True,
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
        nargs="+",
        help="Input sample files to merge.",
        required=True,
    )
    subparser.add_argument("--output", type=str, required=True)
    subparser.set_defaults(func=merge_sampledata_files)

    subparser = subparsers.add_parser("remove-moderns-reich")
    subparser.add_argument(
        "input",
        type=str,
        help="Input reich sampledata file with moderns and ancients.",
    )
    subparser.add_argument("output", type=str, help="Output sampledata file name")
    subparser.set_defaults(func=remove_moderns_reich)

    subparser = subparsers.add_parser("combined-ts-dated-samples")
    subparser.add_argument("--modern", type=str, help="HGDP + 1kg + SGDP samples")
    subparser.add_argument(
        "--high-cov",
        type=str,
        help="HGDP + 1kg + SGDP + High-Coverage Ancients.",
    )
    subparser.add_argument(
        "--all-samples",
        type=str,
        help="HGDP + 1kg + SGDP + All Ancients.",
    )
    subparser.add_argument(
        "--dated-ts",
        type=str,
        help="HGDP + 1kg + SGDP Dated Tree Sequence.",
    )
    subparser.add_argument("--output", type=str, help="Output sampledata filename")
    subparser.set_defaults(func=combined_ts_constrained_samples)

    subparser = subparsers.add_parser("split-chromosome")
    subparser.add_argument("input", type=str, help="Input SampleData")
    subparser.add_argument("output", type=str, help="Output SampleData")
    subparser.add_argument("chrom", type=str, help="Chromosome name")
    subparser.add_argument("arm", type=str, help="Arm name")
    subparser.add_argument(
        "centromeres", type=str, help="CSV file containing centromere coordinates."
    )
    subparser.set_defaults(func=split_chromosome)

    subparser = subparsers.add_parser("combine-chromosome")
    subparser.add_argument("p_arm", type=str, help="Input p Arm")
    subparser.add_argument("q_arm", type=str, help="Input q Arm")
    subparser.add_argument("output", type=str, help="Output filename")
    subparser.set_defaults(func=combine_chromosome_arms)

    daiquiri.setup(level="INFO")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

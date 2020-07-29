import os.path
import argparse
import collections
import re
import time

import tskit
import msprime
import numpy as np
import tsinfer
import stdpopsim

Params = collections.namedtuple(
    "Params",
    "sample_file, ma_mis_rate, ms_mis_rate, cutoff_exponent, precision, "
    "genetic_map, num_threads")

Results = collections.namedtuple(
    "Results",
    "abs_ma_mis, abs_ms_mis, rel_ma_mis, rel_ms_mis, cutoff_exponent, "
    "precision, edges, muts, num_trees, "
    "kc, mean_node_children, var_node_children, process_time, ts_size, ts_path")

def run(params):
    """
    Run a single inference, with the specified rates
    """
    samples = tsinfer.load(params.sample_file)
    chr_map = get_genetic_map(params.sample_file, params.genetic_map)
    start_time = time.process_time()
    extra_params =  dict(num_threads=params.num_threads)
    if params.cutoff_exponent is not None:
        extra_params['cutoff_power'] = params.cutoff_exponent
    anc = tsinfer.generate_ancestors(
        samples,
        progress_monitor=tsinfer.cli.ProgressMonitor(1, 1, 0, 0, 0),
        **extra_params,
    )
    print(f"GA done (cutoff exponent: {params.cutoff_exponent}")
    inference_pos = anc.sites_position[:]
    if chr_map is not None:
        inference_distances = physical_to_genetic(chr_map, inference_pos)
        d = np.diff(inference_distances)
        rho = np.concatenate(([0.0], d))
    else:
        inference_distances = inference_pos
        d = np.diff(inference_distances)
        rho = np.concatenate(
	    ([0.0], d/anc.sequence_length))

    av_rho = np.quantile(rho, 0.5)
    ma_mis = av_rho * params.ma_mis_rate
    ms_mis = av_rho * params.ms_mis_rate
    if params.precision is None:
        # Smallest nonzero recombination rate
        min_rho = int(np.ceil(-np.min(np.log10(rho[rho > 0]))))
        # Smallest mean
        av_min = int(np.ceil(-np.log10(min(ma_mis, ms_mis))))
        precision = max(min_rho, av_min) + 3
    else:
        precision = params.precision
    print(
        f"Starting match_ancestors, {params.ma_mis_rate} {params.ms_mis_rate}",
        f"with av rho {av_rho:.5g}",
        f"(mean {np.mean(rho):.4g}, median {np.quantile(rho, 0.5):.4g}, ",
        f"nonzero min {np.min(rho[rho > 0]):.4g}, ",
        f"2.5% quantile {np.quantile(rho, 0.025):.4g}) precision {precision}")
    prefix = None
    if params.sample_file is not None:
        assert params.sample_file.endswith(".samples")
        prefix = params.sample_file[0:-len(".samples")]
        #inf_prefix = "{}_ma{}_ms{}_N{}_p{}".format(
        #    prefix,
        #    params.ma_mis_rate,
        #    params.ms_mis_rate,
        #    params.cutoff_exponent,
        #    precision)
    extra_params =  dict(
        num_threads=params.num_threads,
        recombination_rate=rho,
        precision=precision,
    )

    if prefix is not None:
        anc_copy = anc.copy(prefix + ".ancestors")
        anc_copy.finalise()
    if np.any(samples.individuals_time[:] > 0) and not np.any(samples.sites_time[:] == tsinfer.constants.TIME_UNSPECIFIED):
        inferred_anc_ts = match_ancestors_including_noncontemporanous_samples(
            samples,
            anc,
            mismatch_rate=ma_mis,
            create_sample_nodes=True,
            progress_monitor=tsinfer.cli.ProgressMonitor(1, 0, 1, 0, 0),
            **extra_params,
        )
    else:
        inferred_anc_ts = tsinfer.match_ancestors(
            samples,
            anc,
            mismatch_rate=ma_mis,
            progress_monitor=tsinfer.cli.ProgressMonitor(1, 0, 1, 0, 0),
            **extra_params,
        )
    inferred_anc_ts.dump(path=prefix + ".atrees")
    print(f"MA done (ma_mis:{ma_mis}")
    inferred_ts = tsinfer.match_samples(
        samples,
        inferred_anc_ts,
        mismatch_rate=ms_mis,
        simplify=False,
        progress_monitor=tsinfer.cli.ProgressMonitor(1, 0, 0, 0, 1),
        **extra_params,
    )
    process_time = time.process_time() - start_time
    ts_path = prefix + ".trees"
    inferred_ts.dump(path=ts_path)
    print(f"MS done: ms_mis rate = {ms_mis})")
    simplified_inferred_ts = inferred_ts.simplify()  # Remove unary nodes
    ts_path = prefix + ".simplified.trees"
    simplified_inferred_ts.dump(path=ts_path)
    
def physical_to_genetic(recombination_map, input_physical_positions):
    map_pos = recombination_map.get_positions()
    map_rates = recombination_map.get_rates()
    map_genetic_positions = np.insert(np.cumsum(np.diff(map_pos) * map_rates[:-1]), 0, 0)
    return np.interp(input_physical_positions, map_pos, map_genetic_positions)


def get_genetic_map(filename, genetic_map=None):
    """
    Return a Thousand Genomes Project sample data file, the
    corresponding recombination rate array, a prefix to use for files, and None
    """
    filename = args.sample_file
    map = args.genetic_map
    if not filename.endswith(".samples"):
        raise ValueError("Sample data file must end with '.samples'")

    match = re.search(r'(chr\d+)', filename)
    if match or map is not None:
        if map is not None:
            if match is not None:
                chr = match.group(1)
                print(f"Using {chr} from GRCh38 for the recombination map")
                chr_map = msprime.RecombinationMap.read_hapmap(map + chr + ".txt")
            else:
                chr_map = msprime.RecombinationMap.read_hapmap(map)
        else:
            chr = match.group(1)
            print(f"Using {chr} from HapMapII_GRCh37 for the recombination map")
            map = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
            if not map.is_cached():
                map.download()
            chr_map = map.get_chromosome_map(chr)

    return chr_map 

def match_ancestors_including_noncontemporanous_samples(
    sample_data, ancestor_data, *, create_sample_nodes=None, **kwargs,
):
    """
    Like match_ancestors, but also include as ancestral nodes those samples in the
    sample_data file from individuals whose time is > 0. These "non-contemporaneous"
    samples are added into the list of ancestors before running match_ancestors. This
    is only possible if the times of mutations used for inference (the site times in the
    ancestor_data file) have been explicitly set by the user, e.g. by using the
    ``time`` parameter of ``SampleData.add_sites()``, or by setting ``use_times=True`` if
    creating a sample data file via ``tsinfer.SampleData.from_tree_sequence()``.
    If ``create_sample_nodes`` is True, the inserted ancestors will be flagged with
    tsinfer.NODE_IS_TRUE_SAMPLE_ANCESTOR, and the individuals associated with the
    non-contemporaneous samples will be present in the resulting ancestors tree sequence.
    This will create internal sample nodes in the final tree sequence produced by
    match_samples. If create_sample_nodes is False, the inserted ancestors are treated as
    *proxies* for the true samples (i.e. close, unrecombined relatives of the
    true samples) and the nodes will not be contained within an individual in the final
    tree sequence.
    Returns an ancestors tree sequence suitable for input into match_samples. If
    ``create_sample_nodes`` is True, you probably want to exclude any noncontemporaneous
    samples during the match_samples phase, otherwise they risk being duplicated in the
    final tree sequence.
    """

    # First find the sites from the samples file used in the built ancestors
    sites_used = np.isin(sample_data.sites_position[:], ancestor_data.sites_position[:])
    sites_used_time = sample_data.sites_time[:][sites_used]
    if np.any(sites_used_time == tsinfer.constants.TIME_UNSPECIFIED):
        raise ValueError(
            "Noncontemporaneous samples can only be ancestors if site times are present"
        )
    individuals_population = sample_data.individuals_population[:]

    # Find the non-contemporaneous (nc) samples and group them by individuals_time
    samples_individual = sample_data.samples_individual[:]
    nc_individuals = sample_data.individuals_time[:] > 0
    nc_samples = np.where(np.isin(samples_individual, np.where(nc_individuals)[0]))[0]
    samples_by_timepoint = collections.defaultdict(list)
    for s in nc_samples:
        timepoint = sample_data.individuals_time[samples_individual[s]]
        samples_by_timepoint[timepoint].append(s)

    logger.info("Creating new ancestors file with added ancestors")
    ancestor_to_sample_map = {}
    inferred_ancestors = ancestor_data.ancestors()
    current_anc = next(inferred_ancestors)
    sample_data_subset = sample_data.subset(sites=np.where(sites_used)[0])
    with tsinfer.formats.AncestorData(sample_data=sample_data_subset) as new_data:
        for insert_time, sample_ids in sorted(samples_by_timepoint.items()):
            while current_anc is not None and current_anc.time > insert_time:
                new_data.add_ancestor(**attr.asdict(current_anc, filter=exclude_id))
                current_anc = next(inferred_ancestors, None)
            for s in sample_ids:
                logger.debug(f"Adding sample {s} as an ancestor")
                haplotype = sample_data.sites_genotypes[:, s][sites_used]
                if np.any(sites_used_time[haplotype > 0] < insert_time):
                    bad = sites_used_time[haplotype > 0] < insert_time
                    bad_pos = sample_data.sites_position[:][bad]
                    raise ValueError(
                        "Non-contemporanous ancestor is older than its derived mutations"
                        + f" at sites {bad_pos}"
                    )
                anc_id = new_data.add_ancestor(
                    start=0,
                    end=ancestor_data.num_sites,
                    time=insert_time,
                    focal_sites=[],
                    haplotype=haplotype,
                )
                ancestor_to_sample_map[anc_id] = s
    logger.info("Matching ancestors including noncontemporaneous samples")
    ancestor_tables = tsinfer.match_ancestors(
        sample_data, new_data, **kwargs
    ).dump_tables()
    nodes_flags = ancestor_tables.nodes.flags

    if create_sample_nodes is True:
        logger.info("Adding associated individuals and populations")

        ancestor_individuals = ancestor_tables.individuals
        ancestor_populations = ancestor_tables.populations
        assert ancestor_individuals.num_rows == 0
        assert ancestor_populations.num_rows == 0

        # Insert individuals into ancestors tree sequence
        old_to_new_indiv = {}
        for i in np.unique(samples_individual[nc_samples]):
            old_to_new_indiv[i] = ancestor_individuals.add_row(
                flags=0,
                location=sample_data.individual(i).location,
                metadata=json.dumps(sample_data.individual(i).metadata).encode(),
            )
        # Insert populations into ancestors tree sequence
        old_to_new_pop = {}
        for p in np.unique(individuals_population[nc_individuals]):
            old_to_new_pop[p] = ancestor_populations.add_row(
                metadata=json.dumps(sample_data.population(p).metadata).encode()
            )

        logger.info("Setting node information for true sample ancestors")
        nodes = ancestor_tables.nodes
        nodes_individual = nodes.individual
        nodes_population = nodes.population
        for anc_id, samp_id in ancestor_to_sample_map.items():
            i = samples_individual[samp_id]
            # The ids of the ancestors in the a_ts should correspond to the ones in the
            # ancestors file, as all the pc ancestors are at the end but we check anyway
            assert anc_id == json.loads(nodes[anc_id].metadata)["ancestor_data_id"]
            nodes_individual[anc_id] = old_to_new_indiv[i]
            nodes_population[anc_id] = old_to_new_pop[individuals_population[i]]
            nodes_flags[anc_id] = np.bitwise_or(
                nodes_flags[anc_id], NODE_IS_TRUE_SAMPLE_ANCESTOR,
            )
        ancestor_tables.nodes.individual = nodes_individual
        ancestor_tables.nodes.population = nodes_population

    else:
        logger.info("Setting node information for proxy sample ancestors")
        for anc_id, _ in ancestor_to_sample_map.items():
            nodes_flags[anc_id] = np.bitwise_or(
                nodes_flags[anc_id], NODE_IS_PROXY_SAMPLE_ANCESTOR,
            )

    ancestor_tables.nodes.flags = nodes_flags
    return ancestor_tables.tree_sequence()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_file", default=None,
        help="A tsinfer sample file ending in '.samples'. If no genetic map is provided"
            " via the -m switch, and the filename contains chrNN"
            " where 'NN' is a number, assume this is a human samples file and use the"
            " appropriate recombination map from the thousand genomes project, otherwise"
            " use the physical distance between sites.")
    # The _mrate parameter defaults set from analysis ot 1000G, see
    # https://github.com/tskit-dev/tsinfer/issues/263#issuecomment-639060101
    parser.add_argument("-A", "--mismatch_ancestors", type=float, default=0.1,
        help="The mismatch probability in the match ancestors phase,"
            " as a fraction of the median recombination probability between sites")
    parser.add_argument("-S", "--mismatch_samples", type=float, default=0.1,
        help="The mismatch probability in the match samples phase,"
            " as a fraction of the median recombination probability between sites")
    parser.add_argument("-p", "--precision", type=int, default=None,
        help="The precision, as a number of decimal places, which will affect the speed"
            " of the matching algorithm (higher precision = lower speed). If None,"
            " calculate the smallest of the recombination rates or mismatch rates, and"
            " use the negative exponent of that number plus one. E.g. if the smallest"
            " recombination rate is 2.5e-6, use precision = 6+3 = 7")
    parser.add_argument("-t", "--num_threads", type=int, default=0,
        help="The number of threads to use in inference")
    parser.add_argument("-m", "--genetic_map", default=None,
        help="An alternative genetic map to be used for this analysis, in the format"
            "expected by msprime.RecombinationMap.read_hapmap")
    parser.add_argument("-x", "--cutoff_exponent", default=None, type=float,
        help="The value, x to be used as the exponenent of the freq, to shorten"
            "ancestor building")
    args = parser.parse_args()
    prefix = args.sample_file[:-len(".samples")]
    params = Params(
        args.sample_file,
        args.mismatch_ancestors,
        args.mismatch_samples,
        args.cutoff_exponent,
        args.precision,
        args.genetic_map,
        args.num_threads,
    )
    run(params)

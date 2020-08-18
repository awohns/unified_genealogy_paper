import os.path
import argparse
import collections
import re
import time
import logging

import tskit
import msprime
import numpy as np
import tsinfer
import stdpopsim

logger = logging.getLogger(__name__)

Params = collections.namedtuple(
    "Params",
    "sample_file, ancestors_file, ma_mis_rate, ms_mis_rate, cutoff_exponent, precision, "
    "genetic_map, ancestors_ts, modern_samples_match, ancient_ancestors, num_threads")

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
    anc = tsinfer.load(params.ancestors_file)
#    anc = tsinfer.generate_ancestors(
#        samples,
#        progress_monitor=tsinfer.cli.ProgressMonitor(1, 1, 0, 0, 0),
#        **extra_params,
#    )
#    print(f"GA done (cutoff exponent: {params.cutoff_exponent}")
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
    if params.ma_mis_rate is not None:
        ma_mis_rate = params.ma_mis_rate
    else:
        ma_mis_rate = 0.1
    if params.ms_mis_rate is not None:
        ms_mis_rate = params.ms_mis_rate
    else:
        ms_mis_rate = 0.1
    ma_mis = av_rho * ma_mis_rate
    ms_mis = av_rho * ms_mis_rate
    if params.precision is None:
        # Smallest nonzero recombination rate
        min_rho = int(np.ceil(-np.min(np.log10(rho[rho > 0]))))
        # Smallest mean
        av_min = int(np.ceil(-np.log10(min(ma_mis, ms_mis))))
        precision = max(min_rho, av_min) + 3
    else:
        precision = params.precision
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

#    if prefix is not None:
#        anc_copy = anc.copy(prefix + ".ancestors")
#        anc_copy.finalise()
    if params.ma_mis_rate is not None:
        print(
            f"Starting match_ancestors, {params.ma_mis_rate} {params.ms_mis_rate}",
            f"with av rho {av_rho:.5g}",
            f"(mean {np.mean(rho):.4g}, median {np.quantile(rho, 0.5):.4g}, ",
            f"nonzero min {np.min(rho[rho > 0]):.4g}, ",
            f"2.5% quantile {np.quantile(rho, 0.025):.4g}) precision {precision}")

        if params.ancient_ancestors:
            anc = anc.insert_proxy_samples(samples, allow_mutation=True)
        inferred_anc_ts = tsinfer.match_ancestors(
            samples,
            anc,
            mismatch_rate=ma_mis,
            progress_monitor=tsinfer.cli.ProgressMonitor(1, 0, 1, 0, 0),
            **extra_params,
        )
        inferred_anc_ts.dump(path=prefix + ".atrees")
        print(f"MA done (ma_mis:{ma_mis}")
    if params.ancestors_ts is not None:
        inferred_anc_ts = tskit.load(params.ancestors_ts)
    if params.ms_mis_rate is not None and params.modern_samples_match:
        samples = samples.subset(np.where(samples.individuals_time[:] == 0)[0])
        prefix = prefix + ".modern"
    if params.ms_mis_rate is not None and params.ancient_ancestors:
        prefix = prefix + ".historic"
        force_sample_times = True
    else:
        force_sample_times = False

    if params.ms_mis_rate is not None:
        inferred_ts = tsinfer.match_samples(
            samples,
            inferred_anc_ts,
            mismatch_rate=ms_mis,
            simplify=False,
            progress_monitor=tsinfer.cli.ProgressMonitor(1, 0, 0, 0, 1),
            force_sample_times=force_sample_times,
            **extra_params,
        )
        process_time = time.process_time() - start_time
        ts_path = prefix + ".nosimplify.trees"
        inferred_ts.dump(path=ts_path)
        print(f"MS done: ms_mis rate = {ms_mis})")
        simplified_inferred_ts = inferred_ts.simplify(filter_sites=False)  # Remove unary nodes
        ts_path = prefix + ".trees"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_file", default=None,
        help="A tsinfer sample file ending in '.samples'. If no genetic map is provided"
            " via the -m switch, and the filename contains chrNN"
            " where 'NN' is a number, assume this is a human samples file and use the"
            " appropriate recombination map from the thousand genomes project, otherwise"
            " use the physical distance between sites.")
    parser.add_argument("ancestors_file", default=None,
        help="A tsinfer ancestors file ending in '.ancestors'") 
    # The _mrate parameter defaults set from analysis ot 1000G, see
    # https://github.com/tskit-dev/tsinfer/issues/263#issuecomment-639060101
    parser.add_argument("-A", "--mismatch_ancestors", type=float, default=None,
        help="The mismatch probability in the match ancestors phase,"
            " as a fraction of the median recombination probability between sites")
    parser.add_argument("-S", "--mismatch_samples", type=float, default=None,
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
    parser.add_argument("--ancestors-ts", default=None,
            help="An inferred ancestors tree sequence to match samples against")
    parser.add_argument("--ancient-ancestors", action="store_true",
            help="If true, ancients are added as ancestors in match_ancestors")
    parser.add_argument("--modern-samples-match", action="store_true",
            help="If true, only match modern samples to ancestors tree sequence.")

    args = parser.parse_args()
    prefix = args.sample_file[:-len(".samples")]
    if args.ancestors_ts is not None and args.mismatch_ancestors is not None:
        raise ValueError("Cannot match ancestors when providing ancestors ts")
    params = Params(
        args.sample_file,
        args.ancestors_file,
        args.mismatch_ancestors,
        args.mismatch_samples,
        args.cutoff_exponent,
        args.precision,
        args.genetic_map,
        args.ancestors_ts,
        args.modern_samples_match,
        args.ancient_ancestors,
        args.num_threads,
    )
    run(params)

import argparse
import re
import logging

import tskit
import msprime
import numpy as np
import tsinfer
import stdpopsim

logger = logging.getLogger(__name__)


def get_rho(samples, ancestors, chr_map, ma_mis_rate=0.1,
            ms_mis_rate=0.1, precision=None, num_threads=1):
    inference_pos = ancestors.sites_position[:]
    if chr_map is not None:
        inference_distances = physical_to_genetic(chr_map, inference_pos)
        d = np.diff(inference_distances)
        rho = np.concatenate(([0.0], d))
    else:
        inference_distances = inference_pos
        d = np.diff(inference_distances)
        rho = np.concatenate(([0.0], d/ancestors.sequence_length))

    av_rho = np.quantile(rho, 0.5)
    ma_mis = av_rho * ma_mis_rate
    ms_mis = av_rho * ms_mis_rate
    if precision is None:
        # Smallest nonzero recombination rate
        min_rho = int(np.ceil(-np.min(np.log10(rho[rho > 0]))))
        # Smallest mean
        av_min = int(np.ceil(-np.log10(min(ma_mis, ms_mis))))
        precision = max(min_rho, av_min) + 3
    else:
        precision = precision
    return rho, ma_mis, ms_mis, precision


def match_ancestors(samples, ancestors, rho, ma_mis,
                    precision, prefix=None, ancient_ancestors=False, num_threads=1, progress=False):
    av_rho = np.quantile(rho, 0.5)
    extra_params = dict(
        num_threads=num_threads,
        recombination_rate=rho,
        precision=precision,
    )

    print(
        f"Starting match_ancestors, {ma_mis} "
        f"with av rho {av_rho:.5g}",
        f"(mean {np.mean(rho):.4g}, median {np.quantile(rho, 0.5):.4g}, ",
        f"nonzero min {np.min(rho[rho > 0]):.4g}, ",
        f"2.5% quantile {np.quantile(rho, 0.025):.4g}) precision {precision}")

    if ancient_ancestors:
        ancestors = ancestors.insert_proxy_samples(samples, allow_mutation=True)
        copy = ancestors.copy(path=prefix + ".proxy.ancestors")
        copy.finalise()
        path_compression = False
    else:
        path_compression = True 
    if progress:
        progress_bar=tsinfer.cli.ProgressMonitor(1, 0, 0, 0, 1)
    else:
        progress_bar = None

    inferred_anc_ts = tsinfer.match_ancestors(
        samples,
        ancestors,
        mismatch_rate=ma_mis,
        progress_monitor=progress_bar,
        path_compression=path_compression,
        **extra_params,
    )
    if prefix is not None:
        inferred_anc_ts.dump(path=prefix + ".atrees")
    print(f"MA done (ma_mis:{ma_mis}")
    return inferred_anc_ts


def match_samples(samples, inferred_anc_ts, rho, ms_mis, precision, prefix=None,
                  modern_samples_match=False, ancient_ancestors=False, num_threads=1, progress=False):
    if modern_samples_match:
        samples = samples.subset(np.where(samples.individuals_time[:] == 0)[0])
        if prefix is not None:
            prefix = prefix + ".modern"
    if ancient_ancestors:
        if prefix is not None:
            prefix = prefix + ".historic"
        force_sample_times = True
    else:
        force_sample_times = False
    extra_params = dict(
        num_threads=num_threads,
        recombination_rate=rho,
        precision=precision,
    )
    if progress:
        progress_bar=tsinfer.cli.ProgressMonitor(1, 0, 0, 0, 1)
    else:
        progress_bar=None

    inferred_ts = tsinfer.match_samples(
        samples,
        inferred_anc_ts,
        mismatch_rate=ms_mis,
        simplify=False,
        progress_monitor=progress_bar,
        force_sample_times=force_sample_times,
        **extra_params,
    )
    if prefix is not None:
        ts_path = prefix + ".nosimplify.trees"
        inferred_ts.dump(path=ts_path)
    print(f"MS done: ms_mis rate = {ms_mis})")
    simplified_inferred_ts = inferred_ts.simplify(filter_sites=False)
    if prefix is not None:
        ts_path = prefix + ".trees"
        simplified_inferred_ts.dump(path=ts_path)
    return inferred_ts, simplified_inferred_ts


def physical_to_genetic(recombination_map, input_physical_positions):
    map_pos = recombination_map.get_positions()
    map_rates = recombination_map.get_rates()
    map_genetic_positions = np.insert(np.cumsum(np.diff(map_pos) * map_rates[:-1]), 0, 0)
    return np.interp(input_physical_positions, map_pos, map_genetic_positions)


def get_genetic_map(filename=None, genetic_map=None):
    """
    Return a Thousand Genomes Project sample data file, the
    corresponding recombination rate array, a prefix to use for files, and None
    """
    map = genetic_map
    match = None
    if filename is not None:
        match = re.search(r'(chr\d+)', filename)
    chr_map = None
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
                        help="A tsinfer sample file ending in '.samples'. If no genetic"
                        " map is provided via the -m switch, and the filename contains"
                        " chrNN where 'NN' is a number, assume this is a human samples"
                        " file and use the appropriate recombination map from the"
                        " thousand genomes project, otherwise use the physical distance"
                        " between sites.")
    parser.add_argument("ancestors_file", default=None,
                        help="A tsinfer ancestors file ending in '.ancestors'")
    # The _mrate parameter defaults set from analysis ot 1000G, see
    # https://github.com/tskit-dev/tsinfer/issues/263#issuecomment-639060101
    parser.add_argument("-A", "--mismatch_ancestors", type=float, default=None,
                        help="The mismatch probability in the match ancestors phase,"
                        " as a fraction of the median recombination probability between"
                        " sites")
    parser.add_argument("-S", "--mismatch_samples", type=float, default=None,
                        help="The mismatch probability in the match samples phase,"
                        " as a fraction of the median recombination probability between"
                        " sites")
    parser.add_argument("-p", "--precision", type=int, default=None,
                        help="The precision, as a number of decimal places, which will"
                        " affect the speed of the matching algorithm (higher precision"
                        " = lower speed). If None, calculate the smallest of the"
                        " recombination rates or mismatch rates, and use the negative"
                        " exponent of that number plus one. E.g. if the smallest"
                        " recombination rate is 2.5e-6, use precision = 6+3 = 7")
    parser.add_argument("-t", "--num_threads", type=int, default=0,
                        help="The number of threads to use in inference")
    parser.add_argument("-m", "--genetic_map", default=None,
                        help="An alternative genetic map to be used for this analysis,"
                        " in the format expected by"
                        " msprime.RecombinationMap.read_hapmap")
    parser.add_argument(
            "-x", "--cutoff_exponent", default=None, type=float,
            help="The value, x to be used as the exponenent of the freq, to shorten"
            "ancestor building")
    parser.add_argument(
            "--ancestors-ts", default=None,
            help="An inferred ancestors tree sequence to match samples against")
    parser.add_argument(
            "--ancient-ancestors", action="store_true",
            help="If true, ancients are added as ancestors in match_ancestors")
    parser.add_argument(
            "--modern-samples-match", action="store_true",
            help="If true, only match modern samples to ancestors tree sequence.")

    args = parser.parse_args()
    prefix = args.sample_file[:-len(".samples")]
    if args.ancestors_ts is not None and args.mismatch_ancestors is not None:
        raise ValueError("Cannot match ancestors when providing ancestors ts")
    assert args.sample_file.endswith(".samples")
    prefix = args.sample_file[0:-len(".samples")]

    samples = tsinfer.load(args.sample_file)
    ancestors = tsinfer.load(args.ancestors_file)
    chr_map = get_genetic_map(args.sample_file, args.genetic_map)
    rho, ma_mis, ms_mis, precision = get_rho(
            samples, ancestors, chr_map, args.num_threads, args.mismatch_ancestors,
            args.mismatch_samples, args.precision)

    if args.mismatch_ancestors is not None:
        inferred_anc_ts = match_ancestors(samples, ancestors, rho, ma_mis, precision,
                                          prefix, args.ancient_ancestors,
                                          args.num_threads)
    if args.mismatch_samples is not None:
        if args.ancestors_ts is not None:
            inferred_anc_ts = tskit.load(args.ancestors_ts)
        else:
            raise ValueError(
                    "Need to either run match_ancestors or provide an ancestors_ts")

        match_samples(
                samples, inferred_anc_ts, prefix, rho, ms_mis,
                args.modern_samples_match, args.ancient_ancestors, args.num_threads)

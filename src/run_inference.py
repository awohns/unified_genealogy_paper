import os.path
import argparse
import collections
import re
import time
import daiquiri
import logging

import tskit
import msprime
import numpy as np
import tsinfer
import stdpopsim

# Uncomment to debug output
# daiquiri.setup(level=logging.DEBUG)
Params = collections.namedtuple(
    "Params",
    "sample_data, filename, genetic_map, ma_mut_rate, ms_mut_rate, precision,"
    "num_threads",
)

Results = collections.namedtuple(
    "Results",
    "ma_mut, ms_mut, precision, edges, muts, num_trees, "
    "process_time, ga_process_time, ma_process_time, ms_process_time, ts_size, ts_path",
)


def run(params):
    """
    Run a single inference, with the specified rates
    """

    prefix = None
    if params.sample_data.path is not None:
        assert params.sample_data.path.endswith(".samples")
        prefix = params.sample_data.path[0 : -len(".samples")]
    start_time = time.process_time()
    ga_start_time = time.process_time()
    # Check if we have created the ancestors file already, if not generate it
    if not os.path.isfile(prefix + ".ancestors"):
        anc = tsinfer.generate_ancestors(
            params.sample_data,
            num_threads=params.num_threads,
            path=prefix + ".ancestors",
            progress_monitor=True,
        )
        path_compression = True
        if np.any(params.sample_data.individuals_time[:] != 0):
            anc_w_proxy = anc.insert_proxy_samples(
                params.sample_data, allow_mutation=True
            )
            anc = anc_w_proxy.copy(path=prefix + ".proxy.ancestors")
            anc.finalise()
            # Don't use path compression if ancient ancestors (can be fixed in the future)
            path_compression = False
        print(f"GA done (ma_mut: {params.ma_mut_rate}, ms_mut: {params.ms_mut_rate})")
    else:
        # Check to see if proxy ancestors were added, if not, load ancestordata file
        if os.path.isfile(prefix + ".proxy.ancestors"):
            anc = tsinfer.load(prefix + ".proxy.ancestors")
            path_compression = False
        else:
            anc = tsinfer.load(prefix + ".ancestors")
            path_compression = True
    ga_process_time = time.process_time() - ga_start_time

    r_prob, m_prob = get_rho(anc, params.filename)
    precision = params.precision
    print(
        f"Starting {params.ma_mut_rate} {params.ms_mut_rate}",
        f"Recombination probabiltity summary stats:",
        f"(mean {np.mean(r_prob):.4g} median {np.quantile(r_prob, 0.5):.4g}",
        f"max {np.max(r_prob):.4g} min {np.min(r_prob):.4g}",
        f"2.5% quantile {np.quantile(r_prob, 0.025):.4g})",
        f"precision {precision}",
    )
    ma_start_time = time.process_time()

    # If have not run match ancestors, run it now
    if not os.path.isfile(prefix + ".atrees"):
        inferred_anc_ts = tsinfer.match_ancestors(
            params.sample_data,
            anc,
            num_threads=params.num_threads,
            precision=precision,
            recombination=r_prob,
            mismatch=m_prob,
            path_compression=path_compression,
            progress_monitor=True,
        )
        inferred_anc_ts.dump(prefix + ".atrees")
        print(f"MA done (ma_mut:{params.ma_mut_rate} ms_mut{params.ms_mut_rate})")
    else:
        inferred_anc_ts = tskit.load(prefix + ".atrees")
    ma_process_time = time.process_time() - ma_start_time
    ms_start_time = time.process_time()
    if not os.path.isfile(prefix + ".trees"):
        inferred_ts = tsinfer.match_samples(
            params.sample_data,
            inferred_anc_ts,
            num_threads=params.num_threads,
            recombination=r_prob,
            mismatch=m_prob,
            precision=precision,
            progress_monitor=True,
            force_sample_times=True,
            simplify=False,
        )
        print(f"MS done: ms_mut rate = {params.ms_mut_rate})")
        process_time = time.process_time() - start_time
        ms_process_time = time.process_time() - ms_start_time
        ts_path = prefix + ".nosimplify.trees"
        inferred_ts.dump(ts_path)
    else:
        raise ValueError("Inferred tree sequence already present")

    return Results(
        ma_mut=params.ma_mut_rate,
        ms_mut=params.ms_mut_rate,
        precision=precision,
        edges=inferred_ts.num_edges,
        muts=inferred_ts.num_mutations,
        num_trees=inferred_ts.num_trees,
        process_time=process_time,
        ga_process_time=ga_process_time,
        ma_process_time=ma_process_time,
        ms_process_time=ms_process_time,
        ts_size=os.path.getsize(ts_path),
        ts_path=ts_path,
    )


def setup_sample_file(args):
    """
    Return a Thousand Genomes Project sample data file, the
    corresponding recombination rate array, a prefix to use for files, and None
    """
    filename = args.sample_file
    if not filename.endswith(".samples"):
        raise ValueError("Sample data file must end with '.samples'")
    sd = tsinfer.load(filename)
    return (
        sd,
        filename[: -len(".samples")],
    )


def get_rho(ancestors, filename):
    inference_pos = ancestors.sites_position[:]

    match = re.search(r"(chr\d+)", filename)
    if match is None:
        raise ValueError("chr must be in filename")
    chr = match.group(1)
    map = params.genetic_map
    if match or map is not None:
        if map is not None:
            print(f"Using {chr} from GRCh38 for the recombination map")
            rmap = msprime.RateMap.read_hapmap(
                map + chr + ".txt", sequence_length=ancestors.sequence_length
            )
        else:
            print(f"Using {chr} from HapMapII_GRCh37 for the recombination map")
            map = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
            if not map.is_cached():
                map.download()
            map_file = os.path.join(
                map.map_cache_dir, map.file_pattern.format(id="chr20")
            )
            rmap = msprime.RateMap.read_hapmap(
                map_file, sequence_length=ancestors.sequence_length
            )
        genetic_dists = tsinfer.Matcher.recombination_rate_to_dist(rmap, inference_pos)
        recombination = tsinfer.Matcher.recombination_dist_to_prob(genetic_dists)
        # Set 0 probability recombination positions to small value
        recombination[recombination == 0] = 1e-20
        # Hardcoded mismatch ratio to 1 since that's all we use in paper
        mismatch_ratio = 1
        num_alleles = 2
        mismatch = np.full(
            len(inference_pos),
            tsinfer.Matcher.mismatch_ratio_to_prob(mismatch_ratio, np.median(genetic_dists), num_alleles),
        )
    else:
        raise ValueError("must provide a recombination map")

    return recombination, mismatch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sample_file",
        default=None,
        help="A tsinfer sample file ending in '.samples'. If no genetic map is provided"
        " via the -m switch, and the filename contains chrNN"
        " where 'NN' is a number, assume this is a human samples file and use the"
        " appropriate recombination map from the thousand genomes project, otherwise"
        " use the physical distance between sites.",
    )
    # The _mrate parameter defaults set from analysis ot 1000G, see
    # https://github.com/tskit-dev/tsinfer/issues/263#issuecomment-639060101
    parser.add_argument(
        "-A",
        "--match_ancestors_mrate",
        type=float,
        default=1,
        help="The recurrent mutation probability in the match ancestors phase,"
        " as a fraction of the median recombination probability between sites",
    )
    parser.add_argument(
        "-S",
        "--match_samples_mrate",
        type=float,
        default=1,
        help="The recurrent mutation probability in the match samples phase,"
        " as a fraction of the median recombination probability between sites",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=15,
        help="The precision, as a number of decimal places, which will affect the speed"
        " of the matching algorithm (higher precision = lower speed). If None,"
        " calculate the smallest of the recombination rates or mutation rates, and"
        " use the negative exponent of that number plus one. E.g. if the smallest"
        " recombination rate is 2.5e-6, use precision = 6+3 = 7",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=0,
        help="The number of threads to use in inference",
    )
    parser.add_argument(
        "-m",
        "--genetic_map",
        default=None,
        help="An alternative genetic map to be used for this analysis, in the format"
        "expected by msprime.RateMap.read_hapmap",
    )
        
    args = parser.parse_args()
    # We only use mismatch ratio of 1 and precision of 15 in the paper
    assert args.match_ancestors_mrate == args.match_samples_mrate == 1
    assert args.precision == 15

    (
        samples,
        prefix,
    ) = setup_sample_file(args)

    params = Params(
        samples,
        args.sample_file,
        args.genetic_map,
        args.match_ancestors_mrate,
        args.match_samples_mrate,
        args.precision,
        args.num_threads,
    )
    print(f"Running inference with {params}")
    with open(prefix + ".results", "wt") as file:
        result = run(params)
        print("\t".join(str(r) for r in result), file=file, flush=True)

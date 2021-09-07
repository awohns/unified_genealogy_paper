"""
Simple CLI to run tsinf on the command line.
"""
import argparse
import os
import re

import numpy as np

import tskit
import tsinfer
import msprime
import stdpopsim


def generate_ancestors(samples_fn, num_threads, prefix):
    sample_data = tsinfer.load(samples_fn)
    anc = tsinfer.generate_ancestors(
        sample_data,
        num_threads=num_threads,
        path=prefix + ".ancestors",
        progress_monitor=True,
    )
    if np.any(sample_data.individuals_time[:] != 0):
        anc_w_proxy = anc.insert_proxy_samples(sample_data, allow_mutation=True)
        anc = anc_w_proxy.copy(path=prefix + ".proxy.ancestors")
        anc.finalise()
    maximum_time = np.max(anc.ancestors_time[:])
    if (
        maximum_time < 3
    ):  # hacky way of checking if we used frequency to order ancestors
        anc = anc.truncate_ancestors(
            0.4, 0.6, length_multiplier=1, path=prefix + ".truncated.ancestors"
        )
    else:
        upper_time_limit = maximum_time * 0.6
        lower_time_limit = maximum_time * 0.4
        anc = anc.truncate_ancestors(
            lower_time_limit,
            upper_time_limit,
            length_multiplier=1,
            path=prefix + ".truncated.ancestors",
        )
    return anc


def match_ancestors(samples_fn, anc, num_threads, precision, r_prob, m_prob, prefix):
    sample_data = tsinfer.load(samples_fn)
    inferred_anc_ts = tsinfer.match_ancestors(
        sample_data,
        anc,
        num_threads=num_threads,
        precision=precision,
        recombination=r_prob,
        mismatch=m_prob,
        progress_monitor=True,
    )
    inferred_anc_ts.dump(prefix + ".atrees")
    return inferred_anc_ts


def match_samples(
    samples_fn, inferred_anc_ts, num_threads, r_prob, m_prob, precision, prefix
):
    sample_data = tsinfer.load(samples_fn)
    inferred_ts = tsinfer.match_samples(
        sample_data,
        inferred_anc_ts,
        num_threads=num_threads,
        recombination=r_prob,
        mismatch=m_prob,
        precision=precision,
        progress_monitor=True,
        force_sample_times=True,
        simplify=False,
    )
    ts_path = prefix + ".nosimplify.trees"
    inferred_ts.dump(ts_path)
    return inferred_ts


def get_rho(ancestors, genetic_map, filename):
    inference_pos = ancestors.sites_position[:]

    match = re.search(r"(chr\d+)", filename)
    arm = "_q" in filename
    sequence_length = None
    if arm:
        sequence_length = ancestors.sequence_length
    if match is None:
        raise ValueError("chr must be in filename")
    chr = match.group(1)
    map = genetic_map
    if match or map is not None:
        if map is not None:
            print(f"Using {chr} from GRCh38 for the recombination map")
            rmap = msprime.RateMap.read_hapmap(
                map + chr + ".txt", sequence_length=sequence_length
            )
        else:
            print(f"Using {chr} from HapMapII_GRCh37 for the recombination map")
            map = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
            if not map.is_cached():
                map.download()
            map_file = os.path.join(map.map_cache_dir, map.file_pattern.format(id="20"))
            rmap = msprime.RateMap.read_hapmap(
                map_file, sequence_length=sequence_length
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
            tsinfer.Matcher.mismatch_ratio_to_prob(
                mismatch_ratio, np.median(genetic_dists), num_alleles
            ),
        )
    else:
        raise ValueError("must provide a recombination map")
    print(
        f"Recombination probabiltity summary stats:",
        f"(mean {np.mean(recombination):.4g} median {np.quantile(recombination, 0.5):.4g}",
        f"max {np.max(recombination):.4g} min {np.min(recombination):.4g}",
        f"2.5% quantile {np.quantile(recombination, 0.025):.4g})",
    )

    return recombination, mismatch


def main():

    description = """Simple CLI wrapper for tsinfer
        tskit version: {}
        tsinfer version: {}""".format(
        tskit.__version__, tsinfer.__version__
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verbosity", "-v", action="count", default=0)
    parser.add_argument(
        "samples",
        help="The samples file name, as saved by tsinfer.SampleData.initialise()",
    )
    parser.add_argument("prefix", help="The prefix of the output filename")
    parser.add_argument(
        "-t",
        "--threads",
        default=1,
        type=int,
        help="The number of worker threads to use",
    )
    parser.add_argument(
        "-s",
        "--step",
        default="infer",
        choices=["GA", "MA", "MS"],
        help="Which step of the algorithm to run: generate ancestors (GA), match ancestors"
        "(MA), or match samples (MS) or all three (infer)",
    )
    parser.add_argument(
        "-m",
        "--genetic-map",
        default=None,
        help="An alternative genetic map to be used for this analysis, in the format"
        "expected by msprime.RateMap.read_hapmap",
    )
    parser.add_argument(
        "-p",
        "--precision",
        default=None,
        type=int,
        help="The precision parameter to pass to the function",
    )

    parser.add_argument("-V", "--version", action="version", version=description)

    args = parser.parse_args()

    if not os.path.isfile(args.samples):
        raise ValueError("No samples file")

    if args.step == "infer":
        anc = generate_ancestors(args.samples, args.threads, args.prefix)
        if args.genetic_map == "None":
            genetic_map = None
        r_prob, m_prob = get_rho(anc, genetic_map, args.prefix)
        inferred_anc_ts = match_ancestors(
            args.samples, anc, args.threads, args.precision, r_prob, m_prob
        )
        match_samples(
            args.samples, inferred_anc_ts, args.threads, r_prob, m_prob, args.precision
        )
    if args.step == "GA":
        anc = generate_ancestors(args.samples, args.threads, args.prefix)
    if args.step == "MA":
        anc = tsinfer.load(args.prefix + ".truncated.ancestors")
        if args.genetic_map == "None":
            genetic_map = None
        else:
            genetic_map = args.genetic_map
        r_prob, m_prob = get_rho(anc, genetic_map, args.prefix)
        inferred_anc_ts = match_ancestors(
            args.samples, anc, args.threads, args.precision, r_prob, m_prob, args.prefix
        )
    if args.step == "MS":
        anc = tsinfer.load(args.prefix + ".truncated.ancestors")
        inferred_anc_ts = tskit.load(args.prefix + ".atrees")
        if args.genetic_map == "None":
            genetic_map = None
        else:
            genetic_map = args.genetic_map
        r_prob, m_prob = get_rho(anc, genetic_map, args.prefix)
        match_samples(
            args.samples,
            inferred_anc_ts,
            args.threads,
            r_prob,
            m_prob,
            args.precision,
            args.prefix,
        )


if __name__ == "__main__":
    main()

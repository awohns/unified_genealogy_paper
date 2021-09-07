import os.path
import argparse
import collections
import time
import subprocess
import sys

import tskit
import tsinfer

tsinfer_executable = os.path.join("../", "src", "run_tsinfer.py")

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
    "process_time, ga_cpu_time, ga_memory_use, ma_cpu_time, ma_memory_use, ms_cpu_time, ms_memory_use, ts_size, ts_path",
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
    # Check we have not created any files already
    if os.path.isfile(prefix + ".ancestors"):
        raise ValueError(".ancestors file already exists")
    if os.path.isfile(prefix + ".atrees"):
        raise ValueError(".atrees file already exists")
    if os.path.isfile(prefix + ".trees"):
        raise ValueError(".trees file already exists")

    cmd = [
        sys.executable,
        tsinfer_executable,
        params.sample_data.path,
        "--step",
        "GA",
    ]
    cmd += ["--threads", str(params.num_threads), prefix]
    ga_cpu_time, ga_memory_use = time_cmd(cmd, prefix + "_GA")
    print(f"GA done (ma_mut: {params.ma_mut_rate}, ms_mut: {params.ms_mut_rate})")

    precision = params.precision
    print(
        f"Starting {params.ma_mut_rate} {params.ms_mut_rate}",
        f"precision {precision}",
    )

    cmd = [
        sys.executable,
        tsinfer_executable,
        params.sample_data.path,
        "--step",
        "MA",
    ]
    cmd += [
        "--precision",
        str(precision),
        "--threads",
        str(params.num_threads),
        prefix,
        "--genetic-map",
        params.genetic_map,
    ]
    ma_cpu_time, ma_memory_use = time_cmd(cmd, prefix + "_MA")
    print(f"MA done (ma_mut:{params.ma_mut_rate} ms_mut{params.ms_mut_rate})")

    cmd = [
        sys.executable,
        tsinfer_executable,
        params.sample_data.path,
        "--step",
        "MS",
    ]
    cmd += [
        "--precision",
        str(precision),
        "--threads",
        str(params.num_threads),
        prefix,
        "--genetic-map",
        params.genetic_map,
    ]
    ms_cpu_time, ms_memory_use = time_cmd(cmd, prefix + "_MS")
    inferred_ts = tskit.load(prefix + ".nosimplify.trees")
    print(f"MS done: ms_mut rate = {params.ms_mut_rate})")
    process_time = time.process_time() - start_time
    ts_path = prefix + ".nosimplify.trees"

    return Results(
        ma_mut=params.ma_mut_rate,
        ms_mut=params.ms_mut_rate,
        precision=precision,
        edges=inferred_ts.num_edges,
        muts=inferred_ts.num_mutations,
        num_trees=inferred_ts.num_trees,
        process_time=process_time,
        ga_cpu_time=ga_cpu_time,
        ga_memory_use=ga_memory_use,
        ma_cpu_time=ma_cpu_time,
        ma_memory_use=ma_memory_use,
        ms_cpu_time=ms_cpu_time,
        ms_memory_use=ms_memory_use,
        ts_size=os.path.getsize(ts_path),
        ts_path=ts_path,
    )


def time_cmd(cmd, output):
    """
    Runs the specified command line (a list suitable for subprocess.call)
    and writes the stdout to the specified file object.
    """
    output_file = output + ".tsinfer.time.txt"
    time_cmd = "/usr/bin/time"
    full_cmd = [time_cmd, "-o", output_file, "-f%M %S %U"] + cmd
    exit_status = subprocess.call(full_cmd, stderr=sys.stderr, stdout=sys.stdout)
    f = open(output_file)
    split = f.readlines()[-1].split()
    # From the time man page:
    # M: Maximum resident set size of the process during its lifetime,
    #    in Kilobytes.
    # S: Total number of CPU-seconds used by the system on behalf of
    #    the process (in kernel mode), in seconds.
    # U: Total number of CPU-seconds that the process used directly
    #    (in user mode), in seconds.
    max_memory = int(split[0]) * 1024
    system_time = float(split[1])
    user_time = float(split[2])
    return user_time + system_time, max_memory


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
        default="None",
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

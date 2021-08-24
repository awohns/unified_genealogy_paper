"""
Dated sample data files with missing data create ancestors at many different time points,
often only one ancestor in each time point, which can cause difficulties parallelising
the inference. This script takes a sampledata file (usually containing missing data),
and bins the resulting times to the nearest 10 (unless the time is <= 1).
"""

import argparse

import numpy as np
import tsinfer
import tskit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="A tsinfer sample file ending in '.samples")
    parser.add_argument("output_file", help="A tsinfer sample file ending in '.samples")
    args = parser.parse_args()

    sd = tsinfer.load(args.input_file).copy(path=args.output_file)

    times = sd.sites_time[:]
    times[times > 1] = np.round(times[times > 1], -1)
    times[times == 0] = 1
    sd.sites_time[:] = times
    print(
        "Number of samples:",
        sd.num_samples,
        ". Number of discrete times:",
        len(np.unique(sd.sites_time[:])),
    )
    sd.finalise()

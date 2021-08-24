"""
Sample data files with missing data create ancestors at many different time points,
often only one ancestor in each time point, which can cause difficulties parallelising
the inference. This script takes a sampledata file (usually containing missing data),
calculates the times-as-freq values, then bins them into frequency bands.
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

    for j, variant in enumerate(sd.variants()):
        time = variant.site.time
        if tskit.is_unknown_time(time):
            counts = tsinfer.allele_counts(variant.genotypes)
            # Non-variable sites have no obvious freq-as-time values
            assert counts.known != counts.derived
            assert counts.known != counts.ancestral
            assert counts.known > 0
            # Time = freq of *all* derived alleles. Note that if n_alleles > 2 this
            # may not be sensible: https://github.com/tskit-dev/tsinfer/issues/228
            times[variant.site.id] = counts.derived / counts.known

    sd.sites_time[:] = np.around(times * sd.num_samples) / sd.num_samples
    print(
        "Number of samples:",
        sd.num_samples,
        ". Number of discrete times:",
        len(np.unique(sd.sites_time[:])),
    )
    sd.finalise()

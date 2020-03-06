"""
Simple CLI to run tsdate on the command line.
"""
import os
import argparse
import logging

import tsdate
import tskit


def main():
    description = """Simple CLI wrapper for tsdate
        tskit version: {}
        tsdate version: {}""".format(tskit.__version__, tsdate.__version__)
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verbosity', '-v', action='count', default=0)
    parser.add_argument(
        "input",
        help="The input tree sequence file name")
    parser.add_argument(
        "output",
        help="The path to write the output tree sequence file to")
    parser.add_argument(
        'Ne', type=float,
        help="Effective population size")
    parser.add_argument(
        "-t", "--timepoints", default=20, type=int,
        help="The number of timepoints to use")
    parser.add_argument(
        "--mutation-rate", default=1e-8, type=float,
        help="Mutation rate")
    parser.add_argument(
        "-m", "--method", default="inside_outside",
        choices=['inside_outside', 'maximization'],
        help="Which implementation to use, inside-outside (more accurate) \
        or maximization (more stable)")
    parser.add_argument(
        "-V", "--version", action='version', version=description)

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise ValueError("No input tree sequence file")
    input_ts = tskit.load(args.input)
    prior = tsdate.build_prior_grid(input_ts, timepoints=args.timepoints, 
                                    approximate_prior=True)
    ts = tsdate.date(
        input_ts, args.Ne, method=args.method, prior=prior)
    ts.dump(args.output)


if __name__ == "__main__":
    main()

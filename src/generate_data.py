#!/usr/bin/env python3
"""
Generate data for plots in tsdate paper
 python3 src/generate_data.py PLOT_NAME
"""
import argparse

import os
import pandas as pd
import numpy as np
import random
import tempfile
from tqdm import tqdm

import tskit

import evaluation


class DataGeneration:
    """
    Superclass for data generation classes for each figure.
    """

    # Default settings
    default_replicates = 10
    default_seed = 123
    data_dir = "data/"

    # Each summary has a unique name. This is used as the identifier for the csv file.
    name = None

    def __init__(self):
        self.data_file = os.path.abspath(
            os.path.join(self.data_dir, self.name + ".csv"))
        self.rng = random.Random(self.default_seed)
        self.sim_cols = ["filename", "replicate", "sample_size", "Ne", "length",
                         "recombination_rate", "mutation_rate", "n_edges",
                         "n_trees", "n_sites", "seed"]

    def summarize(self):
        """
        Take the output of the inference and save to CSV
        """
        self.data.to_csv("data/" + self.name + ".csv")


class CpuScalingSampleSize(DataGeneration):
    """
    Plot CPU times of tsdate, tsinfer, tsdate+tsinfer, Relate, and GEVA
    """
    name = "cpu_scaling_samplesize"

    def __init__(self):
        DataGeneration.__init__(self)
        # self.sample_sizes = [10, 20, 40, 64, 100, 250, 500]
        self.sample_sizes = [10, 20, 40, 64, 100]
        self.tools = ["tsdate", "tsinfer", "relate", "geva"]
        self.sim_cols = self.sim_cols + ["tsdate_cputime", "tsdate_memory",
                                         "tsinfer_cputime", "tsinfer_memory",
                                         "relate_cputime", "relate_memory",
                                         "geva_cputime", "geva_memory"]
        self.num_rows = len(self.sample_sizes) * self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)

    def setup(self):
        """
        Run Simulations
        """

        for sample_size in tqdm(self.sample_sizes, desc="Running Simulations"):
            seeds = [self.rng.randint(1, 2**31) for i in range(self.default_replicates)]
            for index, seed in enumerate(seeds):
                sim = evaluation.run_neutral_sim(sample_size, mutation_rate=1e-8,
                                                 recombination_rate=1e-8, Ne=10000,
                                                 length=1e6, seed=seed)
                # Dump simulated tree
                filename = self.name + "_" + str(sample_size) + "_" + str(index)
                # Save the simulated tree sequence
                sim.dump(os.path.join(self.data_dir, filename + ".trees"))

                # Create sampledata file
                samples = evaluation.generate_samples(sim, os.path.join(self.data_dir,
                                                      filename))

                # Create VCF file
                _ = evaluation.sampledata_to_vcf(samples,
                                                 os.path.join(self.data_dir,
                                                              filename))

                # Update dataframe with details of simulation
                row_data = {"filename": filename, "replicate": index,
                            "sample_size": sample_size, "Ne": 10000, "length": 2e5,
                            "mutation_rate": 1e-8, "recombination_rate": 1e-8,
                            "n_edges": sim.num_edges, "n_trees": sim.num_trees,
                            "n_sites": sim.num_sites, "seed": seed,
                            "tsdate_cputime": None, "tsdate_memory": None,
                            "tsinfer_cputime": None, "tsinfer_memory": None,
                            "relate_cputime": None, "relate_memory": None,
                            "geva_cputime": None, "geva_memory": None}
                self.data = self.data.append(row_data, ignore_index=True)

        # Save dataframe
        self.summarize()

    def inference(self):
        """
        Run four methods on the simulated data
        """
        self.data = pd.read_csv(self.data_file)
        for index, row in tqdm(self.data.iterrows(), desc="Running Inference"):
            path_to_file = os.path.join(self.data_dir, row["filename"])
            sim = tskit.load(path_to_file + ".trees")

            _, tsdate_cputime, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mutation_rate"], 20,
                "inside_outside")
            self.data.loc[index, ["tsdate_cputime", "tsdate_memory"]] = \
                tsdate_cputime, tsdate_memory

            _, tsinfer_cputime, tsinfer_memory = evaluation.run_tsinfer(
                path_to_file + ".samples", sim.get_sequence_length())
            self.data.loc[index, ["tsinfer_cputime", "tsinfer_memory"]] = \
                [tsinfer_cputime, tsinfer_memory]

            _, _, relate_cputime, relate_memory = evaluation.run_relate(
                sim, path_to_file, 1e-8, 10000, "relate_file")

            self.data.loc[index, ["relate_cputime", "relate_memory"]] = \
                [relate_cputime, relate_memory]
            _, geva_cputime, geva_memory = evaluation.run_geva(
                path_to_file, 10000, 1e-8, 1e-8)
            self.data.loc[index, ["geva_cputime", "geva_memory"]] = \
                [geva_cputime, geva_memory]
            self.summarize()
        # Save dataframe
        self.summarize()


class NeutralSimulatedMutationAccuracy(DataGeneration):
    name = "neutral_simulated_mutation_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.tools = ["tsdate", "tsinfer", "relate", "geva"]
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)

    def setup(self):
        """
        Run Simulations
        """

        seeds = [self.rng.randint(1, 2**31) for i in range(self.default_replicates)]
        for index, seed in tqdm(enumerate(seeds), desc="Running Simulations"):
            sim = evaluation.run_neutral_sim(sample_size=250, mutation_rate=1e-8,
                                             recombination_rate=1e-8, Ne=10000,
                                             length=1e6, seed=seed)
            # Dump simulated tree
            filename = self.name + "_" + str(index)
            # Save the simulated tree sequence
            sim.dump(os.path.join(self.data_dir, filename + ".trees"))

            # Create sampledata file
            samples = evaluation.generate_samples(sim, os.path.join(self.data_dir,
                                                  filename))

            # Create VCF file
            _ = evaluation.sampledata_to_vcf(samples,
                                             os.path.join(self.data_dir,
                                                          filename))

            # Update dataframe with details of simulation
            row_data = {"filename": filename, "replicate": index,
                        "sample_size": 250, "Ne": 10000, "length": 2e5,
                        "mutation_rate": 1e-8, "recombination_rate": 1e-8,
                        "n_edges": sim.num_edges, "n_trees": sim.num_trees,
                        "n_sites": sim.num_sites, "seed": seed}
            self.data = self.data.append(row_data, ignore_index=True)

        # Save dataframe
        self.summarize()

    def inference(self):
        """
        Run four methods on the simulated data
        """
        self.data = pd.read_csv(self.data_file)
        for index, row in tqdm(self.data.iterrows(), desc="Running Inference"):
            path_to_file = os.path.join(self.data_dir, row["filename"])
            sim = tskit.load(path_to_file + ".trees")

            dated_ts, tsdate_cputime, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mutation_rate"], 20,
                "inside_outside")
            dated_ts.dump(path_to_file + "tsdated.trees")

            inferred_ts, tsinfer_cputime, tsinfer_memory = evaluation.run_tsinfer(
                path_to_file + ".samples", sim.get_sequence_length())
            inferred_ts.simplify().dump(path_to_file + "tsinferred.trees")

            dated_inferred_ts, tsdate_cputime, tsdate_memory = evaluation.run_tsdate(
                path_to_file + "tsinferred.trees", row["Ne"], row["mutation_rate"], 50,
                "inside_outside")
            dated_inferred_ts.dump(path_to_file + "tsinferred.tsdated.trees")

            relate_ts, relate_age, relate_cputime, relate_memory = evaluation.run_relate(
                sim, path_to_file, 1e-8, row["Ne"] * 2, "relate_file")
            relate_ts.dump(path_to_file + "relate.trees")

            geva_ages, geva_cputime, geva_memory = evaluation.run_geva(
                path_to_file, 10000, 1e-8, 1e-8)
            geva_ages.to_csv(path_to_file + "geva.csv")
            geva_positions = pd.read_csv(path_to_file + ".marker.txt", delimiter=" ",
                                         index_col="MarkerID")

            compare_df = evaluation.compare_mutations(
                ["simulated_ts", "tsdate", "tsdate_inferred", "geva", "relate"],
                [sim, dated_ts, dated_inferred_ts],
                geva_ages=geva_ages, geva_positions=geva_positions,
                relate_ages=relate_age)
            compare_df.to_csv(path_to_file + "_mutaitons.csv")
            self.summarize()
        # Save dataframe
        self.summarize()


# class HumanlikeSimulatedMutationAccuracy(DataGeneration):
#     name = "humanlike_simulated_mutation_accuracy"
#     default_replicates = 20

#     def setup():
#         """
#         Run Simulations
#         """

#     def inference():
#         """
#         Run four methods on the simulated data
#         """


def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


def main():
    figures = get_subclasses(DataGeneration)
    figures = list(get_subclasses(DataGeneration))
    name_map = {fig.name: fig for fig in figures if fig.name is not None}

    parser = argparse.ArgumentParser(description="Generate the data for a figure.")
    parser.add_argument(
        "name", type=str, help="figure name", default='all',
        choices=sorted(list(name_map.keys()) + ['all']))
    parser.add_argument(
        "--setup", action="store_true", default=False, help="Run simulations")
    parser.add_argument(
        "--inference", action="store_true", default=False, help="Run inference")
    parser.add_argument(
        "--summarize", action="store_true", default=False, help="Summarize output")


    args = parser.parse_args()
    if args.name == 'all':
        for name, fig in name_map.items():
            if fig in figures:
                if args.setup:
                    fig.setup()
                if args.inference:
                    fig.inference()
                if args.summarize:
                    fig.summarize()
    else:
        fig = name_map[args.name]()
        if args.setup:
            fig.setup()
        if args.inference:
            fig.inference()
        if args.summarize:
            fig.summarize()


if __name__ == "__main__":
    main()

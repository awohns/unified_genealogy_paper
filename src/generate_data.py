#!/usr/bin/env python3
"""
Generate data for plots in tsdate paper
 python3 src/generate_data.py PLOT_NAME
"""
import argparse

import logging
import os
import pandas as pd
import numpy as np
import random
import subprocess
import tempfile
from tqdm import tqdm

import tskit
import tsinfer

import evaluation


class DataGeneration:
    """
    Superclass for data generation classes for each figure.
    """

    # Default settings
    default_replicates = 10
    default_seed = 123
    data_dir = os.path.join(os.getcwd(), "data/")

    # Each summary has a unique name. This is used as the identifier for the csv file.
    name = None

    def __init__(self):
        self.data_file = os.path.abspath(
            os.path.join(self.data_dir, self.name + ".csv"))
        self.rng = random.Random(self.default_seed)
        self.sim_cols = ["filename", "replicate", "sample_size", "Ne", "length",
                         "rec_rate", "mut_rate", "n_edges",
                         "n_trees", "n_sites", "seed"]

    def setup(self, parameter, parameter_arr, simulate_fn, row_data):
        """
        Run Simulations
        """

        for param in tqdm(parameter_arr, desc="Running Simulations"):
            seeds = [self.rng.randint(1, 2**31) for i in range(self.default_replicates)]
            for index, seed in enumerate(seeds):
                sim = simulate_fn((param, seed))
                # Dump simulated tree
                filename = self.name + "_" + str(parameter) + "_" + str(index)
                row_data["filename"] = filename
                row_data["replicate"] = index
                row_data["n_edges"] = sim.num_edges
                row_data["n_trees"] = sim.num_trees
                row_data["n_sites"] = sim.num_sites
                row_data["seed"] = seed

                if param:
                    row_data[parameter] = param
                # Save the simulated tree sequence
                sim.dump(os.path.join(self.data_dir, filename + ".trees"))

                # Create sampledata file
                _ = tsinfer.formats.SampleData.from_tree_sequence(
                    sim, path=os.path.join(self.data_dir, filename + ".samples"),
                    use_times=False)

                # Create VCF file
                with open(os.path.join(self.data_dir,
                                       filename + ".vcf"), "w") as vcf_file:
                    sim.write_vcf(vcf_file, ploidy=2, position_transform='legacy')

                # Update dataframe with details of simulation
                self.data = self.data.append(row_data, ignore_index=True)

        # Save dataframe
        self.summarize()

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
        self.sample_sizes = [10, 20, 40, 64, 100, 250, 500, 1000, 1500, 2000]
        # self.sample_sizes = [10, 20, 40, 64, 100]
        self.sim_cols = self.sim_cols + ["filename", "replicate", "sample_size", "Ne",
                                         "length", "mut_rate", "rec_rate", "n_edges",
                                         "n_trees", "n_sites", "seed", "tsdate_cputime",
                                         "tsdate_memory", "tsinfer_cputime",
                                         "tsinfer_memory", "tsdate_infer_cputime",
                                         "tsdate_infer_memory", "relate_cputime",
                                         "relate_memory", "geva_cputime", "geva_memory"]
        self.tools = ["tsdate", "tsinfer", "relate", "geva"]
        self.num_rows = len(self.sample_sizes) * self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["Ne"] = 10000
        row_data["length"] = 1e6
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_fn(params):
            sample_size = params[0]
            seed = params[1]
            return(
                evaluation.run_neutral_sim(sample_size=sample_size,
                                           mutation_rate=row_data["mut_rate"],
                                           recombination_rate=row_data["rec_rate"],
                                           Ne=row_data["Ne"],
                                           length=row_data["length"], seed=seed))
        DataGeneration.setup(self, "sample_size", self.sample_sizes, simulate_fn,
                             row_data)

    def inference(self):
        """
        Run four methods on the simulated data
        """
        try:
            self.data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            logging.error("Must run with --setup flag first")
        for index, row in tqdm(self.data.iterrows(), desc="Running Inference"):
            path_to_file = os.path.join(self.data_dir, row["filename"])
            sim = tskit.load(path_to_file + ".trees")

            _, tsdate_cputime, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mut_rate"], 20,
                "inside_outside")
            self.data.loc[index, ["tsdate_cputime", "tsdate_memory"]] = \
                tsdate_cputime, tsdate_memory

            _, tsinfer_cputime, tsinfer_memory = evaluation.run_tsinfer(
                path_to_file + ".samples", sim.get_sequence_length())
            self.data.loc[index, ["tsinfer_cputime", "tsinfer_memory"]] = \
                [tsinfer_cputime, tsinfer_memory]

            _, dated_infer_cputime, dated_infer_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mut_rate"], 20,
                "inside_outside")
            self.data.loc[index, ["tsdate_infer_cputime", "tsdate_infer_memory"]] = \
                dated_infer_cputime, dated_infer_memory

            relate_dir = os.path.join(self.data_dir, "relate_" + row["filename"])
            _, _, relate_cputime, relate_memory = evaluation.run_relate(
                sim, path_to_file, row["mut_rate"], row["Ne"] * 2, relate_dir, "relate_file")

            self.data.loc[index, ["relate_cputime", "relate_memory"]] = \
                [relate_cputime, relate_memory]
            _, geva_cputime, geva_memory = evaluation.run_geva(
                path_to_file, row["Ne"], row["mut_rate"], row["rec_rate"])
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
            row_data = dict.fromkeys(self.sim_cols)
            row_data["sample_size"] = 250
            row_data["Ne"] = 10000
            row_data["length"] = 5e6
            row_data["mut_rate"] = 1e-8
            row_data["rec_rate"] = 1e-8

            def simulate_fn(params):
                seed = params[1]
                return(
                    evaluation.run_neutral_sim(sample_size=row_data["sample_size"],
                                               mutation_rate=row_data["mut_rate"],
                                               recombination_rate=row_data["rec_rate"],
                                               Ne=row_data["Ne"],
                                               length=row_data["length"], seed=seed))
            DataGeneration.setup(self, None, [None], simulate_fn, row_data)

            # Randomly sample mutations from each sampledata file
            self.data = pd.read_csv(self.data_file)
            for index, row in self.data.iterrows():
                samples = tsinfer.load(os.path.join(self.data_dir,
                                                    row["filename"] + ".samples"))
                mut_pos = pd.DataFrame(columns=['CHR', 'POS'])
                mut_pos['POS'] = sorted(
                    np.round(self.rng.sample(list(samples.sites_position[:]),
                                             1000)).astype(int))
                mut_pos['CHR'] = 1
                mut_pos.to_csv(
                    os.path.join(self.data_dir,
                                 row["filename"] + '.subsampled.pos.csv'),
                    index=False, header=False, sep=" ")
                subprocess.check_output(
                    ['vcftools', '--vcf', os.path.join(
                        self.data_dir, row["filename"] + '.vcf'), '--positions',
                     os.path.join(self.data_dir, row["filename"] + '.subsampled.pos.csv'),'--recode', '--recode-INFO-all', '--out', os.path.join(self.data_dir, row["filename"] + '.subsampled')]) 

    def inference(self):
        """
        Run four methods on the simulated data
        """
        try:
            self.data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            logging.error("Must run with --setup flag first")
        for index, row in tqdm(self.data.iterrows(), desc="Running Inference"):
            path_to_file = os.path.join(self.data_dir, row["filename"])
            sim = tskit.load(path_to_file + ".trees")

            dated_ts, tsdate_cputime, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mut_rate"], 20,
                "inside_outside")
            dated_ts.dump(path_to_file + ".tsdated.trees")

            inferred_ts, tsinfer_cputime, tsinfer_memory = evaluation.run_tsinfer(
                path_to_file + ".samples", sim.get_sequence_length())
            inferred_ts.simplify().dump(path_to_file + ".tsinferred.trees")

            dated_inferred_ts, tsdate_cputime, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".tsinferred.trees", row["Ne"], row["mut_rate"], 50,
                "inside_outside")
            dated_inferred_ts.dump(path_to_file + ".tsinferred.tsdated.trees")

            relate_dir = os.path.join(self.data_dir, "relate_" + row["filename"])
            relate_ts, relate_age, relate_cputime, relate_memory = evaluation.run_relate(
                sim, path_to_file, row["mut_rate"], row["Ne"] * 2, relate_dir, "relate_run")
            relate_age.to_csv(path_to_file + "relate_age")

            geva_ages, geva_cputime, geva_memory = evaluation.run_geva(
                path_to_file + ".subsampled.recode", row["Ne"], row["mut_rate"], row["rec_rate"])
            geva_ages.to_csv(path_to_file + ".geva.csv")
            geva_positions = pd.read_csv(path_to_file + ".subsampled.recode.marker.txt", delimiter=" ",
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


class HumanlikeSimulatedMutationAccuracy(DataGeneration):
    name = "neutral_simulated_mutation_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.tools = ["tsdate", "tsinfer", "relate", "geva"]
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)


    def setup(self):
            row_data = dict.fromkeys(self.sim_cols)
            row_data["sample_size"] = 250
            row_data["Ne"] = 10000
            row_data["length"] = 5e6
            row_data["mut_rate"] = 1e-8
            row_data["rec_rate"] = 1e-8

            def simulate_fn(params):
                seed = params[1]
                return(
                    evaluation.run_neutral_sim(sample_size=row_data["sample_size"],
                                               mutation_rate=row_data["mut_rate"],
                                               recombination_rate=row_data["rec_rate"],
                                               Ne=row_data["Ne"],
                                               length=row_data["length"], seed=seed))
            DataGeneration.setup(self, None, [None], simulate_fn, row_data)

            # Randomly sample mutations from each sampledata file
            self.data = pd.read_csv(self.data_file)
            for index, row in self.data.iterrows():
                samples = tsinfer.load(os.path.join(self.data_dir,
                                                    row["filename"] + ".samples"))
                mut_pos = pd.DataFrame(columns=['CHR', 'POS'])
                mut_pos['POS'] = sorted(
                    np.round(self.rng.sample(list(samples.sites_position[:]),
                                             1000)).astype(int))
                mut_pos['CHR'] = 1
                mut_pos.to_csv(
                    os.path.join(self.data_dir,
                                 row["filename"] + '.subsampled.pos.csv'),
                    index=False, header=False, sep=" ")
                subprocess.check_output(
                    ['vcftools', '--vcf', os.path.join(
                        self.data_dir, row["filename"] + '.vcf'), '--positions',
                     os.path.join(self.data_dir, row["filename"] + '.subsampled.pos.csv'),'--recode', '--recode-INFO-all', '--out', os.path.join(self.data_dir, row["filename"] + '.subsampled')]) 

    def inference(self):
        """
        Run four methods on the simulated data
        """
        try:
            self.data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            logging.error("Must run with --setup flag first")
        for index, row in tqdm(self.data.iterrows(), desc="Running Inference"):
            path_to_file = os.path.join(self.data_dir, row["filename"])
            sim = tskit.load(path_to_file + ".trees")

            dated_ts, tsdate_cputime, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mut_rate"], 20,
                "inside_outside")
            dated_ts.dump(path_to_file + ".tsdated.trees")

            inferred_ts, tsinfer_cputime, tsinfer_memory = evaluation.run_tsinfer(
                path_to_file + ".samples", sim.get_sequence_length())
            inferred_ts.simplify().dump(path_to_file + ".tsinferred.trees")

            dated_inferred_ts, tsdate_cputime, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".tsinferred.trees", row["Ne"], row["mut_rate"], 50,
                "inside_outside")
            dated_inferred_ts.dump(path_to_file + ".tsinferred.tsdated.trees")

            relate_dir = os.path.join(self.data_dir, "relate_" + row["filename"])
            relate_ts, relate_age, relate_cputime, relate_memory = evaluation.run_relate(
                sim, path_to_file, row["mut_rate"], row["Ne"] * 2, relate_dir, "relate_run")
            relate_age.to_csv(path_to_file + "relate_age")

            geva_ages, geva_cputime, geva_memory = evaluation.run_geva(
                path_to_file + ".subsampled.recode", row["Ne"], row["mut_rate"], row["rec_rate"])
            geva_ages.to_csv(path_to_file + ".geva.csv")
            geva_positions = pd.read_csv(path_to_file + ".subsampled.recode.marker.txt", delimiter=" ",
                                         index_col="MarkerID")

            compare_df = evaluation.compare_mutations(
                ["simulated_ts", "tsdate", "tsdate_inferred", "geva", "relate"],
                [sim, dated_ts, dated_inferred_ts],
                geva_ages=geva_ages, geva_positions=geva_positions,
                relate_ages=relate_age)
            compare_df.to_csv(path_to_file + "_mutations.csv")
            self.summarize()
        # Save dataframe
        self.summarize()


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


    args = parser.parse_args()
    if args.name == 'all':
        for name, fig in name_map.items():
            if fig in figures:
                if args.setup:
                    fig().setup()
                if args.inference:
                    fig().inference()
    else:
        fig = name_map[args.name]()
        if args.setup:
            fig.setup()
        if args.inference:
            fig.inference()


if __name__ == "__main__":
    main()

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
import shutil
import stdpopsim
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
                filename = self.name + "_" + str(param) + "_" + str(index)
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

    def make_uniform_genetic_map(self, row_data, param=None):
        snippet_pos = np.array([0, row_data["length"]])
        snippet_rates = np.array([row_data["rec_rate"] * 1e6 * 100, 0])
        snippet_map = np.array([0, (row_data["length"] * row_data["rec_rate"]) * 100])
        genetic_map_output = pd.DataFrame(
            data=np.stack([snippet_pos, snippet_rates, snippet_map], axis=1),
            columns=["position", "COMBINED_rate.cM.Mb.", "Genetic_Map.cM."])
        if param:
            path_to_genetic_map = os.path.join(
                self.data_dir, self.name + "_" + param + "_uniform_genetic_map.txt")
        else:
            path_to_genetic_map = os.path.join(
                self.data_dir, self.name + "_uniform_genetic_map.txt")
        genetic_map_output.to_csv(path_to_genetic_map, sep=' ', index=False)
        return path_to_genetic_map

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
        self.sample_sizes = [10, 260, 526, 790, 1052, 1316, 1578, 1842, 2106,
                             2368, 2632, 2894, 3158, 3421, 3684, 3948, 4210,
                             4474, 4736, 5000]
        # self.sample_sizes = [10, 20, 40, 64, 100]
        self.sim_cols = self.sim_cols + ["filename", "replicate", "sample_size", "Ne",
                                         "length", "mut_rate", "rec_rate", "n_edges",
                                         "n_trees", "n_sites", "seed", "tsdate_cpu",
                                         "tsdate_memory", "tsinfer_cpu",
                                         "tsinfer_memory", "tsdate_infer_cpu",
                                         "tsdate_infer_memory", "relate_cpu",
                                         "relate_memory", "geva_cpu", "geva_memory"]
        self.tools = ["tsdate", "tsinfer", "relate", "geva"]
        self.num_rows = len(self.sample_sizes) * self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["Ne"] = 10000
        row_data["length"] = 5e6
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_fn(params):
            sample_size = params[0]
            seed = params[1]
            return evaluation.run_neutral_sim(sample_size=sample_size,
                                              mutation_rate=row_data["mut_rate"],
                                              recombination_rate=row_data["rec_rate"],
                                              Ne=row_data["Ne"],
                                              length=row_data["length"], seed=seed)
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

            _, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mut_rate"], 20,
                "inside_outside")
            self.data.loc[index, ["tsdate_cpu", "tsdate_memory"]] = \
                tsdate_cpu, tsdate_memory

            _, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
                path_to_file + ".samples", sim.get_sequence_length())
            self.data.loc[index, ["tsinfer_cpu", "tsinfer_memory"]] = \
                [tsinfer_cpu, tsinfer_memory]

            _, dated_infer_cpu, dated_infer_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mut_rate"], 20,
                "inside_outside")
            self.data.loc[index, ["tsdate_infer_cpu", "tsdate_infer_memory"]] = \
                dated_infer_cpu, dated_infer_memory

            path_to_genetic_map = self.make_uniform_genetic_map(row)
            relate_dir = os.path.join(self.data_dir, "relate_" + row["filename"])
            _, _, relate_cpu, relate_memory = evaluation.run_relate(
                sim, path_to_file, row["mut_rate"], row["Ne"] * 2, path_to_genetic_map,
                relate_dir, "relate_file")

            self.data.loc[index, ["relate_cpu", "relate_memory"]] = \
                [relate_cpu, relate_memory]
            _, geva_cpu, geva_memory = evaluation.run_geva(
                path_to_file, row["Ne"], row["mut_rate"], row["rec_rate"])
            self.data.loc[index, ["geva_cpu", "geva_memory"]] = \
                [geva_cpu, geva_memory]
            self.summarize()
            # Delete all generated files to save diskspace
            self.clear(row["filename"])

    def clear(self, filename):
        """
        To save disk space, delete tree sequences, VCFs, relate subdirectories etc.
        associated with this index.
        """
        for file in os.listdir(self.data_dir):
            if file.startswith(filename) and not file.endswith(".csv"):
                os.remove(os.path.join(self.data_dir, file))

        for relate_subdir in os.walk(self.data_dir):
            if relate_subdir[0].startswith(os.path.join(self.data_dir, "relate_" + filename)):
                shutil.rmtree(os.path.join(self.data_dir, relate_subdir[0]))


class CpuScalingLength(CpuScalingSampleSize):
    """
    Plot CPU times of tsdate, tsinfer, tsdate+tsinfer, Relate, and GEVA with increasing
    lengths of simulated sequence.
    """
    name = "cpu_scaling_length"

    def __init__(self):
        CpuScalingSampleSize.__init__(self)
        # self.lengths = [1e4, 1e5]
        self.lengths = np.linspace(1e5, 10e6, 20)

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["Ne"] = 10000
        row_data["sample_size"] = 1000
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_fn(params):
            length = params[0]
            seed = params[1]
            return evaluation.run_neutral_sim(sample_size=row_data["sample_size"],
                                              mutation_rate=row_data["mut_rate"],
                                              recombination_rate=row_data["rec_rate"],
                                              Ne=row_data["Ne"],
                                              length=length, seed=seed)
        DataGeneration.setup(self, "length", self.lengths, simulate_fn, row_data)


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
            row_data["length"] = 1e6
            row_data["mut_rate"] = 1e-8
            row_data["rec_rate"] = 1e-8

            def simulate_fn(params):
                seed = params[1]
                return evaluation.run_neutral_sim(sample_size=row_data["sample_size"],
                                                  mutation_rate=row_data["mut_rate"],
                                                  recombination_rate=row_data["rec_rate"],
                                                  Ne=row_data["Ne"],
                                                  length=row_data["length"], seed=seed)
            DataGeneration.setup(self, None, [None], simulate_fn, row_data)

            # # Randomly sample mutations from each sampledata file
            # self.data = pd.read_csv(self.data_file)
            # for index, row in self.data.iterrows():
            #     samples = tsinfer.load(os.path.join(self.data_dir,
            #                                         row["filename"] + ".samples"))
            #     mut_pos = pd.DataFrame(columns=['CHR', 'POS'])
            #     mut_pos['POS'] = sorted(
            #         np.round(self.rng.sample(list(samples.sites_position[:]),
            #                                  1000)).astype(int))
            #     mut_pos['CHR'] = 1
            #     mut_pos.to_csv(
            #         os.path.join(self.data_dir,
            #                      row["filename"] + '.subsampled.pos.csv'),
            #         index=False, header=False, sep=" ")
            #     subprocess.check_output(
            #         ['vcftools', '--vcf', os.path.join(
            #             self.data_dir, row["filename"] + '.vcf'), '--positions',
            #          os.path.join(self.data_dir, row["filename"] + '.subsampled.pos.csv'),'--recode', '--recode-INFO-all', '--out', os.path.join(self.data_dir, row["filename"] + '.subsampled')]) 

    def inference(self):
        """
        Run four methods on the simulated data
        """
        try:
            self.data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            logging.error("Must run with --setup flag first")

        # Name of output file with mutations ages
        output_fn = os.path.join(self.data_dir, self.name + "_mutations.csv")
        for index, row in tqdm(self.data.iterrows(), desc="Running Inference"):
            path_to_file = os.path.join(self.data_dir, row["filename"])
            sim = tskit.load(path_to_file + ".trees")

            dated_ts, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mut_rate"], 20,
                "inside_outside")
            dated_ts.dump(path_to_file + ".tsdated.trees")

            inferred_ts, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
                path_to_file + ".samples", sim.get_sequence_length())
            inferred_ts.simplify().dump(path_to_file + ".tsinferred.trees")

            dated_inferred_ts, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".tsinferred.trees", row["Ne"], row["mut_rate"], 50,
                "inside_outside")
            dated_inferred_ts.dump(path_to_file + ".tsinferred.tsdated.trees")

            relate_dir = os.path.join(self.data_dir, "relate_" + row["filename"])
            path_to_genetic_map = self.make_uniform_genetic_map(row)
            relate_ts, relate_age, relate_cpu, relate_memory = evaluation.run_relate(
                sim, path_to_file, row["mut_rate"], row["Ne"] * 2, 
                path_to_genetic_map, relate_dir, "relate_run")
            relate_age.to_csv(path_to_file + "relate_age")

            geva_ages, geva_cpu, geva_memory = evaluation.run_geva(
                path_to_file, row["Ne"], row["mut_rate"], row["rec_rate"])
            geva_ages.to_csv(path_to_file + ".geva.csv")
            geva_positions = pd.read_csv(
                path_to_file + ".marker.txt", delimiter=" ", index_col="MarkerID")

            compare_df = evaluation.compare_mutations(
                ["simulated_ts", "tsdate", "tsdate_inferred", "geva", "relate"],
                [sim, dated_ts, dated_inferred_ts],
                geva_ages=geva_ages, geva_positions=geva_positions,
                relate_ages=relate_age)
            if index != 0:
                master_df = pd.read_csv(output_fn, index_col=0)
                pd.concat([master_df, compare_df]).to_csv(output_fn)
            else:
                compare_df.to_csv(output_fn)
            self.summarize()
        # Save dataframe
        self.summarize()


class Chr20SimulatedMutationAccuracy(DataGeneration):
    name = "chr20_simulated_mutation_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.tools = ["tsdate", "tsinfer", "relate", "geva"]
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.default_replicates = 1

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["sample_size"] = 250
        row_data["Ne"] = 10000
        row_data["length"] = 5e6
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_fn(params):
            seed = params[1]
            species = stdpopsim.get_species("HomSap")
            contig = species.get_contig("chr20", genetic_map='HapMapII_GRCh37')
            model = species.get_demographic_model('OutOfAfrica_3G09')
            samples = model.get_samples(100)
            engine = stdpopsim.get_default_engine()
            ts = engine.simulate(model, contig, samples, seed=seed)
            return ts.keep_intervals(np.array([[20e6, 25e6]])).trim()
        DataGeneration.setup(self, None, [None], simulate_fn, row_data)

    def get_genetic_map_chr20_snippet(self):
        """
        For each chromosome 20 simulation, select a 10mb region to run inference on
        """
        species = stdpopsim.get_species("HomSap")
        genetic_map = species.get_genetic_map("HapMapII_GRCh37")
        cm = genetic_map.get_chromosome_map('chr20')
        pos = np.array(cm.get_positions())
        snippet = np.where(np.logical_and(pos > 1e6, pos < 5e6))
        snippet_pos = pos[snippet] - 1e6
        snippet_rates = np.array(cm.get_rates())[snippet]
        map_distance = np.concatenate(
            [[0], (np.diff(snippet_pos) * snippet_rates[:-1]) / 1e6])
        genetic_map_output = pd.DataFrame(
            data=np.stack([snippet_pos, snippet_rates, map_distance], axis=1),
            columns=["position", "COMBINED_rate.cM.Mb.", "Genetic_Map.cM."])
        path_to_genetic_map = os.path.join(self.data_dir, "genetic_map_chr20_snippet")
        genetic_map_output.to_csv(path_to_genetic_map, sep=' ', index=False)
        return genetic_map_output

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

            dated_ts, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".trees", row["Ne"], row["mut_rate"], 20,
                "inside_outside")
            dated_ts.dump(path_to_file + ".tsdated.trees")

            inferred_ts, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
                path_to_file + ".samples", sim.get_sequence_length())
            inferred_ts.simplify().dump(path_to_file + ".tsinferred.trees")

            dated_inferred_ts, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
                path_to_file + ".tsinferred.trees", row["Ne"], row["mut_rate"], 50,
                "inside_outside")
            dated_inferred_ts.dump(path_to_file + ".tsinferred.tsdated.trees")

            relate_dir = os.path.join(self.data_dir, "relate_" + row["filename"])
            path_to_genetic_map = self.get_genetic_map_chr20_snippet()
            relate_ts, relate_age, relate_cpu, relate_memory = evaluation.run_relate(
                sim, path_to_file, row["mut_rate"], row["Ne"] * 2, path_to_genetic_map,
                relate_dir, "relate_run")
            relate_age.to_csv(path_to_file + "relate_age")

            geva_ages, geva_cpu, geva_memory = evaluation.run_geva(
                path_to_file, row["Ne"], row["mut_rate"], row["rec_rate"])
            geva_ages.to_csv(path_to_file + ".geva.csv")
            geva_positions = pd.read_csv(path_to_file + ".marker.txt",
                                         delimiter=" ", index_col="MarkerID")
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

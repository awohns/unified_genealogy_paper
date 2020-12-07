#!/usr/bin/env python3
"""
Simulation based evaluation for the unified genealogy paper
Pattern for generating data for each figure is:
 python3 src/run_evaluation.py PLOT_NAME --setup
 python3 src/run_evaluation.py PLOT_NAME --inference
"""
import argparse
import pickle

import logging
import os
import pandas as pd
import multiprocessing
import numpy as np
import random
import shutil
from tqdm import tqdm
import scipy
from sklearn.metrics import mean_squared_log_error

import tskit
import tsinfer
import msprime
import stdpopsim
import tsdate
from tsdate.prior import SpansBySamples, ConditionalCoalescentTimes

import evaluation
import constants
import iteration
import utility
import run_inference
import error_generation


class DataGeneration:
    """
    Superclass for data generation classes for each figure.
    """

    # Default settings
    default_replicates = 10
    default_seed = 123
    data_dir = os.path.join(os.getcwd(), "simulated-data/")

    # Each summary has a unique name. This is used as the identifier for the csv file.
    name = None

    def __init__(self):
        self.data_file = os.path.abspath(
            os.path.join(self.data_dir, self.name + ".csv")
        )
        self.rng = random.Random(self.default_seed)
        self.sim_cols = [
            "filename",
            "replicate",
            "sample_size",
            "Ne",
            "length",
            "rec_rate",
            "mut_rate",
            "n_edges",
            "n_trees",
            "n_sites",
            "seed",
        ]
        self.make_vcf = True
        self.empirical_error = False
        self.ancestral_state_error = False

    def setup(
        self,
        parameter,
        parameter_arr,
        simulate_func,
        genetic_map_func,
        row_data,
        progress=False,
    ):
        """
        Run Simulations
        """
        for param in tqdm(
            parameter_arr, desc="Running Simulations", disable=not progress
        ):
            seeds = [
                self.rng.randint(1, 2 ** 31) for i in range(self.default_replicates)
            ]
            for index, seed in tqdm(
                enumerate(seeds), desc="Running Iterations", total=len(seeds)
            ):
                sim = simulate_func((param, seed))
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
                samples = tsinfer.formats.SampleData.from_tree_sequence(
                    sim,
                    use_sites_time=False,
                )
                sample_data_indiv_times = samples.copy(
                    path=os.path.join(self.data_dir, filename + ".samples")
                )
                sample_data_indiv_times.individuals_time[:] = np.array(
                    sim.tables.nodes.time[sim.samples()]
                )
                sample_data_indiv_times.finalise()

                # Add error to sampledata file
                if self.empirical_error:
                    error_samples = error_generation.add_errors(sample_data_indiv_times)
                    copy = error_samples.copy(
                        os.path.join(self.data_dir, filename + ".error.samples")
                    )
                    copy.finalise()

                # Add error to sampledata file
                if self.ancestral_state_error:
                    anc_error_samples = error_generation.add_errors(
                        sample_data_indiv_times, ancestral_allele_error=0.01
                    )
                    copy = anc_error_samples.copy(
                        os.path.join(
                            self.data_dir, filename + ".ancestral_state.error.samples"
                        )
                    )
                    copy.finalise()

                # Create VCF file
                if self.make_vcf:
                    with open(
                        os.path.join(self.data_dir, filename + ".vcf"), "w"
                    ) as vcf_file:
                        sim.write_vcf(vcf_file, ploidy=2, position_transform="legacy")
                    if self.empirical_error:
                        evaluation.sampledata_to_vcf(
                            error_samples,
                            os.path.join(self.data_dir, filename + ".error"),
                        )
                    if self.ancestral_state_error:
                        evaluation.sampledata_to_vcf(
                            anc_error_samples,
                            os.path.join(
                                self.data_dir, filename + ".ancestral_state.error"
                            ),
                        )

                # Create the genetic map
                genetic_map_func(row_data, filename)

                # Update dataframe with details of simulation
                self.data = self.data.append(row_data, ignore_index=True)

        # Save dataframe
        self.summarize()

    def make_genetic_map(self, row_data, filename):
        pos = np.array([0, row_data["length"]])
        rates = np.array([row_data["rec_rate"] * 1e6 * 100, 0])
        genetic_pos = np.array([0, (row_data["length"] * row_data["rec_rate"]) * 100])
        genetic_map_output = pd.DataFrame(
            data=np.stack([pos, rates, genetic_pos], axis=1),
            columns=["position", "COMBINED_rate.cM.Mb.", "Genetic_Map.cM."],
        )
        path_to_genetic_map = os.path.join(
            os.path.join(self.data_dir, filename + "_genetic_map.txt")
        )
        genetic_map_output.to_csv(path_to_genetic_map, sep=" ", index=False)
        return path_to_genetic_map

    def run_multiprocessing(self, function, num_processes=1):
        """
        Run multiprocessing of inputted function a specified number of times
        """
        try:
            self.data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            logging.error("Must run with --setup flag first")

        if num_processes > 1:
            logging.info(
                "Setting up using multiprocessing ({} processes)".format(num_processes)
            )
            with multiprocessing.Pool(
                processes=num_processes, maxtasksperchild=10
            ) as pool:

                for index, row in tqdm(
                    pool.imap_unordered(function, self.data.iterrows()),
                    desc="Inference Run",
                    total=self.data.shape[0],
                ):
                    self.data.loc[index] = row
                    self.summarize()
        else:
            # When we have only one process it's easier to keep everything in the
            # same process for debugging.
            logging.info("Setting up using a single process")
            for index, row in map(function, self.data.iterrows()):
                logging.info("Running inference")
                self.data.loc[index] = row
                self.summarize()

    def summarize(self):
        """
        Take the output of the inference and save to CSV
        """
        self.data.to_csv("simulated-data/" + self.name + ".csv")


class NeutralSimulatedMutationAccuracy(DataGeneration):
    """
    Template for mutation-based evaluation of various methods.
    Generates data for supplemental figure 4.
    """

    name = "neutral_simulated_mutation_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred", "relate", "geva"]
        self.output_suffixes = [
            "_mutations.csv",
            "_error_mutations.csv",
            "_anc_error_mutations.csv",
            "_kc_distances.csv",
            "_error_kc_distances.csv",
            "_anc_error_kc_distances.csv",
        ]
        self.sim_cols = self.sim_cols
        self.default_replicates = 30
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.empirical_error = False
        self.ancestral_state_error = False
        self.tsinfer_mismatch = False
        self.tsinfer_iterate = False
        self.relate_reinfer = False

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["sample_size"] = 500
        row_data["Ne"] = 10000
        row_data["length"] = 5e6
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_func(params):
            seed = params[1]
            return evaluation.run_neutral_sim(
                sample_size=row_data["sample_size"],
                mutation_rate=row_data["mut_rate"],
                recombination_rate=row_data["rec_rate"],
                Ne=row_data["Ne"],
                length=row_data["length"],
                seed=seed,
            )

        DataGeneration.setup(
            self, None, [None], simulate_func, self.make_genetic_map, row_data
        )

    def run_multiprocessing(self, function, num_processes=1):
        """
        Run multiprocessing of inputted function a specified number of times
        """
        try:
            self.data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            logging.error("Must run with --setup flag first")

        output_names = []
        for name in self.output_suffixes:
            output_names.append(os.path.join(self.data_dir, self.name + name))
        master_dfs = [
            pd.DataFrame(columns=self.columns)
            for index in range(len(self.output_suffixes))
        ]
        if num_processes > 1:
            logging.info(
                "Setting up using multiprocessing ({} processes)".format(num_processes)
            )
            with multiprocessing.Pool(
                processes=num_processes, maxtasksperchild=10
            ) as pool:

                for index, row, dfs in tqdm(
                    pool.imap_unordered(function, self.data.iterrows()),
                    desc="Inference Run",
                    total=self.data.shape[0],
                ):
                    self.data.loc[index] = row
                    self.summarize()
                    for index, (name, df) in enumerate(dfs.items()):
                        master_dfs[index] = pd.concat(
                            [master_dfs[index], df], sort=False
                        )

        else:
            # When we have only one process it's easier to keep everything in the
            # same process for debugging.
            logging.info("Setting up using a single process")
            for index, row, dfs in tqdm(
                map(function, self.data.iterrows()), total=self.data.shape[0]
            ):
                logging.info("Running inference")
                self.data.loc[index] = row
                self.summarize()
                for index, (name, df) in enumerate(dfs.items()):
                    master_dfs[index] = pd.concat([master_dfs[index], df], sort=False)

        for master_df, output_name in zip(master_dfs, output_names):
            master_df.to_csv(output_name)

    def inference(self, row_data):
        """
        Run four methods on the simulated data
        """
        index = row_data[0]
        row = row_data[1]

        # Name of output file with mutations ages
        path_to_file = os.path.join(self.data_dir, row["filename"])
        sim = tskit.load(path_to_file + ".trees")

        dated_ts = tsdate.date(sim, row["Ne"], row["mut_rate"])
        dated_ts.dump(path_to_file + ".tsdated.trees")

        sample_data = tsinfer.load(path_to_file + ".samples")

        def run_all_inference(
            sim,
            dated_ts,
            samples,
            output_fn,
            tsinfer_mismatch=False,
            relate_reinfer=False,
        ):
            compare_ts_dict = {"simulated_ts": sim, "tsdate": dated_ts}
            if self.tsinfer_mismatch:
                _, inferred_ts = evaluation.infer_with_mismatch(
                    samples, "chr20", ma_mismatch=0.1, ms_mismatch=0.1, num_threads=1
                )
                inferred_ts = tsdate.preprocess_ts(inferred_ts)
            else:
                inferred_ts = tsinfer.infer(samples).simplify()
            inferred_ts.dump(path_to_file + output_fn + ".tsinferred.trees")
            dated_inferred_ts = tsdate.date(inferred_ts, row["Ne"], row["mut_rate"])
            dated_inferred_ts.dump(
                path_to_file + output_fn + ".tsinferred.tsdated.trees"
            )
            compare_ts_dict["tsdate_inferred"] = dated_inferred_ts
            if self.tsinfer_iterate:
                dated_samples = tsdate.get_sites_time(
                    dated_inferred_ts, samples=samples
                )
                if self.tsinfer_mismatch:
                    _, reinferred_ts = evaluation.infer_with_mismatch(
                        dated_samples,
                        "chr20",
                        ma_mismatch=0.1,
                        ms_mismatch=0.1,
                        num_threads=1,
                    )
                    reinferred_ts = tsdate.preprocess_ts(reinferred_ts)
                else:
                    reinferred_ts = tsinfer.infer(dated_samples).simplify()
                reinferred_ts.dump(path_to_file + output_fn + ".iter.tsinferred.trees")
                compare_ts_dict["iter_tsdate_inferred"] = redated_inferred_ts
                redated_inferred_ts = tsdate.date(
                    reinferred_ts, row["Ne"], row["mut_rate"]
                )
                redated_inferred_ts.dump(
                    path_to_file + output_fn + ".iter.tsinferred.tsdated.trees"
                )
            else:
                redated_inferred_ts = None

            relate_dir = os.path.join(
                self.data_dir, "relate_" + row["filename"] + output_fn
            )
            path_to_genetic_map = path_to_file + "_genetic_map.txt"
            relate_ts, relate_age, relate_cpu, relate_memory = evaluation.run_relate(
                sim,
                path_to_file + output_fn,
                row["mut_rate"],
                row["Ne"] * 2,
                path_to_genetic_map,
                relate_dir,
                "relate_run" + output_fn,
            )
            relate_ts.dump(path_to_file + output_fn + ".relate.trees")
            relate_age.to_csv(path_to_file + output_fn + ".relate_age.csv")
            if self.relate_reinfer:
                relate_iter_ages, relate_iter_ts = evaluation.run_relate_pop_size(
                    sim,
                    "relate_run" + output_fn,
                    row["mut_rate"],
                    "relate_reinfer" + output_fn,
                    relate_dir,
                )
            else:
                relate_iter_ages = None
                relate_iter_ts = None
            geva_ages, geva_cpu, geva_memory = evaluation.run_geva(
                path_to_file + output_fn, row["Ne"], row["mut_rate"], row["rec_rate"]
            )
            geva_positions = pd.read_csv(
                path_to_file + output_fn + ".marker.txt",
                delimiter=" ",
                index_col="MarkerID",
            )
            geva_ages.to_csv(path_to_file + output_fn + ".geva.csv")
            print("Compare Mutations")
            mutation_df = evaluation.compare_mutations(
                list(compare_ts_dict.values()),
                list(compare_ts_dict.keys()),
                geva_ages=geva_ages,
                geva_positions=geva_positions,
                relate_ages=relate_age,
                relate_reinfer=relate_iter_ages,
            )

            sim_pos = sim.tables.sites.position
            sim = sim.keep_intervals(
                [[np.round(sim_pos[0]), np.round(sim_pos[-1])]]
            ).trim()

            dated_ts_pos = dated_ts.tables.sites.position
            dated_ts = dated_ts.keep_intervals(
                [[np.round(dated_ts_pos[0]), np.round(dated_ts_pos[-1])]]
            ).trim()
            compare_ts_dict = {"simulated_ts": sim, "tsdate": dated_ts}

            dated_inferred_ts_pos = dated_inferred_ts.tables.sites.position
            dated_inferred_ts = dated_inferred_ts.keep_intervals(
                [
                    [
                        np.round(dated_inferred_ts_pos[0]),
                        np.round(dated_inferred_ts_pos[-1]),
                    ]
                ]
            ).trim()
            tables = dated_inferred_ts.dump_tables()
            tables.sequence_length = sim.get_sequence_length()
            dated_inferred_ts = tables.tree_sequence()
            compare_ts_dict["tsdate_inferred"] = dated_inferred_ts

            if self.relate_reinfer:
                relate_ts_pos = relate_retrees.tables.sites.position
                relate_ts = relate_retrees.keep_intervals(
                    [[np.round(relate_ts_pos[0]), np.round(relate_ts_pos[-1])]]
                ).trim()
                compare_ts_dict["relate_iterate"] = relate_ts
            else:
                relate_ts_pos = relate_ts.tables.sites.position
                relate_ts = relate_ts.keep_intervals(
                    [
                        [
                            np.round(relate_ts.tables.sites.position[0]),
                            np.round(relate_ts.tables.sites.position[-1]),
                        ]
                    ]
                ).trim()
                compare_ts_dict["relate"] = relate_ts

            print("Find KC Distances")
            if self.tsinfer_iterate:
                redated_inferred_ts_pos = redated_inferred_ts.tables.sites.position
                redated_inferred_ts = redated_inferred_ts.keep_intervals(
                    [
                        [
                            np.round(redated_inferred_ts_pos[0]),
                            np.round(redated_inferred_ts_pos[-1]),
                        ]
                    ]
                ).trim()
                tables = redated_inferred_ts.dump_tables()
                tables.sequence_length = sim.get_sequence_length()
                redated_inferred_ts = tables.tree_sequence()
                compare_ts_dict["tsdate_iterate"] = redated_inferred_ts
            kc_df = evaluation.get_kc_distances(
                list(compare_ts_dict.values()), list(compare_ts_dict.keys())
            )
            return mutation_df, kc_df

        mut_df, kc_df = run_all_inference(sim, dated_ts, sample_data, "")
        error_mut_df = None
        error_kc_df = None
        if self.empirical_error:
            error_samples = tsinfer.load(path_to_file + ".error.samples")
            error_mut_df, error_kc_df = run_all_inference(
                sim, dated_ts, error_samples, ".error"
            )
        anc_error_mut_df = None
        anc_error_kc_df = None
        if self.ancestral_state_error:
            anc_error_samples = tsinfer.load(
                path_to_file + ".ancestral_state.error.samples"
            )
            anc_error_mut_df, anc_error_kc_df = run_all_inference(
                sim, dated_ts, anc_error_samples, ".ancestral_state.error"
            )
        return_vals = {
            "muts_noerr": mut_df,
            "muts_err": error_mut_df,
            "muts_anc_err": anc_error_mut_df,
            "kc_noerr": kc_df,
            "kc_err": error_kc_df,
            "kc_anc_err": anc_error_kc_df,
        }
        print(mut_df)
        return index, row, return_vals


class TsdateNeutralSimulatedMutationAccuracy(NeutralSimulatedMutationAccuracy):
    """
    Generate data for figure 1b: tsdate evaluation on neutral simulations
    """

    name = "tsdate_neutral_simulated_mutation_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred"]
        self.output_suffixes = ["_mutations.csv"]
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)

    def inference(self, row_data):
        """
        Run tsdate and tsinfer+tsdate on the simulated data
        """
        index = row_data[0]
        row = row_data[1]
        # Name of output file with mutations ages
        path_to_file = os.path.join(self.data_dir, row["filename"])
        sim = tskit.load(path_to_file + ".trees")
        samples = tsinfer.load(path_to_file + ".samples")
        dated_ts = tsdate.date(sim, row["Ne"], row["mut_rate"])
        dated_ts.dump(path_to_file + ".tsdated.trees")

        inferred_ts = tsinfer.infer(samples).simplify()
        inferred_ts.dump(path_to_file + ".tsinferred.trees")

        dated_inferred_ts = tsdate.date(inferred_ts, row["Ne"], row["mut_rate"])
        dated_inferred_ts.dump(path_to_file + ".tsinferred.tsdated.trees")

        compare_df = evaluation.compare_mutations(
            [sim, dated_ts, dated_inferred_ts],
            ["simulated_ts", "tsdate", "tsdate_inferred"],
        )
        return_vals = {"mut_df": compare_df}
        return index, row, return_vals


class CpuScalingSampleSize(DataGeneration):
    """
    Plot CPU times of tsdate, tsinfer, tsdate+tsinfer, Relate, and GEVA
    Run the following to occupy other threads: nice -n 15 stress -c 40
    WARNING: GEVA uses a *large* amount of memory, ~20Gb per run when the SampleSize
    is 2000.
    """

    name = "cpu_scaling_samplesize"
    default_replicates = 5
    include_geva = True

    def __init__(self):
        DataGeneration.__init__(self)
        self.sample_sizes = np.linspace(110, 2000, 10, dtype=int)
        self.length = 1e6
        self.sim_cols = self.sim_cols + [
            "filename",
            "replicate",
            "sample_size",
            "Ne",
            "length",
            "mut_rate",
            "rec_rate",
            "n_edges",
            "n_trees",
            "n_sites",
            "seed",
            "tsdate_cpu",
            "tsdate_memory",
            "tsinfer_cpu",
            "tsinfer_memory",
            "tsdate_infer_cpu",
            "tsdate_infer_memory",
            "relate_cpu",
            "relate_memory",
            "geva_cpu",
            "geva_memory",
        ]
        if self.include_geva:
            self.tools = ["tsdate", "tsinfer", "relate", "geva"]
        else:
            self.tools = ["tsdate", "tsinfer", "relate"]
        self.num_rows = len(self.sample_sizes) * self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["Ne"] = 10000
        row_data["length"] = self.length
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_func(params):
            sample_size = params[0]
            seed = params[1]
            return evaluation.run_neutral_sim(
                sample_size=sample_size,
                mutation_rate=row_data["mut_rate"],
                recombination_rate=row_data["rec_rate"],
                Ne=row_data["Ne"],
                length=row_data["length"],
                seed=seed,
            )

        DataGeneration.setup(
            self,
            "sample_size",
            self.sample_sizes,
            simulate_func,
            self.make_genetic_map,
            row_data,
        )

    def inference(self, row_data):
        """
        Run four methods on the simulated data
        """
        index = row_data[0]
        row = row_data[1]
        path_to_file = os.path.join(self.data_dir, row["filename"])
        sim = tskit.load(path_to_file + ".trees")

        _, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
            path_to_file + ".trees", row["Ne"], row["mut_rate"], 20, "inside_outside"
        )
        row["tsdate_cpu", "tsdate_memory"] = tsdate_cpu, tsdate_memory

        _, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
            path_to_file + ".samples", sim.get_sequence_length()
        )
        row["tsinfer_cpu", "tsinfer_memory"] = [tsinfer_cpu, tsinfer_memory]

        _, dated_infer_cpu, dated_infer_memory = evaluation.run_tsdate(
            path_to_file + ".trees", row["Ne"], row["mut_rate"], 20, "inside_outside"
        )
        row["tsdate_infer_cpu", "tsdate_infer_memory"] = (
            dated_infer_cpu,
            dated_infer_memory,
        )

        relate_dir = os.path.join(self.data_dir, "relate_" + row["filename"])
        path_to_genetic_map = path_to_file + "_genetic_map.txt"
        _, _, relate_cpu, relate_memory = evaluation.run_relate(
            sim,
            path_to_file,
            row["mut_rate"],
            row["Ne"] * 2,
            path_to_genetic_map,
            relate_dir,
            "relate_file",
        )

        row["relate_cpu", "relate_memory"] = [relate_cpu, relate_memory]

        if self.include_geva:
            _, geva_cpu, geva_memory = evaluation.run_geva(
                path_to_file, row["Ne"], row["mut_rate"], row["rec_rate"]
            )
            row["geva_cpu", "geva_memory"] = [geva_cpu, geva_memory]

        # Delete all generated files to save diskspace
        self.clear(row["filename"])

        return index, row

    def clear(self, filename):
        """
        To save disk space, delete tree sequences, VCFs, relate subdirectories etc.
        associated with this index.
        """
        for file in os.listdir(self.data_dir):
            if file.startswith(filename) and not file.endswith(".csv"):
                os.remove(os.path.join(self.data_dir, file))

        for relate_subdir in os.walk(self.data_dir):
            if relate_subdir[0].startswith(
                os.path.join(self.data_dir, "relate_" + filename)
            ):
                shutil.rmtree(os.path.join(self.data_dir, relate_subdir[0]))


class CpuScalingLength(CpuScalingSampleSize):
    """
    Plot CPU times of tsdate, tsinfer, tsdate+tsinfer, Relate, and GEVA with increasing
    lengths of simulated sequence.
    """

    name = "cpu_scaling_length"

    def __init__(self):
        CpuScalingSampleSize.__init__(self)
        self.lengths = np.linspace(1e5, 1e7, 10, dtype=int)
        # self.lengths = np.linspace(1e5, 5e5, 4, dtype=int)
        self.sample_size = 500

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["Ne"] = 10000
        row_data["sample_size"] = self.sample_size
        # row_data["sample_size"] = 50
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_func(params):
            length = params[0]
            seed = params[1]
            return evaluation.run_neutral_sim(
                sample_size=row_data["sample_size"],
                mutation_rate=row_data["mut_rate"],
                recombination_rate=row_data["rec_rate"],
                Ne=row_data["Ne"],
                length=length,
                seed=seed,
            )

        DataGeneration.setup(
            self, "length", self.lengths, simulate_func, self.make_genetic_map, row_data
        )


class IterationTsdate(NeutralSimulatedMutationAccuracy):
    name = "tsdate_iteration_neutral_simulated_mutation_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = [
            "simulated_ts",
            "tsdate",
            "tsdate_1stiter",
            "tsdate_2nditer",
            "tsdate_inferred",
            "tsdate_inferred_1stiter",
            "tsdate_inferred_2nditer",
            "tsdate_inferred_error",
            "tsdate_inferred_error_1stiter",
            "tsdate_inferred_error_2nditer",
        ]
        self.default_replicates = 50
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.empirical_error = False
        self.make_vcf = False

    def inference(self, row_data):
        """
        Run tsdate and tsinfer+tsdate on the simulated data, with iterations
        """
        index = row_data[0]
        row = row_data[1]
        Ne = row["Ne"]
        mut_rate = row["mut_rate"]
        # Name of output file with mutations ages
        path_to_file = os.path.join(self.data_dir, row["filename"])
        sim = tskit.load(path_to_file + ".trees")
        priors = tsdate.build_prior_grid(sim, timepoints=20)
        dated_ts, dates, posterior = evaluation.run_tsdate_posterior_ts(
            sim, Ne, mut_rate, "inside_outside", priors
        )
        iter_dated_1, dates_1, posterior = evaluation.tsdate_iter(
            sim, Ne, mut_rate, "inside_outside", priors, posterior
        )
        iter_dated_2, dates_2, _ = evaluation.tsdate_iter(
            sim, Ne, mut_rate, "inside_outside", priors, posterior
        )

        # Infer and date SampleData file
        inferred_ts, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
            path_to_file + ".samples", sim.get_sequence_length()
        )
        inferred_ts = inferred_ts.simplify()
        priors = tsdate.build_prior_grid(inferred_ts, timepoints=20)
        (
            dated_inferred_ts,
            inferred_dates,
            posterior,
        ) = evaluation.run_tsdate_posterior_ts(
            inferred_ts, Ne, mut_rate, "inside_outside", priors
        )
        infer_iter_dated_1, inferred_dates_1, posterior = evaluation.tsdate_iter(
            inferred_ts, Ne, mut_rate, "inside_outside", priors, posterior
        )
        infer_iter_dated_2, inferred_dates_2, _ = evaluation.tsdate_iter(
            inferred_ts, Ne, mut_rate, "inside_outside", priors, posterior
        )

        # Infer and date SampleData file with error
        error_inferred_ts, _, _ = evaluation.run_tsinfer(
            path_to_file + ".error.samples", sim.get_sequence_length()
        )
        error_inferred_ts = error_inferred_ts.simplify()
        priors = tsdate.build_prior_grid(error_inferred_ts, timepoints=20)
        (
            dated_error_inferred_ts,
            error_inferred_dates,
            posterior,
        ) = evaluation.run_tsdate_posterior_ts(
            error_inferred_ts, Ne, mut_rate, "inside_outside", priors
        )
        (
            error_infer_iter_dated_1,
            error_inferred_dates_1,
            posterior,
        ) = evaluation.tsdate_iter(
            error_inferred_ts, Ne, mut_rate, "inside_outside", priors, posterior
        )
        error_infer_iter_dated_2, error_inferred_dates_2, _ = evaluation.tsdate_iter(
            error_inferred_ts, Ne, mut_rate, "inside_outside", priors, posterior
        )
        compare_df = evaluation.compare_mutations_tslist(
            [
                sim,
                sim,
                sim,
                sim,
                dated_inferred_ts,
                dated_inferred_ts,
                dated_inferred_ts,
                dated_error_inferred_ts,
                dated_error_inferred_ts,
                dated_error_inferred_ts,
            ],
            [
                sim.tables.nodes.time,
                dates,
                dates_1,
                dates_2,
                inferred_dates,
                inferred_dates_1,
                inferred_dates_2,
                error_inferred_dates,
                error_inferred_dates_1,
                error_inferred_dates_2,
            ],
            [
                "simulated_ts",
                "tsdate",
                "tsdate_1stiter",
                "tsdate_2nditer",
                "tsdate_inferred",
                "tsdate_inferred_1stiter",
                "tsdate_inferred_2nditer",
                "tsdate_inferred_error",
                "tsdate_inferred_error_1stiter",
                "tsdate_inferred_error_2nditer",
            ],
        )
        return index, row, compare_df


class Chr20SimulatedMutationAccuracy(NeutralSimulatedMutationAccuracy):
    name = "chr20_simulated_mutation_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred", "relate", "geva"]
        self.default_replicates = 1
        self.num_rows = self.default_replicates
        self.sim_cols = self.sim_cols + ["snippet"]
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.empirical_error = True
        self.ancestral_state_error = True
        self.tsinfer_mismatch = True
        self.tsinfer_iterate = True
        self.relate_reinfer = True
        self.modern_sample_size = 150
        self.ancient_sample_size = 0
        self.remove_ancient_mutations = False
        self.ancient_times = None

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["sample_size_modern"] = self.modern_sample_size
        row_data["sample_size_ancient"] = self.ancient_sample_size
        if self.ancient_sample_size != 0 and self.ancient_times is None:
            raise ValueError("must specify ancient times if simulating ancients")

        row_data["Ne"] = 10000
        row_data["length"] = 5e6
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_func(params):
            seed = params[1]
            species = stdpopsim.get_species("HomSap")
            contig = species.get_contig("chr20", genetic_map="HapMapII_GRCh37")
            model = species.get_demographic_model("OutOfAfrica_3G09")
            # samples = model.get_samples(row_data["sample_size"], row_data["sample_size"], row_data["sample_size"])
            yri_samples = [
                msprime.Sample(population=0, time=0)
                for samp in range(row_data["sample_size_modern"] // 3)
            ]
            ceu_samples = [
                msprime.Sample(population=1, time=0)
                for samp in range(row_data["sample_size_modern"] // 3)
            ]
            chb_samples = [
                msprime.Sample(population=2, time=0)
                for samp in range(row_data["sample_size_modern"] // 3)
            ]
            ancient_samples = None
            if self.ancient_times == "empirical_age_distribution":
                ancient_sample_times = evaluation.sample_times(
                    row_data["sample_size_ancient"], constants.GENERATION_TIME
                )
                ancient_sample_times = np.array(ancient_sample_times, dtype=float)
                ancient_samples = [
                    msprime.Sample(population=1, time=time)
                    for samp, time in zip(
                        range(row_data["sample_size_ancient"]), ancient_sample_times
                    )
                ]
            elif self.ancient_times == "ooa_samples":
                ancient_samples = [
                    msprime.Sample(population=0, time=5650)
                    for samp in range(row_data["sample_size_ancient"])
                ]
            elif self.ancient_times == "amh_samples":
                ancient_samples = [
                    msprime.Sample(population=0, time=10000)
                    for samp in range(row_data["sample_size_ancient"])
                ]

            samples = yri_samples + ceu_samples + chb_samples + ancient_samples
            engine = stdpopsim.get_default_engine()
            ts = engine.simulate(model, contig, samples, seed=seed)
            if self.remove_ancient_mutations:
                ts = evaluation.remove_ancient_only_muts(ts)
            chr20_centromere = [25700000, 30400000]
            snippet_start = self.rng.randint(
                0, ts.get_sequence_length() - row_data["length"]
            )
            snippet_end = snippet_start + row_data["length"]
            # Don't allow snippets to include the centromere
            while (
                snippet_end > chr20_centromere[0] and snippet_end < chr20_centromere[1]
            ) or (
                snippet_start > chr20_centromere[0]
                and snippet_start < chr20_centromere[1]
            ):
                print("Rechoosing snippet")
                snippet_start = self.rng.randint(
                    0, ts.get_sequence_length() - row_data["length"]
                )
                snippet_end = snippet_start + row_data["length"]
            print(
                "Random Snippet Start:"
                + str(snippet_start)
                + " end: "
                + str(snippet_end)
            )
            self.snippet = [snippet_start, snippet_start + row_data["length"]]
            row_data["snippet"] = self.snippet
            return ts.keep_intervals(np.array([self.snippet]))

        genetic_map_func = self.get_genetic_map_chr20_snippet
        DataGeneration.setup(
            self, None, [None], simulate_func, genetic_map_func, row_data
        )

    def get_genetic_map_chr20_snippet(self, rowdata, filename):
        """
        For each chromosome 20 simulation, randomly select a region to run inference on
        """
        species = stdpopsim.get_species("HomSap")
        genetic_map = species.get_genetic_map("HapMapII_GRCh37")
        cm = genetic_map.get_chromosome_map("chr20")
        pos = np.array(cm.get_positions())
        snippet = np.where(np.logical_and(pos > self.snippet[0], pos < self.snippet[1]))
        snippet_pos = pos[snippet]
        snippet_rates = np.array(cm.get_rates())[snippet]
        map_distance = np.concatenate(
            [[0], (np.diff(snippet_pos) * snippet_rates[:-1]) / 1e6]
        )
        genetic_map_output = pd.DataFrame(
            data=np.stack([snippet_pos, snippet_rates, map_distance], axis=1),
            columns=["position", "COMBINED_rate.cM.Mb.", "Genetic_Map.cM."],
        )
        path_to_genetic_map = os.path.join(self.data_dir, filename + "_genetic_map.txt")
        genetic_map_output.to_csv(path_to_genetic_map, sep=" ", index=False)
        return genetic_map_output


class IterativeApproachNoAncients(NeutralSimulatedMutationAccuracy):
    name = "iterate_no_ancients"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred"]
        self.default_replicates = 2
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.make_vcf = False
        self.empirical_error = True

    def inference(self, row_data):
        """
        Iterative approach with only modern samples
        Compare with using true ages and with the true topology
        """
        index = row_data[0]
        row = row_data[1]
        path_to_file = os.path.join(self.data_dir, row["filename"])
        # Load the original simulation and the one with only modern samples
        sim = tskit.load(path_to_file + ".trees")

        # Name of output file with mutations ages
        path_to_file = os.path.join(self.data_dir, row["filename"])

        # Infer the ts based on sample data with keep_times
        keep_time_samples = tsinfer.formats.SampleData.from_tree_sequence(
            sim, use_sites_time=True
        )
        inferred_ts_keep_times = tsinfer.infer(keep_time_samples)
        inferred_ts_keep_times = inferred_ts_keep_times.simplify()
        # inferred_ts_keep_times.dump(path_to_file + ".tsinferred.keep_times.trees")
        dated_ts_keep_times = tsdate.date(
            inferred_ts_keep_times, row["Ne"], row["mut_rate"]
        )
        dated_ts_true_topo = tsdate.date(sim, row["Ne"], row["mut_rate"])

        # Infer the ts based on sample data
        samples = tsinfer.load(path_to_file + ".samples")

        def infer_date_iterate(samples):
            inferred_ts = tsinfer.infer(samples)
            inferred_ts = inferred_ts.simplify()
            dated_ts = tsdate.date(inferred_ts, row["Ne"], row["mut_rate"])
            sample_data_copy = samples.copy()
            sample_data_copy.sites_time[:] = tsdate.get_sites_time(dated_ts)
            sample_data_copy.finalise()
            iter_infer = tsinfer.infer(sample_data_copy).simplify()
            dated_ts_iter = tsdate.date(iter_infer, row["Ne"], row["mut_rate"])
            return inferred_ts, dated_ts, iter_infer, dated_ts_iter

        inferred_ts, dated_ts, iter_infer, dated_ts_iter = infer_date_iterate(samples)

        error_samples = tsinfer.load(path_to_file + ".error.samples")
        # Remove sites that have no derived allele
        # This can happen as a result of the error injection process
        derived = np.where(np.sum(error_samples.sites_genotypes[:], axis=1) != 0)[0]
        if len(derived) != sim.num_sites:
            no_deriv = np.where(np.sum(error_samples.sites_genotypes[:], axis=1) == 0)[
                0
            ]
            error_samples = error_samples.subset(sites=derived)
            sim_error_compatible = sim.delete_sites(no_deriv)
        assert sim_error_compatible.num_sites == error_samples.num_sites
        (
            error_inferred_ts,
            error_dated_ts,
            error_iter_infer,
            error_dated_ts_iter,
        ) = infer_date_iterate(error_samples)

        compare_df = evaluation.compare_mutation_msle_noancients(
            sim,
            inferred_ts,
            dated_ts,
            iter_infer,
            dated_ts_iter,
            inferred_ts_keep_times,
            dated_ts_keep_times,
            dated_ts_true_topo,
            sim_error_compatible,
            error_inferred_ts,
            error_dated_ts,
            error_iter_infer,
            error_dated_ts_iter,
        )

        return index, row, compare_df


class IterativeApproach_Chr20_no_ancients(Chr20SimulatedMutationAccuracy):
    name = "iterate_chr20_no_ancients"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred"]
        self.default_replicates = 50
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.make_vcf = False
        self.empirical_error = True

    def inference(self, row_data):
        """
        Iterative approach with only modern samples
        Compare with using true ages and with the true topology
        """
        index = row_data[0]
        row = row_data[1]
        path_to_file = os.path.join(self.data_dir, row["filename"])
        # Load the original simulation and the one with only modern samples
        sim = tskit.load(path_to_file + ".trees")

        # Name of output file with mutations ages
        path_to_file = os.path.join(self.data_dir, row["filename"])

        # Infer the ts based on sample data
        inferred_ts, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
            path_to_file + ".samples", sim.get_sequence_length()
        )
        inferred_ts = inferred_ts.simplify()
        inferred_ts.dump(path_to_file + ".tsinferred.trees")

        # Infer the ts based on sample data with keep_times
        tsinfer.formats.SampleData.from_tree_sequence(
            sim, path=path_to_file + ".keep_times.samples", use_sites_time=True
        )
        inferred_ts_keep_times, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
            path_to_file + ".keep_times.samples", sim.get_sequence_length()
        )
        inferred_ts_keep_times = inferred_ts_keep_times.simplify()
        inferred_ts_keep_times.dump(path_to_file + ".tsinferred.keep_times.trees")

        # tsdate runs
        tsdate_dates, _, _, _, _ = tsdate.get_dates(inferred_ts, 10000, 1e-8)
        tsdate_dates = tsdate_dates * 2 * row["Ne"]
        tsdate_keep_times, _, _, _, _ = tsdate.get_dates(
            inferred_ts_keep_times, 10000, 1e-8
        )
        tsdate_dates = tsdate_dates * 2 * row["Ne"]
        tsdate_true_topo, _, _, _, _ = tsdate.get_dates(sim, 10000, 1e-8)
        tsdate_true_topo = tsdate_true_topo * 2 * row["Ne"]

        # Constrain and redate
        sample_data = tsinfer.load(path_to_file + ".samples")
        sample_data_copy = sample_data.copy()
        sample_data_copy.sites_time[:] = np.array(
            tsdate_dates[inferred_ts.tables.mutations.node]
        )
        sample_data_copy.finalise()
        iter_infer = tsinfer.infer(sample_data_copy).simplify()
        iter_dates, _, _, _, _ = tsdate.get_dates(iter_infer, 10000, 1e-8)
        iter_dates = iter_dates * 2 * row["Ne"]
        compare_df = evaluation.compare_mutation_msle_noancients(
            sim,
            inferred_ts,
            tsdate_dates,
            iter_infer,
            iter_dates,
            inferred_ts_keep_times,
            tsdate_keep_times,
            tsdate_true_topo,
        )

        return index, row, compare_df


class SimulateVanillaAncient(DataGeneration):
    name = "simulate_vanilla_ancient"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = [
            "ancient_sample_size",
            "tsdateTime",
            "ConstrainedTime",
            "ConstrainedTSTime",
            "IterationTime",
            "SimulatedTopoTime",
        ]

        self.default_replicates = 1
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)

    def setup(self, num_processes=1):
        """
        Run Simulations
        """
        seeds = [self.rng.randint(1, 2 ** 31) for i in range(self.default_replicates)]
        if num_processes > 1:
            logging.info(
                "Setting up using multiprocessing ({} processes)".format(num_processes)
            )
            with multiprocessing.Pool(
                processes=num_processes, maxtasksperchild=10
            ) as pool:

                for index, row_data in tqdm(
                    enumerate(
                        pool.imap_unordered(self.setup_fn, iter(enumerate(seeds)))
                    ),
                    desc="Running Simulations",
                    total=len(seeds),
                ):
                    logging.info("Done with sim {}".format(index))
                    # Update dataframe with details of simulation
                    self.data = self.data.append(row_data, ignore_index=True)
        else:
            logging.info("Setting up using a single process")
            for seed in tqdm(
                enumerate(seeds),
                desc="Running Simulations",
                total=len(seeds),
            ):
                row_data = self.setup_fn(seed)
                logging.info("Done with sim {}".format(seed[0]))
                # Update dataframe with details of simulation
                self.data = self.data.append(row_data, ignore_index=True)

        # Save dataframe
        self.summarize()

    def setup_fn(self, seed):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["Ne"] = 10000
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8
        row_data["sample_size_modern"] = 100
        row_data["sample_size_ancient"] = 10
        row_data["length"] = 1e5
        index = seed[0]
        seed = seed[1]
        # randomly sample ancient times
        ancient_sample_times = evaluation.sample_times(
            row_data["sample_size_ancient"], constants.GENERATION_TIME
        )
        sim = evaluation.run_neutral_ancients(
            row_data["sample_size_modern"],
            row_data["sample_size_ancient"],
            ancient_sample_times,
            row_data["length"],
            row_data["Ne"],
            row_data["mut_rate"],
            row_data["rec_rate"],
            seed,
        )

        # Dump simulated tree
        filename = self.name + "_" + str(index)
        row_data["filename"] = filename
        row_data["replicate"] = index
        row_data["n_edges"] = sim.num_edges
        row_data["n_trees"] = sim.num_trees
        row_data["n_sites"] = sim.num_sites
        row_data["seed"] = seed

        # Save the simulated tree sequence
        sim.dump(os.path.join(self.data_dir, filename + ".trees"))

        # Remove mutations only present in ancients and mutations which become
        # fixed when ancients are removed
        sim = evaluation.remove_ancient_only_muts(sim)

        # Create tree sequence with only modern samples
        tables = sim.dump_tables()
        #        modern_samples = np.where(tables.nodes.time[sim.samples()] == 0)[0]
        #        modern_ts = sim.simplify(samples=modern_samples, keep_unary=True)
        #        assert np.all(np.isclose(tables.nodes.time[tables.mutations.node], modern_ts.tables.nodes.time[modern_ts.tables.mutations.node]))

        # Save the simulated tree sequence
        #        modern_ts.dump(os.path.join(self.data_dir, filename + ".modern.trees"))

        # Create sampledata file, with and without keeping times
        # Need to add the time of ancient samples from nodes
        sample_data = tsinfer.formats.SampleData.from_tree_sequence(
            sim,
            use_sites_time=False,
        )
        sample_data_indiv_times = sample_data.copy(
            path=os.path.join(self.data_dir, filename + ".samples")
        )
        sample_data_indiv_times.individuals_time[:] = np.array(
            tables.nodes.time[sim.samples()]
        )
        sample_data_indiv_times.finalise()

        tsinfer.formats.SampleData.from_tree_sequence(
            sim,
            path=os.path.join(self.data_dir, filename + ".keep_times.samples"),
            use_sites_time=True,
        )

        evaluation.add_error(sample_data)
        sample_data.copy(os.path.join(self.data_dir, filename + ".empirical_error"))
        sample_data.finalise()
        return row_data

    def inference(self, row_data):
        """
        Run four methods on the simulated data
        """
        index = row_data[0]
        row = row_data[1]
        path_to_file = os.path.join(self.data_dir, row["filename"])
        # Load the original simulation and the one with only modern samples
        sim = tskit.load(path_to_file + ".trees")
        tables = sim.dump_tables()
        modern_ts = tskit.load(path_to_file + ".modern.trees")

        # Load sampledata files
        samples = tsinfer.load(path_to_file + ".samples")
        error_samples = tsinfer.load(path_to_file + ".empirical_error.samples")

        # Name of output file with mutations ages
        path_to_file = os.path.join(self.data_dir, row["filename"])

        # Infer the ts based on modern sample data with keep_times
        inferred_ts_keep_times, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
            path_to_file + ".keep_times.samples", modern_ts.get_sequence_length()
        )
        inferred_ts_keep_times = inferred_ts_keep_times.simplify()
        inferred_ts_keep_times.dump(path_to_file + ".tsinferred.keep_times.trees")
        tsdate_keep_times, _, _, _, _ = tsdate.get_dates(
            inferred_ts_keep_times, row["Ne"], row["mut_rate"]
        )
        tsdate_keep_times = tsdate_keep_times * 2 * row["Ne"]
        tsdate_true_topo, _, _, _, _ = tsdate.get_dates(
            modern_ts, row["Ne"], row["mut_rate"]
        )
        tsdate_true_topo = tsdate_true_topo * 2 * row["Ne"]
        ancient_sample_sizes = [0, 1, 5, 10, 50, 100]
        modern_samples = np.where(tables.nodes.time[sim.samples()] == 0)[0]
        ancient_samples = np.where(tables.nodes.time[sim.samples()] != 0)[0]
        msle_master_df = pd.DataFrame(columns=self.columns)
        pearsonr_master_df = pd.DataFrame(columns=self.columns)
        spearmanr_master_df = pd.DataFrame(columns=self.columns)
        master_kc_df = pd.DataFrame(columns=self.columns)
        msle_master_df_err = pd.DataFrame(columns=self.columns)
        pearsonr_master_df_err = pd.DataFrame(columns=self.columns)
        spearmanr_master_df_err = pd.DataFrame(columns=self.columns)
        master_kc_df_err = pd.DataFrame(columns=self.columns)

        for ancient_sample_size in ancient_sample_sizes:
            sampledata_subset = samples.subset(
                individuals=ancient_samples[ancient_sample_size:]
            )
            sampledata_subset_err = error_samples.subset(
                individuals=ancient_samples[ancient_sample_size:]
            )

            # Constrain, reinfer, and redate
            (
                inferred_ts,
                tsdate_ages,
                constrained_ages,
                reinferred_ts,
                iter_dates,
                inferred_dated_ts,
                reinferred_dated_ts,
            ) = iteration.iter_infer(
                sampledata_subset,
                row["Ne"],
                row["mut_rate"],
                num_threads=1,
                output_fn=path_to_file
                + "."
                + str(ancient_sample_size)
                + "ancients.iteroutput",
                progress=False,
                return_trees=True,
            )
            (
                inferred_ts_err,
                tsdate_ages_err,
                constrained_ages_err,
                reinferred_ts_err,
                iter_dates_err,
                inferred_dated_ts_err,
                reinferred_dated_ts_err,
            ) = iteration.iter_infer(
                sampledata_subset_err,
                row["Ne"],
                row["mut_rate"],
                num_threads=1,
                output_fn=path_to_file
                + "."
                + str(ancient_sample_size)
                + "ancients.iteroutput",
                progress=False,
                return_trees=True,
            )
            (
                msle_compare_df,
                pearsonr_compare_df,
                spearmanr_compare_df,
            ) = evaluation.compare_mutations_iterative(
                ancient_sample_size,
                sim,
                modern_ts,
                inferred_ts,
                tsdate_ages,
                constrained_ages,
                reinferred_ts,
                iter_dates,
                inferred_ts_keep_times,
                tsdate_keep_times,
                tsdate_true_topo,
            )
            msle_master_df = pd.concat([msle_master_df, msle_compare_df], sort=False)
            pearsonr_master_df = pd.concat(
                [pearsonr_master_df, pearsonr_compare_df], sort=False
            )
            spearmanr_master_df = pd.concat(
                [spearmanr_master_df, spearmanr_compare_df], sort=False
            )
            (
                msle_compare_df_err,
                pearsonr_compare_df_err,
                spearmanr_compare_df_err,
            ) = evaluation.compare_mutations_iterative(
                ancient_sample_size,
                sim,
                modern_ts,
                inferred_ts_err,
                tsdate_ages_err,
                constrained_ages_err,
                reinferred_ts_err,
                iter_dates_err,
                inferred_ts_keep_times,
                tsdate_keep_times,
                tsdate_true_topo,
            )
            msle_master_df_err = pd.concat(
                [msle_master_df_err, msle_compare_df_err], sort=False
            )
            pearsonr_master_df_err = pd.concat(
                [pearsonr_master_df_err, pearsonr_compare_df_err], sort=False
            )
            spearmanr_master_df_err = pd.concat(
                [spearmanr_master_df_err, spearmanr_compare_df_err], sort=False
            )
            # Run KC distance comparisons
            inferred_ts_keeptimes_dated = evaluation.get_dated_ts(
                inferred_ts_keep_times, tsdate_keep_times, row["Ne"], 1e-6
            )
            true_topo_dated = evaluation.get_dated_ts(
                modern_ts, tsdate_true_topo, row["Ne"], 1e-6
            )
            kc_distances = evaluation.get_kc_distances(
                [
                    modern_ts.simplify(),
                    inferred_dated_ts.simplify(),
                    reinferred_dated_ts.simplify(samples=modern_samples),
                    inferred_ts_keeptimes_dated.simplify(),
                    true_topo_dated.simplify(),
                ],
                [
                    "simulated_ts",
                    "tsdateTime",
                    "IterationTime",
                    "tsinfer_keep_time",
                    "SimulatedTopoTime",
                ],
            )
            kc_distances["ancient_sample_size"] = ancient_sample_size
            kc_distances["lambda_param"] = [0, 1]
            master_kc_df = pd.concat([master_kc_df, kc_distances], sort=False)

            # Run KC distance comparisons with error
            kc_distances_err = evaluation.get_kc_distances(
                [
                    modern_ts.simplify(),
                    inferred_dated_ts_err.simplify(),
                    reinferred_dated_ts_err.simplify(samples=modern_samples),
                    inferred_ts_keeptimes_dated.simplify(),
                    true_topo_dated.simplify(),
                ],
                [
                    "simulated_ts",
                    "tsdateTime",
                    "IterationTime",
                    "tsinfer_keep_time",
                    "SimulatedTopoTime",
                ],
            )
            kc_distances_err["ancient_sample_size"] = ancient_sample_size
            kc_distances_err["lambda_param"] = [0, 1]
            master_kc_df_err = pd.concat(
                [master_kc_df_err, kc_distances_err], sort=False
            )

        return (
            index,
            row,
            msle_master_df,
            pearsonr_master_df,
            spearmanr_master_df,
            master_kc_df,
            msle_master_df_err,
            pearsonr_master_df_err,
            spearmanr_master_df_err,
            master_kc_df_err,
        )

    def run_multiprocessing(self, function, num_processes=1):
        """
        Run multiprocessing of inputted function a specified number of times
        """
        try:
            self.data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            logging.error("Must run with --setup flag first")

        mutation_output_fn = os.path.join(self.data_dir, self.name + "_mutations")
        kc_output_fn = os.path.join(self.data_dir, self.name + "_kc_distances")
        msle_master_df = pd.DataFrame(columns=self.columns)
        pearsonr_master_df = pd.DataFrame(columns=self.columns)
        spearmanr_master_df = pd.DataFrame(columns=self.columns)
        kc_distances_master_df = pd.DataFrame(columns=self.columns)
        msle_master_df_err = pd.DataFrame(columns=self.columns)
        pearsonr_master_df_err = pd.DataFrame(columns=self.columns)
        spearmanr_master_df_err = pd.DataFrame(columns=self.columns)
        kc_distances_master_df_err = pd.DataFrame(columns=self.columns)

        if num_processes > 1:
            logging.info(
                "Setting up using multiprocessing ({} processes)".format(num_processes)
            )
            with multiprocessing.Pool(
                processes=num_processes, maxtasksperchild=10
            ) as pool:

                for (
                    index,
                    row,
                    msle_mutations_df,
                    pearsonr_mutations_df,
                    spearmanr_mutations_df,
                    kc_distances_df,
                    msle_mutations_df_err,
                    pearsonr_mutations_df_err,
                    spearmanr_mutations_df_err,
                    kc_distances_df_err,
                ) in tqdm(
                    pool.imap_unordered(function, self.data.iterrows()),
                    desc="Inference Run",
                    total=self.data.shape[0],
                ):
                    logging.info("Running inference")
                    self.data.loc[index] = row
                    self.summarize()
                    msle_master_df = pd.concat(
                        [msle_master_df, msle_mutations_df], sort=False
                    )
                    pearsonr_master_df = pd.concat(
                        [pearsonr_master_df, pearsonr_mutations_df], sort=False
                    )
                    spearmanr_master_df = pd.concat(
                        [spearmanr_master_df, spearmanr_mutations_df], sort=False
                    )
                    kc_distances_master_df = pd.concat(
                        [kc_distances_master_df, kc_distances_df], sort=False
                    )
                    msle_master_df_err = pd.concat(
                        [msle_master_df_err, msle_mutations_df_err], sort=False
                    )
                    pearsonr_master_df_err = pd.concat(
                        [pearsonr_master_df_err, pearsonr_mutations_df_err], sort=False
                    )
                    spearmanr_master_df_err = pd.concat(
                        [spearmanr_master_df_err, spearmanr_mutations_df_err],
                        sort=False,
                    )
                    kc_distances_master_df_err = pd.concat(
                        [kc_distances_master_df_err, kc_distances_df_err], sort=False
                    )

        else:
            # When we have only one process it's easier to keep everything in the
            # same process for debugging.
            logging.info("Setting up using a single process")
            for (
                index,
                row,
                msle_mutations_df,
                pearsonr_mutations_df,
                spearmanr_mutations_df,
                kc_distances_df,
                msle_mutations_df_err,
                pearsonr_mutations_df_err,
                spearmanr_mutations_df_err,
                kc_distances_df_err,
            ) in map(function, self.data.iterrows()):
                logging.info("Running inference")
                self.data.loc[index] = row
                self.summarize()
                msle_master_df = pd.concat(
                    [msle_master_df, msle_mutations_df], sort=False
                )
                pearsonr_master_df = pd.concat(
                    [pearsonr_master_df, pearsonr_mutations_df], sort=False
                )
                spearmanr_master_df = pd.concat(
                    [spearmanr_master_df, spearmanr_mutations_df], sort=False
                )
                kc_distances_master_df = pd.concat(
                    [kc_distances_master_df, kc_distances_df], sort=False
                )
                msle_master_df_err = pd.concat(
                    [msle_master_df_err, msle_mutations_df_err], sort=False
                )
                pearsonr_master_df_err = pd.concat(
                    [pearsonr_master_df_err, pearsonr_mutations_df_err], sort=False
                )
                spearmanr_master_df_err = pd.concat(
                    [spearmanr_master_df_err, spearmanr_mutations_df_err], sort=False
                )
                kc_distances_master_df_err = pd.concat(
                    [kc_distances_master_df_err, kc_distances_df_err], sort=False
                )

        msle_master_df.to_csv(mutation_output_fn + ".msle.csv")
        pearsonr_master_df.to_csv(mutation_output_fn + ".pearsonr.csv")
        spearmanr_master_df.to_csv(mutation_output_fn + ".spearmanr.csv")
        kc_distances_master_df.to_csv(kc_output_fn + ".csv")
        msle_master_df_err.to_csv(mutation_output_fn + ".msle.empiricalerror.csv")
        pearsonr_master_df_err.to_csv(
            mutation_output_fn + ".pearsonr.empiricalerror.csv"
        )
        spearmanr_master_df_err.to_csv(
            mutation_output_fn + ".spearmanr.empiricalerror.csv"
        )
        kc_distances_master_df_err.to_csv(kc_output_fn + ".empiricalerror.csv")


class Chr20AncientIteration(Chr20SimulatedMutationAccuracy):
    """
    Data for Figure 1d: Ancient samples improve inference accuracy
    """

    name = "chr20_ancient_iteration"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred", "tsdate_iteration"]
        self.output_suffixes = [
            "_mutations.csv",
            "_msle.csv",
            "_spearman.csv",
            "_kc.csv",
        ]
        self.default_replicates = 20
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.empirical_error = False
        self.ancestral_state_error = False
        self.remove_ancient_mutations = True
        self.modern_sample_size = 1008
        self.ancient_sample_size = 50
        self.ancient_times = "empirical_age_distribution"

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["sample_size_modern"] = self.modern_sample_size
        row_data["sample_size_ancient"] = self.ancient_sample_size
        if self.ancient_sample_size != 0 and self.ancient_times is None:
            raise ValueError("must specify ancient times if simulating ancients")

        row_data["Ne"] = 10000
        row_data["length"] = 1e6
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_func(params):
            seed = params[1]
            species = stdpopsim.get_species("HomSap")
            contig = species.get_contig("chr20", length_multiplier=0.05)
            model = species.get_demographic_model("OutOfAfrica_3G09")
            yri_samples = [
                msprime.Sample(population=0, time=0)
                for samp in range(row_data["sample_size_modern"] // 3)
            ]
            ceu_samples = [
                msprime.Sample(population=1, time=0)
                for samp in range(row_data["sample_size_modern"] // 3)
            ]
            chb_samples = [
                msprime.Sample(population=2, time=0)
                for samp in range(row_data["sample_size_modern"] // 3)
            ]
            ancient_samples = []
            if self.ancient_times == "empirical_age_distribution":
                ancient_sample_times = evaluation.sample_times(
                    row_data["sample_size_ancient"], constants.GENERATION_TIME
                )
                ancient_sample_times = np.array(ancient_sample_times, dtype=float)
                for time in ancient_sample_times:
                    # If older than CEU-CHB split, the out of africa population is 1
                    if time >= 848:
                        ancient_samples.append(msprime.Sample(population=1, time=time))
                    else:
                        # If younger than CEU-CHB split, flip coin to decide CEU or CHB
                        ceu_chb = random.randint(1, 2)
                        ancient_samples.append(
                            msprime.Sample(population=ceu_chb, time=time)
                        )
            elif self.ancient_times == "ooa_samples":
                # Only population 0 exists at this time
                ancient_samples = [
                    msprime.Sample(population=0, time=5650)
                    for samp in range(row_data["sample_size_ancient"])
                ]
            elif self.ancient_times == "amh_samples":
                # Only population 0 exists at this time
                ancient_samples = [
                    msprime.Sample(population=0, time=10000)
                    for samp in range(row_data["sample_size_ancient"])
                ]

            samples = yri_samples + ceu_samples + chb_samples + ancient_samples
            engine = stdpopsim.get_default_engine()
            ts = engine.simulate(model, contig, samples, seed=seed)
            if self.remove_ancient_mutations:
                ts = evaluation.remove_ancient_only_muts(ts)
            return ts

        def genetic_map_func(row_data, filename):
            pass

        DataGeneration.setup(
            self, None, [None], simulate_func, genetic_map_func, row_data
        )

    def inference(self, row_data):
        index = row_data[0]
        row = row_data[1]

        path_to_file = os.path.join(self.data_dir, row["filename"])
        # Load the original simulation
        sim = tskit.load(path_to_file + ".trees")
        samples = tsinfer.load(path_to_file + ".samples")
        assert samples.num_sites == sim.num_sites

        modern_samples = samples.subset(
            individuals=np.arange(0, row["sample_size_modern"])
        )
        inferred_ts = tsinfer.infer(modern_samples)
        inferred_ts = tsdate.preprocess_ts(inferred_ts, filter_sites=False)
        dated = tsdate.date(inferred_ts, row["Ne"], row["mut_rate"])
        assert dated.num_sites == modern_samples.num_sites

        # Iterate with only modern samples
        sites_time = tsdate.sites_time_from_ts(dated)
        dated_samples = tsdate.add_sampledata_times(modern_samples, sites_time)
        iter_inferred_ts = tsinfer.infer(dated_samples, path_compression=False)
        iter_inferred_ts = tsdate.preprocess_ts(iter_inferred_ts)
        iter_dated = tsdate.date(
            iter_inferred_ts.simplify(), row["Ne"], row["mut_rate"]
        )

        modern_sim = sim.simplify(
            samples=np.arange(0, row["sample_size_modern"]).astype("int32")
        )
        assert sim.num_sites == modern_sim.num_sites
        modern_samples_keeptimes = tsinfer.formats.SampleData.from_tree_sequence(
            modern_sim, use_sites_time=True
        )
        inferred_modern = tsinfer.infer(modern_samples_keeptimes)
        inferred_modern_dated = tsdate.date(
            inferred_modern.simplify(), row["Ne"], row["mut_rate"]
        )
        assert np.array_equal(
            sim.tables.sites.position, inferred_modern_dated.tables.sites.position
        )

        # ancient_sample_sizes = [1, 5, 10, 20, 40]
        ancient_sample_sizes = [1, 5, 10]
        tables = sim.tables
        iter_ts_ancients = []
        iter_ts_inferred = []
        iter_ts_moderns_only = []
        for subset_size in ancient_sample_sizes:
            subsetted = samples.subset(
                individuals=np.arange(0, row["sample_size_modern"] + subset_size)
            )
            sites_time = tsdate.sites_time_from_ts(
                dated.simplify(
                    samples=np.arange(
                        0, row["sample_size_modern"] + subset_size
                    ).astype("int32")
                )
            )
            dated_samples = tsdate.add_sampledata_times(subsetted, sites_time)
            ancestors_reinferred = tsinfer.generate_ancestors(dated_samples)
            ancestors_reinferred_with_anc = ancestors_reinferred.insert_proxy_samples(
                dated_samples, allow_mutation=True
            )
            ancestors_ts_reinferred = tsinfer.match_ancestors(
                dated_samples, ancestors_reinferred_with_anc
            )  # , path_compression=False)
            reinferred = tsinfer.match_samples(
                modern_samples, ancestors_ts_reinferred, force_sample_times=True
            )
            reinferred = tsdate.preprocess_ts(reinferred)

            iter_ts_inferred.append(reinferred)
            reinferred_dated = tsdate.date(
                reinferred.simplify(), row["Ne"], row["mut_rate"]
            )
            iter_ts_ancients.append(reinferred_dated)
            reinferred_modern = reinferred_dated.simplify(
                samples=np.arange(0, row["sample_size_modern"]).astype("int32")
            )

            print("num samples:", reinferred_modern.num_samples)
            iter_ts_moderns_only.append(reinferred_modern)
        subset_names = [
            "Subset " + str(sample_size) for sample_size in ancient_sample_sizes
        ]
        mut_df = evaluation.compare_mutations(
            [sim, inferred_modern_dated, dated, iter_dated]
            + [ts for ts in iter_ts_ancients],
            [
                "simulated_ts",
                "tsdate_keep_times",
                "tsdate_inferred",
                "iter_dated_inferred",
            ]
            + subset_names,
        )
        msle_results = {}
        for col in mut_df.columns:
            comparable_muts = np.logical_and(
                mut_df["simulated_ts"] > 0, mut_df[col] > 0
            )
            msle_results[col] = mean_squared_log_error(
                mut_df["simulated_ts"][comparable_muts], mut_df[col][comparable_muts]
            )
        msle_df = pd.DataFrame(msle_results, index=[index])
        mut_df["Run"] = index

        spearman_results = {}
        for ts, name in zip(
            [inferred_modern, inferred_ts, iter_inferred_ts] + iter_ts_inferred,
            ["inferred_keep_times", "inferred", "reinferred"] + subset_names,
        ):
            spearman_results[name] = scipy.stats.spearmanr(
                tsdate.sites_time_from_ts(modern_sim, unconstrained=False),
                tsdate.sites_time_from_ts(ts, unconstrained=False),
            )[0]
        spearman_df = pd.DataFrame(spearman_results, index=[index])

        # Trim tree sequences to allow KC distance to work
        kc_ts_list = [
            modern_sim,
            inferred_modern_dated,
            dated,
            iter_dated,
        ] + iter_ts_moderns_only
        for i, ts in enumerate(kc_ts_list):
            ts_pos = ts.tables.sites.position
            kc_ts_list[i] = ts.keep_intervals(
                [[np.floor(ts_pos[0]), np.ceil(ts_pos[-1])]]
            ).trim()
        kc_df = evaluation.get_kc_distances(
            kc_ts_list,
            [
                "simulated_ts",
                "tsdate_keep_times",
                "tsdate_inferred",
                "iter_tsdate_inferred",
            ]
            + subset_names,
        )

        return_vals = {
            "mutations": mut_df,
            "msle": msle_df,
            "spearman": spearman_df,
            "kc": kc_df,
        }
        return index, row, return_vals


class Chr20AncientIterationOOA(Chr20AncientIteration):
    """
    Data for Figure 1d: OOA samples
    """

    name = "chr20_ancient_iteration_ooa"

    def __init__(self):
        Chr20AncientIteration.__init__(self)
        self.ancient_times = "ooa_samples"


class Chr20AncientIterationAMH(Chr20AncientIteration):
    """
    Data for Figure 1d: AMH samples
    """

    name = "chr20_ancient_iteration_amh"

    def __init__(self):
        Chr20AncientIteration.__init__(self)
        self.ancient_times = "amh_samples"


class EvaluatePrior(DataGeneration):
    """
    Data for Supplemental Figure 1: evaluation of tsdate prior
    """

    name = "evaluateprior"

    def setup(self):
        pass

    def evaluate_prior(self, ts, Ne, prior_distr):
        fixed_node_set = set(ts.samples())
        num_samples = len(fixed_node_set)

        span_data = SpansBySamples(ts, False)
        base_priors = ConditionalCoalescentTimes(None, prior_distr)
        base_priors.add(len(fixed_node_set), False)
        mixture_prior = base_priors.get_mixture_prior_params(span_data)
        confidence_intervals = np.zeros((ts.num_nodes - ts.num_samples, 4))

        if prior_distr == "lognorm":
            lognorm_func = scipy.stats.lognorm
            for node in np.arange(num_samples, ts.num_nodes):
                confidence_intervals[node - num_samples, 0] = np.sum(
                    span_data.get_weights(node)[num_samples]["descendant_tips"]
                    * span_data.get_weights(node)[num_samples]["weight"]
                )
                confidence_intervals[node - num_samples, 1] = (
                    2
                    * Ne
                    * lognorm_func.mean(
                        s=np.sqrt(mixture_prior[node, 1]),
                        scale=np.exp(mixture_prior[node, 0]),
                    )
                )
                confidence_intervals[node - num_samples, 2:4] = (
                    2
                    * Ne
                    * lognorm_func.ppf(
                        [0.025, 0.975],
                        s=np.sqrt(mixture_prior[node, 1]),
                        scale=np.exp(mixture_prior[node, 0]),
                    )
                )
        elif prior_distr == "gamma":
            gamma_func = scipy.stats.gamma
            for node in np.arange(ts.num_samples, ts.num_nodes):
                confidence_intervals[node - num_samples, 0] = np.sum(
                    span_data.get_weights(node)[ts.num_samples]["descendant_tips"]
                    * span_data.get_weights(node)[ts.num_samples]["weight"]
                )
                confidence_intervals[node - num_samples, 1] = (
                    2
                    * Ne
                    * gamma_func.mean(
                        mixture_prior[node, 0], scale=1 / mixture_prior[node, 1]
                    )
                )
                confidence_intervals[node - num_samples, 2:4] = (
                    2
                    * Ne
                    * gamma_func.ppf(
                        [0.025, 0.975],
                        mixture_prior[node, 0],
                        scale=1 / mixture_prior[node, 1],
                    )
                )
        return confidence_intervals

    def run_multiprocessing(self, inference_func, num_processes=1):
        inference_func()

    def inference(self):
        all_results = {
            i: {
                i: []
                for i in [
                    "in_range",
                    "expectations",
                    "real_ages",
                    "ts_size",
                    "upper_bound",
                    "lower_bound",
                    "num_tips",
                ]
            }
            for i in ["Lognormal_0", "Lognormal_1e-8", "Gamma_0", "Gamma_1e-8"]
        }

        for prior, (prior_distr, rec_rate) in tqdm(
            zip(
                all_results.keys(),
                [("lognorm", 0), ("lognorm", 1e-8), ("gamma", 0), ("gamma", 1e-8)],
            ),
            desc="Evaluating Priors",
            total=4,
        ):
            for i in range(1, 11):
                Ne = 10000
                ts = msprime.simulate(
                    sample_size=1000,
                    length=5e5,
                    Ne=Ne,
                    mutation_rate=1e-8,
                    recombination_rate=rec_rate,
                    random_seed=i,
                )

                confidence_intervals = self.evaluate_prior(ts, Ne, prior_distr)
                all_results[prior]["in_range"].append(
                    np.sum(
                        np.logical_and(
                            ts.tables.nodes.time[ts.num_samples :]
                            < confidence_intervals[:, 3],
                            ts.tables.nodes.time[ts.num_samples :]
                            > confidence_intervals[:, 2],
                        )
                    )
                )
                all_results[prior]["lower_bound"].append(confidence_intervals[:, 2])
                all_results[prior]["upper_bound"].append(confidence_intervals[:, 3])
                all_results[prior]["expectations"].append(confidence_intervals[:, 1])
                all_results[prior]["num_tips"].append(confidence_intervals[:, 0])
                all_results[prior]["real_ages"].append(
                    ts.tables.nodes.time[ts.num_samples :]
                )
                all_results[prior]["ts_size"].append(ts.num_nodes - ts.num_samples)
        pickle.dump(all_results, open("simulated-data/" + self.name + ".csv", "wb"))


class TsdateAccuracy(DataGeneration):
    """
    Generate data for Supplementary Figure 4: evaluating tsdate's accuracy at various mutation rates
    """

    name = "tsdate_accuracy"

    def setup(self):
        pass

    def inference(self):
        parameters_arr = [1e-9, 1e-8, 1e-7]

        Ne = 10000

        simulated = []
        io = []
        inferred_io = []
        maximized = []
        inferred_max = []

        io_kc = []
        max_kc = []
        inferred_io_kc = []
        inferred_max_kc = []

        mutation_rate = 1e-8
        recombination_rate = 1e-8

        random_seeds = range(1, 11)

        for index, param in tqdm(
            enumerate(parameters_arr),
            desc="Testing tsdate accuracy",
            total=len(parameters_arr),
        ):
            simulated_mut_ages = pd.DataFrame(columns=["Simulated Age", "Node"])
            io_mut_ages = pd.DataFrame(columns=["IO Age", "Node"])
            inferred_io_mut_ages = pd.DataFrame(columns=["IO Age", "Node"])
            max_mut_ages = pd.DataFrame(columns=["Max Age", "Node"])
            inferred_max_mut_ages = pd.DataFrame(columns=["Max Age", "Node"])

            cur_io_kc = []
            inferred_cur_io_kc = []
            cur_max_kc = []
            inferred_cur_max_kc = []
            for random_seed in random_seeds:
                ts = msprime.simulate(
                    sample_size=500,
                    Ne=Ne,
                    length=1e6,
                    mutation_rate=mutation_rate,
                    recombination_rate=recombination_rate,
                    random_seed=random_seed,
                )

                mutated_ts = msprime.mutate(ts, rate=param, random_seed=random_seed)
                sample_data = tsinfer.formats.SampleData.from_tree_sequence(
                    mutated_ts, use_sites_time=False
                )
                inferred_ts = tsinfer.infer(sample_data).simplify()
                io_dated = tsdate.date(
                    mutated_ts, mutation_rate=param, Ne=Ne, method="inside_outside"
                )
                max_dated = tsdate.date(
                    mutated_ts, mutation_rate=param, Ne=Ne, method="maximization"
                )
                io_inferred_dated = tsdate.date(
                    inferred_ts, mutation_rate=param, Ne=Ne, method="inside_outside"
                )
                max_inferred_dated = tsdate.date(
                    inferred_ts, mutation_rate=param, Ne=Ne, method="maximization"
                )

                # Get Mut Ages
                simulated_mut_ages = pd.concat(
                    [
                        simulated_mut_ages,
                        utility.get_mut_pos_df(
                            mutated_ts, "Simulated Age", mutated_ts.tables.nodes.time
                        ),
                    ],
                    sort=False,
                )
                io_mut_ages = pd.concat(
                    [
                        io_mut_ages,
                        utility.get_mut_pos_df(
                            io_dated, "IO Age", io_dated.tables.nodes.time
                        ),
                    ],
                    sort=False,
                )
                max_mut_ages = pd.concat(
                    [
                        max_mut_ages,
                        utility.get_mut_pos_df(
                            max_dated, "Max Age", max_dated.tables.nodes.time
                        ),
                    ],
                    sort=False,
                )
                inferred_io_mut_ages = pd.concat(
                    [
                        inferred_io_mut_ages,
                        utility.get_mut_pos_df(
                            io_inferred_dated,
                            "IO Age",
                            io_inferred_dated.tables.nodes.time,
                        ),
                    ],
                    sort=False,
                )
                inferred_max_mut_ages = pd.concat(
                    [
                        inferred_max_mut_ages,
                        utility.get_mut_pos_df(
                            max_inferred_dated,
                            "Max Age",
                            max_inferred_dated.tables.nodes.time,
                        ),
                    ],
                    sort=False,
                )

                # Get KC Distances
                cur_io_kc.append(mutated_ts.kc_distance(io_dated, lambda_=1))
                inferred_cur_io_kc.append(
                    mutated_ts.kc_distance(io_inferred_dated, lambda_=1)
                )
                cur_max_kc.append(mutated_ts.kc_distance(max_dated, lambda_=1))
                inferred_cur_max_kc.append(
                    mutated_ts.kc_distance(max_inferred_dated, lambda_=1)
                )
            simulated.append(simulated_mut_ages)
            io.append(io_mut_ages)
            maximized.append(max_mut_ages)
            inferred_io.append(inferred_io_mut_ages)
            inferred_max.append(inferred_max_mut_ages)

            io_kc.append(np.mean(cur_io_kc))
            max_kc.append(np.mean(cur_max_kc))
            inferred_io_kc.append(np.mean(inferred_cur_io_kc))
            inferred_max_kc.append(np.mean(inferred_cur_max_kc))

        pickle.dump(
            [
                simulated,
                io,
                maximized,
                inferred_io,
                inferred_max,
                io_kc,
                max_kc,
                inferred_io_kc,
                inferred_max_kc,
            ],
            open(
                "simulated-data/" + self.name + ".mutation_ages.kc_distances.csv", "wb"
            ),
        )

    def run_multiprocessing(self, inference_func, num_processes=1):
        inference_func()


class TsdateChr20(NeutralSimulatedMutationAccuracy):
    """
    Generate data for Supplementary Figure 6: evaluating tsdate's accuracy on Simulated Chromosome 20
    """

    name = "tsdate_chr20_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.default_replicates = 1
        self.columns = [
            "simulated_ts",
            "tsdate",
            "tsdate_inferred",
            "mismatch_inferred_dated",
            "iter_dated_ts",
        ]
        self.output_suffixes = [
            "_mutations.csv",
            "_error_mutations.csv",
            "_anc_error_mutations.csv",
            "_kc_distances.csv",
            "_error_kc_distances.csv",
            "_anc_error_kc_distances.csv",
        ]
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.empirical_error = True
        self.ancestral_state_error = True
        self.make_vcf = False

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["sample_size"] = 100
        row_data["mutation_rate"] = 1e-8
        row_data["Ne"] = 10000

        def simulate_func(params):
            seed = params[1]
            species = stdpopsim.get_species("HomSap")
            contig = species.get_contig("chr20", genetic_map="HapMapII_GRCh37")
            model = species.get_demographic_model("OutOfAfrica_3G09")
            samples = model.get_samples(
                row_data["sample_size"],
                row_data["sample_size"],
                row_data["sample_size"],
            )
            engine = stdpopsim.get_default_engine()
            ts = engine.simulate(model, contig, samples, seed=seed)
            return ts

        def genetic_map_func(row_data, filename):
            pass

        DataGeneration.setup(
            self, None, [None], simulate_func, genetic_map_func, row_data
        )

    def inference(self, row_data, num_threads=1, progress=False):
        index = row_data[0]
        row = row_data[1]

        filename = row["filename"]
        path_to_file = os.path.join(self.data_dir, filename)
        sim = tskit.load(path_to_file + ".trees")

        sample_data = tsinfer.load(path_to_file + ".samples")
        error_samples = tsinfer.load(path_to_file + ".error.samples")
        anc_error_samples = tsinfer.load(
            path_to_file + ".ancestral_state.error.samples"
        )

        print("Dating Simulated Tree Sequence")
        dated = tsdate.date(
            sim, mutation_rate=1e-8, Ne=int(row["Ne"]), progress=progress
        )
        dated.dump(path_to_file + ".dated.trees")

        def infer_all_methods(sample_data, name, inferred_dated):
            print("Inferring Tree Sequence")
            inferred_ts = tsinfer.infer(sample_data, num_threads=1).simplify(
                filter_sites=False
            )
            print("Dating Inferred Tree Sequence")
            inferred_dated = tsdate.date(
                inferred_ts,
                mutation_rate=row["mutation_rate"],
                Ne=int(row["Ne"]),
                progress=progress,
            )
            print("Inferring TS with Mismatch")
            _, mismatch_simplified_inferred_ts = evaluation.infer_with_mismatch(
                sample_data, "chr20", num_threads=1
            )
            print("Dating Mismatched TS")
            mismatch_inferred_dated = tsdate.date(
                mismatch_simplified_inferred_ts,
                mutation_rate=row["mutation_rate"],
                Ne=int(row["Ne"]),
                progress=progress,
            )

            copy = sample_data.copy()
            sites_time = tsdate.get_sites_time(inferred_dated)
            sites_time[sites_time > 1] = np.round(sites_time[sites_time > 1])
            copy.sites_time[:] = sites_time
            copy.finalise()
            print("Reinferring TS")
            _, iter_simplified_ts = evaluation.infer_with_mismatch(
                copy, "chr20", num_threads=1
            )
            print("Dating Reinferred TS")
            iter_dated_ts = tsdate.date(
                iter_simplified_ts,
                mutation_rate=row["mutation_rate"],
                Ne=int(row["Ne"]),
                num_threads=1,
                progress=progress,
            )

            inferred_dated.dump(path_to_file + name + ".inferred.dated.trees")
            mismatch_inferred_dated.dump(
                path_to_file + name + ".mismatch.inferred.dated.trees"
            )
            iter_dated_ts.dump(path_to_file + name + ".iter.mismatch.dated.trees")
            return inferred_dated, mismatch_inferred_dated, iter_dated_ts

        inferred_dated, mismatch_inferred_dated, iter_dated_ts = infer_all_methods(
            sample_data, "", inferred_dated
        )
        (
            error_inferred_dated,
            error_mismatch_inferred_dated,
            error_iter_dated_ts,
        ) = infer_all_methods(error_samples, ".error", error_inferred_dated)
        (
            anc_error_inferred_dated,
            anc_error_mismatch_inferred_dated,
            anc_error_iter_dated_ts,
        ) = infer_all_methods(anc_error_samples, ".anc_error", anc_error_inferred_dated)
        iter_dated_ts = infer_all_methods(sample_data, "", inferred_dated)
        error_iter_dated_ts = infer_all_methods(
            error_samples, ".error", error_inferred_dated
        )
        anc_error_iter_dated_ts = infer_all_methods(
            anc_error_samples, ".anc_error", anc_error_inferred_dated
        )

        ts_dict = {
            "sim": sim,
            "dated": dated,
            "inferred_dated": inferred_dated,
            "mismatch_inferred_dated": mismatch_inferred_dated,
            "iter_dated_ts": iter_dated_ts,
            "error_inferred_dated": error_inferred_dated,
            "error_mismatch_inferred_dated": error_mismatch_inferred_dated,
            "error_iter_dated_ts": error_iter_dated_ts,
            "anc_error_inferred_dated": anc_error_inferred_dated,
            "anc_error_mismatch_inferred_dated": anc_error_mismatch_inferred_dated,
            "anc_error_iter_dated_ts": anc_error_iter_dated_ts,
        }

        def get_ages(ts_dict):
            mut_ages = {}
            for name, cur_ts in ts_dict.items():
                mut_ages[name] = utility.get_mut_pos_df(
                    cur_ts, "Age", cur_ts.tables.nodes.time
                )["Age"].values
            return mut_ages

        no_error = {
            "sim": sim,
            "dated": dated,
            "inferred_dated": inferred_dated,
            "mismatch_inferred_dated": mismatch_inferred_dated,
            "iter_dated_ts": iter_dated_ts,
        }
        error = {
            "sim": sim,
            "error_inferred_dated": error_inferred_dated,
            "error_mismatch_inferred_dated": error_mismatch_inferred_dated,
            "error_iter_dated_ts": error_iter_dated_ts,
        }
        anc_error = {
            "sim": sim,
            "anc_error_inferred_dated": anc_error_inferred_dated,
            "anc_error_mismatch_inferred_dated": anc_error_mismatch_inferred_dated,
            "anc_error_iter_dated_ts": anc_error_iter_dated_ts,
        }

        mut_df = evaluation.compare_mutations(
            list(no_error.values()), list(no_error.keys())
        )
        error_mut_df = evaluation.compare_mutations(
            list(error.values()), list(error.keys())
        )
        anc_error_mut_df = evaluation.compare_mutations(
            list(anc_error.values()), list(anc_error.keys())
        )
        print("Starting KC No Error")
        kc_df = evaluation.get_kc_distances(
            list(no_error.values()), list(no_error.keys())
        )
        print("Starting KC Error")
        error_kc_df = evaluation.get_kc_distances(
            list(error.values()), list(error.keys())
        )
        print("Starting KC Ancestral State Error")
        anc_error_kc_df = evaluation.get_kc_distances(
            list(anc_error.values()), list(anc_error.keys())
        )
        return_vals = {
            "muts_noerr": mut_df,
            "muts_err": error_mut_df,
            "muts_anc_err": anc_error_mut_df,
            "kc_noerr": kc_df,
            "kc_err": error_kc_df,
            "kc_anc_err": anc_error_kc_df,
        }
        return index, row, return_vals


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
        "name",
        type=str,
        help="figure name",
        default="all",
        choices=sorted(list(name_map.keys()) + ["all"]),
    )
    parser.add_argument(
        "--setup", action="store_true", default=False, help="Run simulations"
    )
    parser.add_argument(
        "--inference", action="store_true", default=False, help="Run inference"
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=1,
        help="number of worker processes, e.g. 40",
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename="simulated-data/" + args.name + ".log",
        filemode="w",
        level=logging.DEBUG,
    )
    if args.name == "all":
        for _, fig in name_map.items():
            if fig in figures:
                fig = fig()
                if args.setup:
                    fig.setup()
                if args.inference:
                    fig.run_multiprocessing(fig.inference, num_processes=args.processes)

    else:
        fig = name_map[args.name]()
        if args.setup:
            # fig.setup(args.processes)
            fig.setup()
        if args.inference:
            fig.run_multiprocessing(fig.inference, args.processes)
    if not args.setup and not args.inference:
        raise ValueError("must run with --setup, --inference, or both.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simulation based evaluation for the unified genealogy paper
Pattern for generating data for each figure is:
 python3 src/run_evaluation.py PLOT_NAME --setup
 python3 src/run_evaluation.py PLOT_NAME --inference
"""
import sys

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

# import stdpopsim
import sys
import tsdate
from tsdate.prior import SpansBySamples, ConditionalCoalescentTimes

import evaluation
import constants
import utility
import error_generation

import stdpopsim


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
                    sim, use_sites_time=False, use_individuals_time=True
                )
                samples_indiv_times = samples.copy(
                    path=os.path.join(self.data_dir, filename + ".samples")
                )
                samples_indiv_times.finalise()

                # Add error to sampledata file
                if self.empirical_error:
                    error_samples = error_generation.add_errors(samples_indiv_times)
                    # Remove invariant sites
                    variant_sites = np.where(
                        np.sum(error_samples.sites_genotypes[:], axis=1) != 0
                    )[0]
                    print(
                        "Number of variant sites remaining after adding error: {}. Total sites: {}.".format(
                            len(variant_sites), error_samples.num_sites
                        )
                    )
                    error_samples = error_samples.subset(sites=variant_sites)
                    copy = error_samples.copy(
                        os.path.join(self.data_dir, filename + ".error.samples")
                    )
                    copy.finalise()

                # Add error to sampledata file
                if self.ancestral_state_error:
                    anc_error_samples = error_generation.add_errors(
                        samples_indiv_times, ancestral_allele_error=0.01
                    )
                    # Remove invariant sites
                    variant_sites = np.where(
                        np.sum(anc_error_samples.sites_genotypes[:], axis=1) != 0
                    )[0]
                    print(
                        "Number of variant sites remaining after adding error and ancestral state error: {}. Total sites: {}".format(
                            len(variant_sites), anc_error_samples.num_sites
                        )
                    )
                    anc_error_samples = anc_error_samples.subset(sites=variant_sites)

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


class NeutralSims(DataGeneration):
    """
    Template for mutation-based evaluation of various methods.
    Generates data for Supplemental Figure 3.
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
        self.tsinfer_iterate_mismatch = False
        self.tsinfer_iterate = True
        self.relate_reinfer = False
        self.geva_genetic_map = False

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["sample_size"] = 250
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
                    for index, (_, df) in enumerate(dfs.items()):
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
                for index, (_, df) in enumerate(dfs.items()):
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

        samples = tsinfer.load(path_to_file + ".samples")

        def run_all_inference(
            sim,
            dated_ts,
            samples,
            output_fn,
            tsinfer_mismatch=False,
            relate_reinfer=False,
        ):
            path_to_genetic_map = path_to_file + "_four_col_genetic_map.txt"
            compare_ts_dict = {"simulated_ts": sim, "tsdate": dated_ts}
            if self.tsinfer_mismatch:
                inferred_ts = evaluation.infer_with_mismatch(
                    samples,
                    path_to_genetic_map,
                    ma_mismatch=1,
                    ms_mismatch=1,
                    num_threads=1,
                )
                inferred_ts = tsdate.preprocess_ts(inferred_ts)
            else:
                inferred_ts = tsinfer.infer(samples).simplify()
            inferred_ts.dump(path_to_file + output_fn + ".tsinferred.trees")
            dated_inferred_ts = tsdate.date(
                inferred_ts, row["Ne"], row["mut_rate"], ignore_oldest_root=True
            )
            dated_inferred_ts.dump(
                path_to_file + output_fn + ".tsinferred.tsdated.trees"
            )
            compare_ts_dict["tsdate_inferred"] = dated_inferred_ts
            if self.tsinfer_iterate:
                sites_time = tsdate.sites_time_from_ts(dated_inferred_ts)
                dated_samples = tsdate.add_sampledata_times(samples, sites_time)
                if self.tsinfer_iterate_mismatch:
                    reinferred_ts = evaluation.infer_with_mismatch(
                        dated_samples,
                        path_to_genetic_map,
                        ma_mismatch=1,
                        ms_mismatch=1,
                        num_threads=1,
                    )
                    reinferred_ts = tsdate.preprocess_ts(reinferred_ts)
                else:
                    reinferred_ts = tsinfer.infer(dated_samples).simplify()
                reinferred_ts.dump(path_to_file + output_fn + ".iter.tsinferred.trees")
                redated_inferred_ts = tsdate.date(
                    reinferred_ts, row["Ne"], row["mut_rate"], ignore_oldest_root=True
                )
                redated_inferred_ts.dump(
                    path_to_file + output_fn + ".iter.tsinferred.tsdated.trees"
                )
                compare_ts_dict["tsdate_iterate"] = redated_inferred_ts

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
                relate_iter_ages.to_csv(path_to_file + output_fn + ".relate_reage.csv")
                relate_iter_ts.dump(path_to_file + output_fn + ".relate.iter.trees")
            else:
                relate_iter_ages = None
                relate_iter_ts = None
            if self.geva_genetic_map is True:
                geva_ages, geva_cpu, geva_memory = evaluation.run_geva(
                    path_to_file + output_fn,
                    row["Ne"],
                    row["mut_rate"],
                    genetic_map_path=path_to_genetic_map,
                )
            else:
                geva_ages, geva_cpu, geva_memory = evaluation.run_geva(
                    path_to_file + output_fn,
                    row["Ne"],
                    row["mut_rate"],
                    row["rec_rate"],
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
            # Trim tree sequences to allow KC distance to work
            compare_ts_dict["relate"] = relate_ts
            if self.relate_reinfer:
                compare_ts_dict["relate_iterate"] = relate_iter_ts
            kc_ts_dict = {}
            # Need to make the trees the same size for KC distances
            # Need to check for max/min positions because relate sometimes cuts off a
            # beginning or ending position
            max_start = 0
            min_end = sim.get_sequence_length()
            for ts in compare_ts_dict.values():
                ts_pos = ts.tables.sites.position
                if np.round(ts_pos[0]) > max_start:
                    max_start = np.round(ts_pos[0])
                if min_end > np.round(ts_pos[-1]):
                    min_end = np.round(ts_pos[-1])
            for ts_name, ts in compare_ts_dict.items():
                kc_ts_dict[ts_name] = ts.keep_intervals([[max_start, min_end]]).trim()
            kc_df = evaluation.get_kc_distances(
                list(kc_ts_dict.values()), list(kc_ts_dict.keys())
            )
            return mutation_df, kc_df

        mut_df, kc_df = run_all_inference(sim, dated_ts, samples, "")
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
        return index, row, return_vals


class TsdateNeutralSims(NeutralSims):
    """
    Generate data for Figure 1b: tsdate evaluation on neutral simulations
    """

    name = "tsdate_neutral_sims"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred"]
        self.output_suffixes = ["_mutations.csv"]
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.make_vcf = False

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

        dated_inferred_ts = tsdate.date(
            inferred_ts, row["Ne"], row["mut_rate"], ignore_oldest_root=True
        )
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


class Chr20Sims(NeutralSims):
    """
    Generate data for Extended Data Figure 2: evaluating accuracy of various methods on Simulated Chromosome 20
    using the out of africa model from stdpopsim
    """

    name = "chr20_sims"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred", "relate", "geva"]
        self.default_replicates = 30
        self.num_rows = self.default_replicates
        self.sim_cols = self.sim_cols + ["snippet"]
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.output_suffixes = [
            "_mutations.csv",
            "_error_mutations.csv",
            "_anc_error_mutations.csv",
            "_kc_distances.csv",
            "_error_kc_distances.csv",
            "_anc_error_kc_distances.csv",
        ]
        self.empirical_error = True
        self.ancestral_state_error = True
        self.tsinfer_mismatch = True
        self.tsinfer_iterate_mismatch = True
        self.tsinfer_iterate = True
        self.relate_reinfer = True
        self.modern_sample_size = 300
        self.ancient_sample_size = 0
        self.remove_ancient_mutations = False
        self.ancient_times = None
        self.geva_genetic_map = False

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
            chr20_centromere = [25700000, 30400000]
            snippet_start = 10000000
            snippet_end = snippet_start + row_data["length"]
            if snippet_end > chr20_centromere[0]:
                raise ValueError("Cannot include chr20 centromere")
            self.snippet = [snippet_start, snippet_start + row_data["length"]]
            row_data["snippet"] = self.snippet
            return ts.keep_intervals(np.array([self.snippet])).trim()

        genetic_map_func = self.get_genetic_map_chr20_snippet
        DataGeneration.setup(
            self, None, [None], simulate_func, genetic_map_func, row_data
        )

    def get_genetic_map_chr20_snippet(self, rowdata, filename):
        """
        For each chromosome 20 simulation, randomly select a region to run inference on
        """
        species = stdpopsim.get_species("HomSap")
        gmap = species.get_genetic_map("HapMapII_GRCh37")
        map_file = os.path.join(gmap.map_cache_dir, gmap.file_pattern.format(id="20"))
        hapmap = msprime.RateMap.read_hapmap(map_file)
        if "snippet" in rowdata:
            hapmap = hapmap.slice(start=self.snippet[0], end=self.snippet[1], trim=True)
        # tsinfer fails when recombination rate is 0, so we set it to a very small value
        # Whenever it is 0 (besides the first and last rows)
        rate = np.copy(hapmap.rate)
        rate[rate == 0] = 1e-20
        newmap = msprime.RateMap(hapmap.position, rate)
        rate = newmap.rate * 1e8
        # Need a final rate of 0
        rate = np.append(rate, 0)
        genetic_map_distance = newmap.get_cumulative_mass(hapmap.position) * 100
        # GEVA fails if first position is at 0
        position = newmap.position.astype(int)
        position[0] = 1
        genetic_map_output = pd.DataFrame(
            data=np.stack([position, rate, genetic_map_distance], axis=1),
            columns=["position", "COMBINED_rate.cM.Mb.", "Genetic_Map.cM."],
        )
        genetic_map_output = genetic_map_output.astype({"position": "int"})
        path_to_genetic_map = os.path.join(self.data_dir, filename + "_genetic_map.txt")
        genetic_map_output.to_csv(path_to_genetic_map, sep=" ", index=False)
        path_to_genetic_map = os.path.join(
            self.data_dir, filename + "_four_col_genetic_map.txt"
        )
        genetic_map_output.insert(0, "Chromosome", "chr20")
        genetic_map_output.to_csv(path_to_genetic_map, sep=" ", index=False)
        return genetic_map_output


class Chr20AncientIteration(Chr20Sims):
    """
    Data for Figure 1d: Ancient samples improve inference accuracy
    Note that we do not use mismatch in inference here
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
        self.default_replicates = 1
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.empirical_error = False
        self.ancestral_state_error = False
        self.remove_ancient_mutations = True
        self.modern_sample_size = 1008
        self.ancient_sample_size = 40
        self.ancient_times = "empirical_age_distribution"
        self.make_vcf = False

    def inference(self, row_data):
        index = row_data[0]
        row = row_data[1]

        path_to_file = os.path.join(self.data_dir, row["filename"])
        path_to_genetic_map = path_to_file + "_four_col_genetic_map.txt"

        # Load the original simulation
        sim = tskit.load(path_to_file + ".trees")
        samples = tsinfer.load(path_to_file + ".samples")
        assert samples.num_sites == sim.num_sites

        # Load the ratemap
        ratemap = msprime.RateMap.read_hapmap(
            path_to_genetic_map, sequence_length=samples.sequence_length + 1
        )

        modern_samples = samples.subset(
            individuals=np.arange(0, row["sample_size_modern"])
        )
        inferred_ts = tsinfer.infer(modern_samples, num_threads=10)
        #inferred_ts = evaluation.infer_with_mismatch(
        #    modern_samples,
        #    path_to_genetic_map,
        #    num_threads=10,
        #)
        inferred_ts = tsdate.preprocess_ts(inferred_ts, filter_sites=False)
        dated = tsdate.date(inferred_ts, row["Ne"], row["mut_rate"])
        dated.dump(path_to_file + ".modern_inferred_dated.trees")
        assert dated.num_sites == modern_samples.num_sites

        # Iterate with only modern samples
        sites_time = tsdate.sites_time_from_ts(dated)
        dated_samples = tsdate.add_sampledata_times(modern_samples, sites_time)
        #iter_inferred_ts = evaluation.infer_with_mismatch(
        #    dated_samples, path_to_genetic_map, num_threads=10
        #)
        iter_inferred_ts = tsinfer.infer(dated_samples, num_threads=10)
        iter_inferred_ts = tsdate.preprocess_ts(iter_inferred_ts)
        iter_dated = tsdate.date(
            iter_inferred_ts, row["Ne"], row["mut_rate"]
        )
        iter_dated.dump(path_to_file + ".iter_modern_inferred_dated.trees")

        modern_sim = sim.simplify(
            samples=np.arange(0, row["sample_size_modern"]).astype("int32")
        )
        assert sim.num_sites == modern_sim.num_sites
        modern_samples_keeptimes = tsinfer.formats.SampleData.from_tree_sequence(
            modern_sim, use_sites_time=True
        )
        inferred_modern = evaluation.infer_with_mismatch(
            modern_samples_keeptimes,
            path_to_genetic_map,
            ma_mismatch=1,
            ms_mismatch=1,
            num_threads=10,
        )
        inferred_modern_dated = tsdate.date(
            inferred_modern.simplify(), row["Ne"], row["mut_rate"]
        )
        assert np.array_equal(
            sim.tables.sites.position, inferred_modern_dated.tables.sites.position
        )

        ancient_sample_sizes = [1, 5, 10, 20, 40]
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
            ancestors_reinferred = tsinfer.generate_ancestors(
                dated_samples, num_threads=10
            )
            ancestors_reinferred_with_anc = ancestors_reinferred.insert_proxy_samples(
                dated_samples, allow_mutation=True
            )
            genetic_dists = tsinfer.Matcher.recombination_rate_to_dist(
                ratemap, ancestors_reinferred_with_anc.sites_position[:]
            )
            recombination = tsinfer.Matcher.recombination_dist_to_prob(genetic_dists)
            recombination[recombination == 0] = 1e-20
            mismatch = np.full(
                len(ancestors_reinferred_with_anc.sites_position[:]),
                tsinfer.Matcher.mismatch_ratio_to_prob(1, np.median(genetic_dists), 2),
            )
            ancestors_ts_reinferred = tsinfer.match_ancestors(
                dated_samples,
                ancestors_reinferred_with_anc,
                recombination=recombination,
                mismatch=mismatch,
                num_threads=10,
            )
            reinferred = tsinfer.match_samples(
                modern_samples,
                ancestors_ts_reinferred,
                recombination=recombination,
                mismatch=mismatch,
                force_sample_times=True,
                num_threads=10,
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
                "tsdate_iterate",
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
        for col in mut_df.columns:
            comparable_muts = np.logical_and(
                mut_df["simulated_ts"] > 0, mut_df[col] > 0
            )
            spearman_results[col] = scipy.stats.spearmanr(
                mut_df["simulated_ts"][comparable_muts], mut_df[col][comparable_muts]
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
                "tsdate_iterate",
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
    Generate data for Supplementary Figure 2: evaluating tsdate's accuracy at various mutation rates
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

        for param in tqdm(
            parameters_arr, desc="Testing tsdate accuracy", total=len(parameters_arr)
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
                samples = tsinfer.formats.SampleData.from_tree_sequence(
                    mutated_ts, use_sites_time=False
                )
                inferred_ts = tsinfer.infer(samples).simplify()
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
                            mutated_ts,
                            "Simulated Age",
                            mutated_ts.tables.nodes.time,
                            node_selection="msprime",
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


class TsdateChr20(Chr20Sims):
    """
    Generate data for Supplementary Figure 4: evaluating tsdate's accuracy on Simulated Chromosome 20
    """

    name = "tsdate_chr20_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.default_replicates = 1
        self.columns = [
            "simulated_ts",
            "tsdate",
            "tsdate_inferred",
            "tsdate_mismatch_inferred",
            "tsdate_iterate",
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
        row_data["mut_rate"] = 1e-8
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

        DataGeneration.setup(
            self,
            None,
            [None],
            simulate_func,
            self.get_genetic_map_chr20_snippet,
            row_data,
        )

    def inference(self, row_data, num_threads=40, progress=True):
        index = row_data[0]
        row = row_data[1]

        filename = row["filename"]
        path_to_file = os.path.join(self.data_dir, filename)
        sim = tskit.load(path_to_file + ".trees")

        samples = tsinfer.load(path_to_file + ".samples")
        error_samples = tsinfer.load(path_to_file + ".error.samples")
        anc_error_samples = tsinfer.load(
            path_to_file + ".ancestral_state.error.samples"
        )

        print("Dating Simulated Tree Sequence")
        dated = tsdate.date(
            sim, mutation_rate=1e-8, Ne=int(row["Ne"]), progress=progress
        )
        dated.dump(path_to_file + ".dated.trees")
        dated = tskit.load(path_to_file + ".dated.trees")

        def infer_all_methods(samples, name):
            print("Inferring Tree Sequence")
            inferred_ts = tsinfer.infer(
                samples, num_threads=10, progress_monitor=progress
            ).simplify(filter_sites=False)
            print("Dating Inferred Tree Sequence")
            tsdate_inferred = tsdate.date(
                inferred_ts,
                mutation_rate=row["mut_rate"],
                Ne=int(row["Ne"]),
                progress=progress,
                ignore_oldest_root=True,
            )
            tsdate_inferred.dump(path_to_file + name + ".inferred.dated.trees")
            tsdate_inferred = tskit.load(path_to_file + name + ".inferred.dated.trees")
            print("Inferring TS with Mismatch")
            path_to_genetic_map = path_to_file + "_four_col_genetic_map.txt"
            mismatch_inferred_ts = evaluation.infer_with_mismatch(
                samples,
                path_to_genetic_map,
                ma_mismatch=1,
                ms_mismatch=1,
                num_threads=10,
                progress_monitor=progress,
            ).simplify(filter_sites=False)
            mismatch_inferred_ts.dump(path_to_file + name + ".mismatch.inferred.trees")
            mismatch_inferred_ts = tskit.load(
                path_to_file + name + ".mismatch.inferred.trees"
            )
            print("Dating Mismatched TS")
            tsdate_mismatch_inferred = tsdate.date(
                mismatch_inferred_ts,
                mutation_rate=row["mut_rate"],
                Ne=int(row["Ne"]),
                progress=progress,
                ignore_oldest_root=True,
            )
            tsdate_mismatch_inferred.dump(
                path_to_file + name + ".mismatch.inferred.dated.trees"
            )
            tsdate_mismatch_inferred = tskit.load(
                path_to_file + name + ".mismatch.inferred.dated.trees"
            )

            sites_time = tsdate.sites_time_from_ts(tsdate_inferred)
            dated_samples = tsdate.add_sampledata_times(samples, sites_time)
            print("Reinferring TS")
            iter_simplified_ts = evaluation.infer_with_mismatch(
                dated_samples,
                path_to_genetic_map,
                ma_mismatch=1,
                ms_mismatch=1,
                num_threads=10,
                progress_monitor=progress,
            ).simplify(filter_sites=False)
            iter_simplified_ts.dump(
                path_to_file + name + ".iter.mismatch.inferred.trees"
            )
            iter_simplified_ts = tskit.load(
                path_to_file + name + ".iter.mismatch.inferred.trees"
            )

            print("Dating Reinferred TS")
            iter_dated_ts = tsdate.date(
                iter_simplified_ts,
                mutation_rate=row["mut_rate"],
                Ne=int(row["Ne"]),
                num_threads=10,
                progress=progress,
                ignore_oldest_root=True,
            )
            iter_dated_ts.dump(
                path_to_file + name + ".iter.mismatch.inferred.dated.trees"
            )
            iter_dated_ts = tskit.load(
                path_to_file + name + ".iter.mismatch.inferred.dated.trees"
            )

            sites_time = tsdate.sites_time_from_ts(tsdate_mismatch_inferred)
            dated_samples = tsdate.add_sampledata_times(samples, sites_time)
            print("Reinferring TS from Mismatch Estimation")
            mismatch_iter_simplified_ts = evaluation.infer_with_mismatch(
                dated_samples,
                path_to_genetic_map,
                ma_mismatch=1,
                ms_mismatch=1,
                num_threads=10,
                progress_monitor=progress,
            ).simplify(filter_sites=False)
            mismatch_iter_simplified_ts.dump(
                path_to_file + name + ".iterfrommismatch.mismatch.inferred.trees"
            )
            print("Dating Reinferred TS from Mismatch")
            mismatch_iter_dated_ts = tsdate.date(
                mismatch_iter_simplified_ts,
                mutation_rate=row["mut_rate"],
                Ne=int(row["Ne"]),
                num_threads=10,
                progress=progress,
                ignore_oldest_root=True,
            )
            mismatch_iter_dated_ts.dump(
                path_to_file + name + ".iterfrommismatch.mismatch.inferred.dated.trees"
            )
            mismatch_iter_dated_ts = tskit.load(
                path_to_file + name + ".iterfrommismatch.mismatch.inferred.dated.trees"
            )

            return (
                tsdate_inferred,
                tsdate_mismatch_inferred,
                iter_dated_ts,
                mismatch_iter_dated_ts,
            )

        (
            tsdate_inferred,
            tsdate_mismatch_inferred,
            iter_dated_ts,
            mismatch_iter_dated_ts,
        ) = infer_all_methods(samples, "")
        (
            error_tsdate_inferred,
            error_tsdate_mismatch_inferred,
            error_iter_dated_ts,
            error_mismatch_iter_dated_ts,
        ) = infer_all_methods(error_samples, ".error")
        (
            anc_error_tsdate_inferred,
            anc_error_tsdate_mismatch_inferred,
            anc_error_iter_dated_ts,
            anc_error_mismatch_iter_dated_ts,
        ) = infer_all_methods(anc_error_samples, ".anc_error")

        no_error = {
            "simulated_ts": sim,
            "tsdate": dated,
            "tsdate_inferred": tsdate_inferred,
            "tsdate_mismatch_inferred": tsdate_mismatch_inferred,
            "tsdate_iterate": iter_dated_ts,
            "tsdate_iterate_frommismatch": mismatch_iter_dated_ts,
        }
        error = {
            "simulated_ts": sim,
            "error_tsdate_inferred": error_tsdate_inferred,
            "error_tsdate_mismatch_inferred": error_tsdate_mismatch_inferred,
            "error_tsdate_iterate": error_iter_dated_ts,
            "error_tsdate_iterate_frommismatch": error_mismatch_iter_dated_ts,
        }
        anc_error = {
            "simulated_ts": sim,
            "anc_error_tsdate_inferred": anc_error_tsdate_inferred,
            "anc_error_tsdate_mismatch_inferred": anc_error_tsdate_mismatch_inferred,
            "anc_error_tsdate_iterate": anc_error_iter_dated_ts,
            "anc_error_tsdate_iterate_frommismatch": anc_error_mismatch_iter_dated_ts,
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


class ArchaicDescent(Chr20Sims):
    """
    Evaluate descent from sampled archaic individuals
    """

    name = "archaic_descent_chr20"

    def __init__(self):
        DataGeneration.__init__(self)
        self.default_replicates = 10
        self.columns = []

        self.output_suffixes = [
            "_denisovan_inf_arr.csv",
            "_denisovan_sim_intro.csv",
            "_denisovan_sim_shared.csv",
            "_vindija_inf_arr.csv",
            "_vindija_sim_intro.csv",
            "_vindija_sim_shared.csv",
            "_altai_inf_arr.csv",
            "_altai_sim_intro.csv",
            "_altai_sim_shared.csv",
            "_denisovan_results.csv",
            "_altai_results.csv",
            "_vindija_results.csv",
        ]
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.empirical_error = False
        self.ancestral_state_error = False
        self.make_vcf = False

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["sample_size"] = 200
        row_data["mut_rate"] = 1e-8
        row_data["Ne"] = 10000
        row_data["length"] = 15e6

        def simulate_func(params):
            seed = params[1]
            species = stdpopsim.get_species("HomSap")
            contig = species.get_contig(
                "chr20",
                genetic_map="HapMapII_GRCh37"
                # length_multiplier=0.3
            )
            model = evaluation.papuans_10j19()
            samples = model.get_samples(
                row_data["sample_size"],
                row_data["sample_size"],
                row_data["sample_size"],
                row_data["sample_size"],
                2,
                2,
                0,
                0,
                2,
            )
            engine = stdpopsim.get_default_engine()
            ts = engine.simulate(
                model, contig, samples, seed=seed, record_migrations=True
            )
            ts = evaluation.remove_ancient_only_muts(ts)
            tables = ts.dump_tables()
            tables.migrations.clear()
            ts_nomig = tables.tree_sequence()

            # return ts
            chr20_centromere = [25700000, 30400000]
            snippet_start = 10000000
            snippet_end = snippet_start + row_data["length"]
            if snippet_end > chr20_centromere[0]:
                raise ValueError("Cannot include chr20 centromere")
            self.snippet = [snippet_start, snippet_end]
            row_data["snippet"] = self.snippet
            ts_nomig = ts_nomig.keep_intervals(
                np.array([self.snippet]), simplify=False
            ).trim()
            tables = ts_nomig.dump_tables()
            mig_tables = ts.dump_tables()
            migrations = mig_tables.migrations
            keep_migs = np.logical_not(
                np.logical_or(
                    migrations.right <= snippet_start, migrations.left >= snippet_end
                )
            )
            tables.migrations.clear()
            tables.migrations.set_columns(
                left=np.fmax(snippet_start, migrations.left[keep_migs]),
                right=np.fmin(snippet_end, migrations.right[keep_migs]),
                node=migrations.node[keep_migs],
                source=migrations.source[keep_migs],
                dest=migrations.dest[keep_migs],
                time=migrations.time[keep_migs],
            )
            leftmost = np.min(tables.migrations.left)
            tables.migrations.set_columns(
                left=tables.migrations.left - leftmost,
                right=tables.migrations.right - leftmost,
                node=tables.migrations.node,
                source=tables.migrations.source,
                dest=tables.migrations.dest,
                time=tables.migrations.time,
            )
            return tables.tree_sequence()

        DataGeneration.setup(
            self,
            None,
            [None],
            simulate_func,
            self.get_genetic_map_chr20_snippet,
            row_data,
        )

    def get_ref_sets(self, ts):
        reference_sets = {}
        times = ts.tables.nodes.time[ts.samples()]
        for index, time in enumerate(np.unique(times)):
            time_nodes = np.where(ts.tables.nodes.time[ts.samples()] == time)[0].astype(
                np.int32
            )
            reference_sets[time] = time_nodes
        return reference_sets

    def get_descent(self, ts, proxy_nodes):
        """
        Get genomic locations of descent from given ancestral nodes in 1Kb chunks
        Chunks are binary: 0 indicates no descent from proxy nodes, 1 indicates descent
        No differentation is given between descent from both proxy nodes or descent from
        only one.
        """

        # Array indicating where indivdual descends from ancient sample
        modern_samples = ts.samples()[ts.tables.nodes.time[ts.samples()] == 0]
        descendants_arr = np.zeros(
            (len(modern_samples), int(ts.get_sequence_length() / 1000)), dtype=int
        )
        for focal_node in proxy_nodes:
            for tree in ts.trees():
                parent = tree.parent(focal_node)
                parent_leaves = list(tree.get_leaves(parent))
                if np.any(np.isin(parent_leaves, modern_samples)):
                    for cur_node in parent_leaves:
                        if cur_node in modern_samples:
                            descendants_arr[
                                cur_node,
                                int(np.floor(tree.interval[0] / 1000)) : int(
                                    np.ceil(tree.interval[1] / 1000)
                                ),
                            ] = 1
        return descendants_arr.astype(int)

    def get_migrating_tracts(self, ts, pop, split_time):
        modern_samples = ts.samples()[ts.tables.nodes.time[ts.samples()] == 0]
        intr_arr = np.zeros(
            (len(modern_samples), int(ts.get_sequence_length() / 1000)), dtype=int
        )
        nodes = list()
        left = list()
        right = list()
        # Get all migrations from the given population more recently than the split time
        for migration in ts.migrations():
            if migration.dest == pop and migration.time < split_time:
                nodes.append(migration.node)
                left.append(migration.left)
                right.append(migration.right)
        migrations = np.array([nodes, left, right]).transpose().astype(int)
        cur_nodes = set()
        for tree, (interval, edges_out, edges_in) in tqdm(
            zip(ts.trees(), ts.edge_diffs())
        ):
            # Keep track of nodes in current tree
            if interval[0] == 0:
                for edge in edges_in:
                    cur_nodes.add(edge.child)
                    cur_nodes.add(edge.parent)
            else:
                for edge in edges_out:
                    cur_nodes.discard(edge.child)
                    cur_nodes.discard(edge.parent)
                for edge in edges_in:
                    cur_nodes.add(edge.child)
                    cur_nodes.add(edge.parent)
            # Iterate over migrations with nodes in this tree
            for migration in migrations[np.isin(migrations[:, 0], list(cur_nodes))]:
                m_node = migration[0]
                m_left = migration[1]
                m_right = migration[2]
                leaves = list(tree.get_leaves(m_node))
                for leaf in leaves:
                    # Check if migration interval overlaps tree interval
                    if interval[0] <= m_right and m_left <= interval[1]:
                        intr_arr[
                            leaf,
                            int(np.floor(interval[0] / 1000)) : int(
                                np.ceil(interval[1] / 1000)
                            ),
                        ] = 1
        return intr_arr

    def get_shared_ancestry(self, ts, target_nodes, split_time, exclude_nodes=[]):
        """
        Get genomic locations of common ancestors with given ancestral nodes more recently than
        split time. Coded in 1Kb chunks, at each chunk: 0 indicates no recent common ancestor
        from proxy nodes, 1 indicates common ancestor. No differentation is given
        between sharing ancestry with both proxy nodes or with only one.
        """

        # Array indicating where indivdual descends from ancient sample
        modern_samples = ts.samples()[ts.tables.nodes.time[ts.samples()] == 0]
        descendants_arr = np.zeros(
            (len(modern_samples), int(ts.get_sequence_length() / 1000)), dtype=int
        )
        for focal_node in target_nodes:
            for tree in ts.trees():
                parent = tree.parent(focal_node)
                while ts.node(parent).time < split_time:
                    parent_leaves = list(tree.get_leaves(parent))
                    if np.any(np.isin(parent_leaves, modern_samples)) and np.all(
                        np.logical_not(np.isin(parent_leaves, exclude_nodes))
                    ):
                        for cur_node in parent_leaves:
                            if cur_node in modern_samples:
                                descendants_arr[
                                    cur_node,
                                    int(np.floor(tree.interval[0] / 1000)) : int(
                                        np.ceil(tree.interval[1] / 1000)
                                    ),
                                ] = 1
                    parent = tree.parent(parent)
        return descendants_arr.astype(int)

    def get_precision_recall_curve(
        ts, inferred, intro_arr, focal_nodes, exclude_nodes, step
    ):
        assert len(focal_nodes) == 2
        assert ts.node(focal_nodes[0]).time == ts.node(focal_nodes[1]).time
        precisions = list()
        recalls = list()
        for i in range(int(ts.node(focal_nodes[0]).time) + 1, top, step):
            desc = find_introgression(inferred, focal_nodes, i, exclude_nodes)
            precision, recall = accuracies(desc, intro_arr)
            precisions.append(precision)
            recalls.append(recall)
        return precisions, recalls

    def precision_recall(self, truth_arr, inferred_arr):
        # True positives: descent and introgression both detected
        tp = np.sum(np.logical_and(inferred_arr == 1, truth_arr == 1))
        # False positives: descent but no introgression
        fp = np.sum(np.logical_and(inferred_arr == 1, truth_arr == 0))
        # True negatives: no descent and no introgression detected
        tn = np.sum(np.logical_and(inferred_arr == 0, truth_arr == 0))
        # False negatives: no descent and introgression
        fn = np.sum(np.logical_and(inferred_arr == 0, truth_arr == 1))
        return (tp / (tp + fp)), (tp / (tp + fn))

    def get_precision_recall_curve(
        self,
        ts,
        inferred_ts,
        intro_arr,
        focal_nodes,
        num,
        top,
        results,
        suffix,
        exclude_nodes=[],
    ):
        assert len(focal_nodes) == 2
        assert ts.node(focal_nodes[0]).time == ts.node(focal_nodes[1]).time
        for i in np.linspace(int(ts.node(focal_nodes[0]).time) + 1, top, num):
            desc = self.get_shared_ancestry(inferred_ts, focal_nodes, i, exclude_nodes)
            precision, recall = self.precision_recall(intro_arr, desc)
            print(i, precision, recall)
            results["precision_" + suffix + "_" + str(i)] = precision
            results["recall_" + suffix + "_" + str(i)] = recall
        return results

    def eval_desc(
        self, inferred_ts, ts, intro_arr, ref_set, split_time, exclude_nodes=[]
    ):
        """
        Evaluate precision and recall for an archaic individual
        """
        # Create results dictionary with precision, recall results
        results = {}

        # Get shared ancestry from simulated tree sequence
        shared_arr = self.get_shared_ancestry(ts, ref_set, split_time, exclude_nodes)

        # Overlap of shared ancestry and introgressed material
        precision_shared_intro, recall_shared_intro = self.precision_recall(
            intro_arr, shared_arr
        )
        assert precision_shared_intro == 1.0
        results["neaden_shared_intro_recall"] = recall_shared_intro

        # Get descent and shared ancestry from inferred tree sequence
        desc_arr = self.get_descent(inferred_ts, ref_set)
        shared_inf_arr = self.get_shared_ancestry(
            inferred_ts, ref_set, split_time, exclude_nodes
        )

        # Determine precision-recall curve
        num = 10  # number of timepoint thresholds between age of sample and split time
        results = self.get_precision_recall_curve(
            ts, inferred_ts, shared_arr, ref_set, num, split_time, results, "shared"
        )
        results = self.get_precision_recall_curve(
            ts,
            inferred_ts,
            intro_arr,
            ref_set,
            num,
            split_time,
            results,
            "introgressed",
        )

        for truth_arr, truth_suffix in zip(
            [intro_arr, shared_arr], ["introgressed", "shared"]
        ):
            for inf_arr, inf_suffix in zip(
                [desc_arr, shared_inf_arr], ["descent_inf", "intro_inf"]
            ):
                precision, recall = self.precision_recall(truth_arr, inf_arr)
                # Calculate rates
                results["precision_" + truth_suffix + "_" + inf_suffix] = precision
                results["recall_" + truth_suffix + "_" + inf_suffix] = recall
                results["sim_total_" + truth_suffix + "_" + inf_suffix] = np.sum(
                    truth_arr
                )
                results["inf_total_" + truth_suffix + "_" + inf_suffix] = np.sum(
                    inf_arr
                )
        return (
            pd.DataFrame(desc_arr),
            pd.DataFrame(intro_arr),
            pd.DataFrame(shared_arr),
            pd.DataFrame([results]),
        )

    def inference(self, row_data, num_threads=40, progress=True):
        index = row_data[0]
        row = row_data[1]

        filename = row["filename"]
        path_to_file = os.path.join(self.data_dir, filename)
        sim = tskit.load(path_to_file + ".trees")

        no_error_samples = tsinfer.load(path_to_file + ".samples")
        anc_error_samples = tsinfer.load(
            path_to_file + ".ancestral_state.error.samples"
        )

        return_vals = {}

        samples = no_error_samples
        suffix = ""
        modern_samples = samples.subset(np.where(samples.individuals_time[:] == 0)[0])
        path_to_genetic_map = path_to_file + "_four_col_genetic_map.txt"
        recombination_map = msprime.RateMap.read_hapmap(path_to_genetic_map)
        afr_archaic_split = 20225
        T_Den_Nea_split = 15090

        # Run inference
        inferred_ts = evaluation.infer_with_mismatch(
            modern_samples,
            path_to_genetic_map,
            ma_mismatch=1,
            ms_mismatch=1,
            num_threads=8,
        )
        inferred_ts.dump(path_to_file + ".first_inferred" + suffix + ".trees")
        inferred_ts = tsdate.preprocess_ts(inferred_ts, filter_sites=False)
        dated_ts = tsdate.date(inferred_ts, Ne=row["Ne"], mutation_rate=row["mut_rate"])
        sites_time = tsdate.sites_time_from_ts(dated_ts)
        dated_samples = tsdate.add_sampledata_times(
            samples, sites_time
        )  # Get SampleData file with time estimates assigned to sites
        copy = dated_samples.copy()
        copy.sites_time[:] = np.round(dated_samples.sites_time[:])
        copy.finalise()
        dated_samples = copy
        ancestors = tsinfer.generate_ancestors(dated_samples, num_threads=8)
        ancestors_w_proxy = ancestors.insert_proxy_samples(
            dated_samples, allow_mutation=True
        )
        ancestors_ts = tsinfer.match_ancestors(
            dated_samples,
            ancestors_w_proxy,
            mismatch_ratio=1,
            recombination_rate=recombination_map,
            path_compression=False,
            num_threads=8,
        )
        inferred = tsinfer.match_samples(
            dated_samples,
            ancestors_ts,
            mismatch_ratio=1,
            recombination_rate=recombination_map,
            force_sample_times=True,
            num_threads=8,
        )
        inferred.dump(path_to_file + ".inferred" + suffix + ".trees")
        inferred = tskit.load(path_to_file + ".inferred" + suffix + ".trees")

        # Analyze Results
        reference_sets = self.get_ref_sets(sim)
        modern_samples = sim.samples()[sim.tables.nodes.time[sim.samples()] == 0]
        den_introgression_arr = np.zeros(
            (len(modern_samples), int(sim.get_sequence_length() / 1000)), dtype=int
        )
        # Find introgressed tracts using migration records
        for pop in [6, 7]:
            cur_intro_arr = self.get_migrating_tracts(sim, pop, T_Den_Nea_split)
            den_introgression_arr = np.logical_or(
                den_introgression_arr, cur_intro_arr
            ).astype(int)
        nea_introgression_arr = self.get_migrating_tracts(sim, 8, T_Den_Nea_split)
        (
            denisovan_inf_desc,
            denisovan_sim_introgression,
            denisovan_sim_shared,
            denisovan_results,
        ) = self.eval_desc(
            inferred,
            sim,
            den_introgression_arr,
            reference_sets[2203],
            T_Den_Nea_split,
            exclude_nodes=np.concatenate([reference_sets[3793], reference_sets[1725]]),
        )
        (
            altai_inf_desc,
            altai_sim_introgression,
            altai_sim_shared,
            altai_results,
        ) = self.eval_desc(
            inferred,
            sim,
            nea_introgression_arr,
            reference_sets[3793],
            T_Den_Nea_split,
            exclude_nodes=reference_sets[2203],
        )
        (
            vindija_inf_desc,
            vindija_sim_introgression,
            vindija_sim_shared,
            vindija_results,
        ) = self.eval_desc(
            inferred,
            sim,
            nea_introgression_arr,
            reference_sets[1725],
            T_Den_Nea_split,
            exclude_nodes=reference_sets[2203],
        )
        return_vals["denisovan_inf_arr" + suffix] = denisovan_inf_desc
        return_vals["denisovan_sim_intro" + suffix] = denisovan_sim_introgression
        return_vals["denisovan_sim_shared" + suffix] = denisovan_sim_shared
        return_vals["vindija_inf_arr" + suffix] = vindija_inf_desc
        return_vals["vindija_sim_intro" + suffix] = vindija_sim_introgression
        return_vals["vindija_sim_shared" + suffix] = vindija_sim_shared
        return_vals["altai_inf_arr" + suffix] = altai_inf_desc
        return_vals["altai_sim_intro" + suffix] = altai_sim_introgression
        return_vals["altai_sim_shared" + suffix] = altai_sim_shared
        return_vals["denisovan_results" + suffix] = denisovan_results
        return_vals["altai_results" + suffix] = altai_results
        return_vals["vindija_results" + suffix] = vindija_results
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
    if not args.setup and not args.inference:
        raise ValueError("must run with --setup, --inference, or both.")
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
            fig.setup()
        if args.inference:
            fig.run_multiprocessing(fig.inference, args.processes)


if __name__ == "__main__":
    main()

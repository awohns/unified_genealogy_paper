#!/usr/bin/env python3
"""
Generate data for plots in tsdate paper
 python3 src/generate_data.py PLOT_NAME
"""
import argparse

import logging
import os
import pandas as pd
import multiprocessing
import numpy as np
import random
import shutil
from tqdm import tqdm

import tskit
import tsinfer
import msprime
import stdpopsim

import evaluation
import constants
import iteration

import tsdate  # NOQA


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
                _ = tsinfer.formats.SampleData.from_tree_sequence(
                    sim,
                    path=os.path.join(self.data_dir, filename + ".samples"),
                    use_times=False,
                )

                # Add error to sampledata file
                if self.empirical_error:
                    evaluation.generate_samples(
                        sim,
                        os.path.join(self.data_dir, filename + ".error"),
                        seq_error=evaluation.empirical_error_probs,
                        empirical_seq_err_name="platinum_error",
                    )

                # Create VCF file
                if self.make_vcf:
                    with open(
                        os.path.join(self.data_dir, filename + ".vcf"), "w"
                    ) as vcf_file:
                        sim.write_vcf(vcf_file, ploidy=2, position_transform="legacy")
                    if self.empirical_error:
                        with open(
                            os.path.join(self.data_dir, filename + ".error.vcf"), "w"
                        ) as vcf_file:
                            sim.write_vcf(
                                vcf_file, ploidy=2, position_transform="legacy"
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


class CpuScalingSampleSize(DataGeneration):
    """
    Plot CPU times of tsdate, tsinfer, tsdate+tsinfer, Relate, and GEVA
    Run the following to occupt other threads: nice -n 15 stress -c 40
    """

    name = "cpu_scaling_samplesize"

    def __init__(self):
        DataGeneration.__init__(self)
        self.sample_sizes = [
            10,
            260,
            526,
            790,
            1052,
            1316,
            1578,
            1842,
            2106,
            2368,
            2632,
            2894,
            3158,
            3422,
            3684,
            3948,
            4210,
            4474,
            4736,
            5000,
        ]
        # self.sample_sizes = [10, 20, 40, 64, 100]
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
        # path_to_genetic_map = (
        #    row["filename"] + "_genetic_map.txt"
        # )
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
        #        _, geva_cpu, geva_memory = evaluation.run_geva(
        #            path_to_file, row["Ne"], row["mut_rate"], row["rec_rate"]
        #        )
        #        row["geva_cpu", "geva_memory"] = [geva_cpu, geva_memory]

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
        self.lengths = np.linspace(1e5, 1e7, 20, dtype=int)

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["Ne"] = 10000
        row_data["sample_size"] = 1000
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


class NeutralSimulatedMutationAccuracy(DataGeneration):
    name = "neutral_simulated_mutation_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred", "relate", "geva"]
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.empirical_error = True

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["sample_size"] = 1000
        row_data["Ne"] = 10000
        row_data["length"] = 1e5
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

    def run_multiprocessing(self, function, num_processes=1):
        """
        Run multiprocessing of inputted function a specified number of times
        """
        try:
            self.data = pd.read_csv(self.data_file)
        except FileNotFoundError:
            logging.error("Must run with --setup flag first")

        output_fn = os.path.join(self.data_dir, self.name + "_mutations.csv")
        master_df = pd.DataFrame(columns=self.columns)
        if num_processes > 1:
            logging.info(
                "Setting up using multiprocessing ({} processes)".format(num_processes)
            )
            with multiprocessing.Pool(
                processes=num_processes, maxtasksperchild=10
            ) as pool:

                for index, row, mutations_df in tqdm(
                    pool.imap_unordered(function, self.data.iterrows()),
                    desc="Inference Run",
                    total=self.data.shape[0],
                ):
                    self.data.loc[index] = row
                    self.summarize()
                    master_df = pd.concat([master_df, mutations_df])
        else:
            # When we have only one process it's easier to keep everything in the
            # same process for debugging.
            logging.info("Setting up using a single process")
            for index, row, mutations_df in tqdm(
                map(function, self.data.iterrows()), total=self.data.shape[0]
            ):
                logging.info("Running inference")
                self.data.loc[index] = row
                self.summarize()
                master_df = pd.concat([master_df, mutations_df])
        master_df.to_csv(output_fn)

    def inference(self, row_data):
        """
        Run four methods on the simulated data
        """
        index = row_data[0]
        row = row_data[1]
        path_to_file = os.path.join(self.data_dir, row["filename"])
        sim = tskit.load(path_to_file + ".trees")

        # Name of output file with mutations ages
        path_to_file = os.path.join(self.data_dir, row["filename"])
        sim = tskit.load(path_to_file + ".trees")

        dated_ts, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
            path_to_file + ".trees", row["Ne"], row["mut_rate"], 20, "inside_outside"
        )
        dated_ts.dump(path_to_file + ".tsdated.trees")

        inferred_ts, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
            path_to_file + ".samples", sim.get_sequence_length()
        )
        inferred_ts.simplify().dump(path_to_file + ".tsinferred.trees")

        dated_inferred_ts, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
            path_to_file + ".tsinferred.trees",
            row["Ne"],
            row["mut_rate"],
            50,
            "inside_outside",
        )
        dated_inferred_ts.dump(path_to_file + ".tsinferred.tsdated.trees")

        relate_dir = os.path.join(self.data_dir, "relate_" + row["filename"])
        path_to_genetic_map = path_to_file + "_genetic_map.txt"
        relate_ts, relate_age, relate_cpu, relate_memory = evaluation.run_relate(
            sim,
            path_to_file,
            row["mut_rate"],
            row["Ne"] * 2,
            path_to_genetic_map,
            relate_dir,
            "relate_run",
        )
        relate_age.to_csv(path_to_file + "relate_age")
        relate_age = pd.read_csv(path_to_file + "relate_age", sep=",")

        geva_ages, geva_cpu, geva_memory = evaluation.run_geva(
            path_to_file, row["Ne"], row["mut_rate"], row["rec_rate"]
        )
        geva_ages.to_csv(path_to_file + ".geva.csv")
        geva_ages = pd.read_csv(path_to_file + ".geva.csv", index_col="MarkerID")
        geva_positions = pd.read_csv(
            path_to_file + ".marker.txt", delimiter=" ", index_col="MarkerID"
        )

        compare_df = evaluation.compare_mutations(
            [sim, dated_ts, dated_inferred_ts],
            geva_ages=geva_ages,
            geva_positions=geva_positions,
            relate_ages=relate_age,
        )
        return index, row, compare_df


class TsdateNeutralSimulatedMutationAccuracy(NeutralSimulatedMutationAccuracy):
    name = "tsdate_neutral_simulated_mutation_accuracy"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = ["simulated_ts", "tsdate", "tsdate_inferred"]
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

        dated_ts, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
            path_to_file + ".trees", row["Ne"], row["mut_rate"], 20, "inside_outside"
        )
        dated_ts.dump(path_to_file + ".tsdated.trees")

        inferred_ts, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
            path_to_file + ".samples", sim.get_sequence_length()
        )
        inferred_ts.simplify().dump(path_to_file + ".tsinferred.trees")

        dated_inferred_ts, tsdate_cpu, tsdate_memory = evaluation.run_tsdate(
            path_to_file + ".tsinferred.trees",
            row["Ne"],
            row["mut_rate"],
            50,
            "inside_outside",
        )
        dated_inferred_ts.dump(path_to_file + ".tsinferred.tsdated.trees")

        compare_df = evaluation.compare_mutations([sim, dated_ts, dated_inferred_ts])
        return index, row, compare_df


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
        self.num_rows = self.default_replicates
        self.sim_cols = self.sim_cols + ["snippet"]
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)

    def setup(self):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["sample_size"] = 1000
        row_data["Ne"] = 10000
        row_data["length"] = 5e6
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8

        def simulate_func(params):
            seed = params[1]
            species = stdpopsim.get_species("HomSap")
            contig = species.get_contig("chr20", genetic_map="HapMapII_GRCh37")
            model = species.get_demographic_model("OutOfAfrica_3G09")
            samples = model.get_samples(row_data["sample_size"])
            engine = stdpopsim.get_default_engine()
            ts = engine.simulate(model, contig, samples, seed=seed)
            chr20_centromere = [25700000, 30400000]
            snippet_start = self.rng.randint(
                0, ts.get_sequence_length() - row_data["length"]
            )
            snippet_end = snippet_start + row_data["length"]
            while (
                snippet_end > chr20_centromere[0] and snippet_end < chr20_centromere[1]
            ) or (
                snippet_start > chr20_centromere[0]
                and snippet_start < chr20_centromere[1]
            ):
                snippet_start = self.rng.randint(
                    0, ts.get_sequence_length() - row_data["length"]
                )
                snippet_end = snippet_start + row_data["length"]
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
        self.default_replicates = 100
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)
        self.make_vcf = False

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
            sim, path=path_to_file + ".keep_times.samples", use_times=True
        )
        inferred_ts_keep_times, tsinfer_cpu, tsinfer_memory = evaluation.run_tsinfer(
            path_to_file + ".keep_times.samples", sim.get_sequence_length()
        )
        inferred_ts_keep_times = inferred_ts_keep_times.simplify()
        inferred_ts_keep_times.dump(path_to_file + ".tsinferred.keep_times.trees")

        # tsdate runs
        tsdate_dates, _, _, _, _ = tsdate.get_dates(
            inferred_ts, row["Ne"], row["mut_rate"]
        )
        tsdate_dates = 2 * row["Ne"] * tsdate_dates
        tsdate_keep_times, _, _, _, _ = tsdate.get_dates(
            inferred_ts_keep_times, row["Ne"], row["mut_rate"]
        )
        tsdate_keep_times = 2 * row["Ne"] * tsdate_keep_times
        tsdate_true_topo, _, _, _, _ = tsdate.get_dates(sim, row["Ne"], row["mut_rate"])
        tsdate_true_topo = 2 * row["Ne"] * tsdate_true_topo

        # Constrain and redate
        sample_data = tsinfer.load(path_to_file + ".samples")
        sample_data_copy = sample_data.copy()
        sample_data_copy.sites_time[:] = np.array(
            tsdate_dates[inferred_ts.tables.mutations.node]
        )
        sample_data_copy.finalise()
        iter_infer = tsinfer.infer(sample_data_copy).simplify()
        iter_dates, _, _, _, _ = tsdate.get_dates(
            iter_infer, row["Ne"], row["mut_rate"]
        )
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
            sim, path=path_to_file + ".keep_times.samples", use_times=True
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

        #        compare_df = evaluation.compare_mutation_msle_noancients(
        #            sim, tsdate_dates, iter_dates, tsdate_keep_times,
        #            tsdate_true_topo)

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

        self.default_replicates = 20
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

                for index, row_data in tqdm(enumerate(
                    pool.imap_unordered(self.setup_fn, iter(enumerate(seeds)))),
                    desc="Running Simulations",
                    total=len(seeds),
                ):
                    logging.info("Done with sim {}".format(index))
                    # Update dataframe with details of simulation
                    self.data = self.data.append(row_data, ignore_index=True)


        else:
            # When we have only one process it's easier to keep everything in the
            # same process for debugging.
            logging.info("Setting up using a single process")
            for index, row_data in tqdm(enumerate(
                pool.imap_unordered(self.setup_fn, iter(enumerate(seeds)))),
                desc="Running Simulations",
                total=len(seeds),
            ):
                logging.info("Done with sim {}".format(index))
                # Update dataframe with details of simulation
                self.data = self.data.append(row_data, ignore_index=True)


        # Save dataframe
        self.summarize()

    def setup_fn(self, seed):
        row_data = dict.fromkeys(self.sim_cols)
        row_data["Ne"] = 10000
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8
        row_data["sample_size_modern"] = 1000
        row_data["sample_size_ancient"] = 100
        row_data["length"] = 1e6
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

        # Remove ancient samples and fixed mutations
        tables = sim.dump_tables()
        modern_samples = np.where(tables.nodes.time[sim.samples()] == 0)[0]
        # modern_ts = evaluation.remove_ancients(sim)
        modern_ts = sim.simplify(samples=modern_samples)

        # Save the simulated tree sequence
        modern_ts.dump(os.path.join(self.data_dir, filename + ".modern.trees"))

        # Create sampledata file, with and without keeping times
        # Need to add the time of ancient samples from nodes
        sample_data = tsinfer.formats.SampleData.from_tree_sequence(
            sim, use_times=False,
        )
        sample_data_indiv_times = sample_data.copy(
            path=os.path.join(self.data_dir, filename + ".samples")
        )
        sample_data_indiv_times.individuals_time[:] = np.array(
            tables.nodes.time[sim.samples()]
        )
        sample_data_indiv_times.finalise()

        tsinfer.formats.SampleData.from_tree_sequence(
            modern_ts,
            path=os.path.join(self.data_dir, filename + ".keep_times.samples"),
            use_times=True,
        )

        evaluation.generate_samples(
            sim,
            os.path.join(self.data_dir, filename + ".empirical_error"),
            seq_error=evaluation.empirical_error_probs,
            empirical_seq_err_name="empirical_error",
        )
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
            sampledata_subset = samples.delete(
                samples=ancient_samples[ancient_sample_size:]
            )
            sampledata_subset_err = error_samples.delete(
                samples=ancient_samples[ancient_sample_size:]
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
            kc_distances = pd.DataFrame.from_dict(
                evaluation.get_kc_distances(
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
            )
            kc_distances["ancient_sample_size"] = ancient_sample_size
            kc_distances["lambda_param"] = [0, 1]
            master_kc_df = pd.concat([master_kc_df, kc_distances], sort=False)

            # Run KC distance comparisons with error
            kc_distances_err = pd.DataFrame.from_dict(
                evaluation.get_kc_distances(
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


class OOA_Chr20_ancient(SimulateVanillaAncient):
    name = "ooa_chr20"

    def __init__(self):
        DataGeneration.__init__(self)
        self.columns = [
            "ancient_sample_size",
            "simulated_ts",
            "tsdate",
            "tsdate_inferred",
            "tsdate_iteration",
            "tsdate_true_times",
            "tsdate_sim_topo",
        ]
        self.default_replicates = 1
        self.sim_cols = self.sim_cols
        self.num_rows = self.default_replicates
        self.data = pd.DataFrame(columns=self.sim_cols)
        self.rng = random.Random(self.default_seed)

    def setup(self):
        """
        Run Simulations
        """
        row_data = dict.fromkeys(self.sim_cols)
        row_data["Ne"] = 10000
        # row_data["sample_size"] = 500
        row_data["mut_rate"] = 1e-8
        row_data["rec_rate"] = 1e-8
        row_data["sample_size_modern"] = 300
        row_data["sample_size_ancient"] = 100
        row_data["length"] = 1e6

        seeds = [self.rng.randint(1, 2 ** 31) for i in range(self.default_replicates)]
        for index, seed in tqdm(enumerate(seeds), desc="Running Simulations"):
            # randomly sample ancient times
            ancient_sample_times = evaluation.sample_times(
                row_data["sample_size_ancient"], constants.GENERATION_TIME
            )
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
            ancient_sample_times = np.array(ancient_sample_times, dtype=float)
            ancient_samples = [
                msprime.Sample(population=0, time=time)
                for samp, time in zip(
                    range(row_data["sample_size_ancient"]), ancient_sample_times
                )
            ]
            samples = yri_samples + ceu_samples + chb_samples + ancient_samples
            sim = evaluation.run_chr20_ooa(
                samples,
                row_data["Ne"],
                row_data["length"],
                row_data["mut_rate"],
                row_data["rec_rate"],
                self.rng,
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

            # Remove ancient samples and fixed mutations
            modern_samples = np.where(sim.tables.nodes.time[sim.samples()] == 0)[0]
            modern_ts = evaluation.remove_ancients(sim)

            # Save the simulated tree sequence
            modern_ts.dump(os.path.join(self.data_dir, filename + ".modern.trees"))

            # Create sampledata file, with and without keeping times
            tsinfer.formats.SampleData.from_tree_sequence(
                modern_ts,
                path=os.path.join(self.data_dir, filename + ".samples"),
                use_times=False,
            )
            tsinfer.formats.SampleData.from_tree_sequence(
                modern_ts,
                path=os.path.join(self.data_dir, filename + ".keep_times.samples"),
                use_times=True,
            )

            # Update dataframe with details of simulation
            self.data = self.data.append(row_data, ignore_index=True)

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
        for name, fig in name_map.items():
            if fig in figures:
                fig = fig()
                if args.setup:
                    fig.setup()
                if args.inference:
                    fig.run_multiprocessing(fig.inference, num_processes=args.processes)

    else:
        fig = name_map[args.name]()
        if args.setup:
            fig.setup(args.processes)
        if args.inference:
            fig.run_multiprocessing(fig.inference, args.processes)
    if not args.setup and not args.inference:
        raise ValueError("must run with --setup, --inference, or both.")


if __name__ == "__main__":
    main()

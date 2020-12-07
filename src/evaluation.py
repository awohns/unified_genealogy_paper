import argparse
import csv
from itertools import combinations
import logging
import math
import multiprocessing
import os
import random
import subprocess
import shutil
import sys
import tempfile
from functools import reduce

import json
import numpy as np
import pandas as pd
import scipy
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_squared_log_error

import msprime
import tsinfer
import tskit
import stdpopsim

import utility
import run_inference

import tsdate  # NOQA

relate_executable = os.path.join("tools", "relate", "bin", "Relate")
relatefileformat_executable = os.path.join(
    "tools", "relate", "bin", "RelateFileFormats"
)
relate_popsize_executable = os.path.join(
    "tools",
    "relate_v1.1.2_x86_64_dynamic",
    "scripts",
    "EstimatePopulationSize",
    "EstimatePopulationSize.sh",
)
geva_executable = os.path.join("tools", "geva", "geva_v1beta")
geva_hmm_initial_probs = os.path.join("tools", "geva", "hmm", "hmm_initial_probs.txt")
geva_hmm_emission_probs = os.path.join("tools", "geva", "hmm", "hmm_emission_probs.txt")
tsinfer_executable = os.path.join("src", "run_tsinfer.py")
tsdate_executable = os.path.join("src", "run_tsdate.py")

TSDATE = "tsdate"
RELATE = "Relate"
GEVA = "GEVA"


def run_neutral_sim(
    sample_size, Ne, length, mutation_rate, recombination_rate, seed=None
):
    """
    Run simulation
    """
    ts = msprime.simulate(
        sample_size=sample_size,
        Ne=Ne,
        length=length,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        random_seed=seed,
    )
    return ts


def run_chr20_ooa(
    samples, Ne, length, mutation_rate, recombination_rate, rng, seed=None
):
    """
    Run StandardPopSim Out of Africa Chromosome 20
    """
    species = stdpopsim.get_species("HomSap")
    contig = species.get_contig("chr20", genetic_map="HapMapII_GRCh37")
    model = species.get_demographic_model("OutOfAfrica_3G09")
    engine = stdpopsim.get_default_engine()
    ts = engine.simulate(model, contig, samples, seed=seed)
    snippet_start = rng.randint(0, ts.get_sequence_length() - length)
    snippet = [snippet_start, snippet_start + length]
    return ts.keep_intervals(np.array([snippet])).trim()


def get_genetic_map_chr20_snippet(rowdata, filename):
    """
    For each chromosome 20 simulation, randomly select a region to run inference on
    """
    species = stdpopsim.get_species("HomSap")
    genetic_map = species.get_genetic_map("HapMapII_GRCh37")
    cm = genetic_map.get_chromosome_map("chr20")
    pos = np.array(cm.get_positions())
    snippet = np.where(
        np.logical_and(pos > rowdata["snippet"][0], pos < rowdata["snippet"][1])
    )
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


def infer_with_mismatch(
    sample_data,
    chromosome,
    ma_mismatch=0.1,
    ms_mismatch=0.1,
    precision=None,
    modern_samples_match=False,
    ancient_ancestors=False,
    num_threads=1,
):
    ancestors = tsinfer.generate_ancestors(sample_data, num_threads=num_threads)
    genetic_map = run_inference.get_genetic_map(chromosome)
    rho, ma_mis, ms_mis, precision = run_inference.get_rho(
        sample_data,
        ancestors,
        genetic_map,
        ma_mismatch,
        ms_mismatch,
        precision=None,
        num_threads=num_threads,
    )
    rho[:-1][rho[:-1] == 0] = np.min(rho[:-1][rho[:-1] != 0]) / 100
    ancestors_ts = run_inference.match_ancestors(
        sample_data,
        ancestors,
        rho,
        ma_mis,
        precision=13,
        ancient_ancestors=ancient_ancestors,
        num_threads=num_threads,
    )
    return run_inference.match_samples(
        sample_data,
        ancestors_ts,
        rho,
        ms_mis,
        13,
        modern_samples_match=modern_samples_match,
        ancient_ancestors=ancient_ancestors,
        num_threads=num_threads,
    )


def sample_times(ancient_sample_size, generation_time):
    """
    Pick random sample times from the Reich dataset
    """
    sample_times = np.genfromtxt("data/reich_ancient_samples_age.txt", skip_header=1)
    age_hist = np.histogram(sample_times, bins=100)
    age_distribution = age_hist[0] / sample_times.shape[0]
    sampled_ages = np.random.choice(
        (age_hist[1][1:] + age_hist[1][:-1]) / 2,
        size=ancient_sample_size,
        p=age_distribution,
    )
    return sampled_ages / generation_time


def run_neutral_ancients(
    sample_size_modern,
    sample_size_ancient,
    ancient_sample_times,
    length,
    Ne,
    mut_rate,
    rec_rate,
    seed=None,
):
    """
    Run a vanilla msprime simulation with a specified number of modern and ancient
    samples (the latter at given ages). Return both the simulated tree with
    ancients and the modern-only tree.
    """
    samples = [
        msprime.Sample(population=0, time=0) for samp in range(sample_size_modern)
    ]
    ancient_sample_times = np.array(ancient_sample_times, dtype=float)
    ancient_samples = [
        msprime.Sample(population=0, time=time)
        for samp, time in zip(range(sample_size_ancient), ancient_sample_times)
    ]
    samples = samples + ancient_samples

    return msprime.simulate(
        samples=samples,
        length=length,
        Ne=Ne,
        mutation_rate=mut_rate,
        recombination_rate=rec_rate,
        random_seed=seed,
    )


def remove_ancient_only_muts(ts, modern_samples=None):
    """
    Remove mutations which only appear in ancients, and mutations which are fixed when
    ancients are removed.
    """
    if modern_samples is None:
        modern_samples = np.where(ts.tables.nodes.time[ts.samples()] == 0)[0]
    modern_ts = ts.simplify(samples=modern_samples, keep_unary=True)

    del_sites = list(
        np.where(
            ~np.isin(ts.tables.sites.position[:], modern_ts.tables.sites.position[:])
        )[0]
    )
    for tree in modern_ts.trees():
        for site in tree.sites():
            assert len(site.mutations) == 1  # Only supports infinite sites muts.
            mut = site.mutations[0]
            # delete fixed mutations
            if tree.num_samples(mut.node) == modern_ts.num_samples:
                del_sites.append(site.id)
    tables = ts.dump_tables()
    tables.delete_sites(del_sites)
    deleted_ts = tables.tree_sequence()

    return deleted_ts


def remove_ancients(ts, modern_samples=None):
    """
    Remove all ancient samples and sites from simulated tree sequence
    """
    if modern_samples is None:
        modern_samples = np.where(ts.tables.nodes.time[ts.samples()] == 0)[0]
    modern_ts = ts.simplify(samples=modern_samples, keep_unary=True)

    del_sites = list()
    for tree in modern_ts.trees():
        for site in tree.sites():
            assert len(site.mutations) == 1  # Only supports infinite sites muts.
            mut = site.mutations[0]
            # delete fixed mutations
            if tree.num_samples(mut.node) == modern_ts.num_samples:
                del_sites.append(site.id)
            elif (
                tree.num_samples(mut.node) == 1
                # delete mutations that have become singletons
                and ts.at(site.position).num_samples(
                    ts.tables.mutations.node[
                        np.where(ts.tables.sites.position == site.position)[0][0]
                    ]
                )
                != 1
            ):
                del_sites.append(site.id)
    tables = modern_ts.dump_tables()
    tables.delete_sites(del_sites)
    modern_ts = tables.tree_sequence()

    return modern_ts


def return_vcf(tree_sequence, filename):
    with open("tmp/" + filename + ".vcf", "w") as vcf_file:
        tree_sequence.write_vcf(vcf_file, ploidy=2)


def sampledata_to_vcf(sample_data, filename):
    """
    Input sample_data file, output VCF
    """

    num_individuals = len(sample_data.individuals_metadata[:])
    ind_list = list()
    pos_geno_dict = {"POS": list()}

    for i in range(int(num_individuals / 2)):
        pos_geno_dict["msp_" + str(i)] = list()
        ind_list.append("msp_" + str(i))

    # add all the sample positions and genotypes
    for var in sample_data.variants():
        pos = int(round(var.site.position))
        if pos not in pos_geno_dict["POS"]:
            pos_geno_dict["POS"].append(pos)
            geno = var.genotypes
            for j in range(0, len(geno), 2):
                pos_geno_dict["msp_" + str(int(j / 2))].append(
                    str(geno[j]) + "|" + str(geno[j + 1])
                )

    df = pd.DataFrame(pos_geno_dict)

    df["#CHROM"] = 1
    df["REF"] = "A"
    df["ALT"] = "T"
    df["ID"] = "."
    df["QUAL"] = "."
    df["FILTER"] = "PASS"
    df["INFO"] = "."
    df["FORMAT"] = "GT"

    cols = [
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
    ] + ind_list
    df = df[cols]

    header = (
        """##fileformat=VCFv4.2
##source=msprime 0.6.0
##FILTER=<ID=PASS, Description="All filters passed">
##contig=<ID=1, length="""
        + str(int(sample_data.sequence_length))
        + """>
##FORMAT=<ID=GT, Number=1, Type=String, Description="Genotype">
"""
    )
    output_VCF = filename + ".vcf"
    with open(output_VCF, "w") as vcf:
        vcf.write(header)

    df.to_csv(output_VCF, sep="\t", mode="a", index=False)
    return df


def compare_mutations(
    ts_list,
    method_names=["tsdate", "tsdate_inferred"],
    relate_ages=None,
    relate_reinfer=None,
    geva_ages=None,
    geva_positions=None,
):
    """
    Given a list of tree sequences, return a pandas dataframe with the age
    estimates for each mutation via each method (tsdate, tsinfer + tsdate,
    relate, geva etc.)
    method_names: list of strings naming methods to be compared
    ts_list: The list of tree sequences
    geva_ages: mutation age estimates from geva (pandas df)
    relate_ages: mutation age estimates from relate (pandas df)
    Returns a DataFrame of mutations and age estimates from each method
    """

    assert len(ts_list) == len(method_names)
    # Load tree sequences: simulated, dated (topo), dated(inferred)
    ts = ts_list[0]
    print("Number of mutations", ts.num_mutations)
    run_results = utility.get_mut_pos_df(
        ts, "simulated_ts", ts.tables.nodes.time
    )  # , mutation_age="arithmetic")
    print("Number of mutations with true dates", run_results.shape[0])

    for cur_ts, method in zip(ts_list[1:], method_names[1:]):
        # Load age of mutations for each tree sequence
        mut_dated_ages = utility.get_mut_pos_df(
            cur_ts, method, cur_ts.tables.nodes.time
        )  # , mutation_age="arithmetic")
        print("Number of mutations dated by " + method + ": ", mut_dated_ages.shape[0])
        run_results = pd.merge(
            run_results, mut_dated_ages, how="left", left_index=True, right_index=True
        )

    # If Relate and GEVA were run, load mutation ages as pandas dataframe
    # Create an "age" column for both
    def get_relate_df(relate_ages, run_results, col_name):
        # remove mutations that relate can't date or flipped
        relate_ages = relate_ages[relate_ages["is_flipped"] == 0]
        relate_ages = relate_ages[relate_ages["is_not_mapping"] == 0]
        relate_ages[col_name] = np.sqrt(
            relate_ages["age_begin"] * relate_ages["age_end"]
        )
        relate = relate_ages[["pos_of_snp", col_name]].copy()
        relate = relate.rename(columns={"pos_of_snp": "position"}).set_index("position")
        print("Number of mutations dated by " + col_name + ": ", relate.shape[0])
        run_results = pd.merge(
            run_results, relate, how="left", left_index=True, right_index=True
        )
        return run_results

    if relate_ages is not None:
        run_results = get_relate_df(relate_ages, run_results, col_name="relate")
    if relate_reinfer is not None:
        run_results = get_relate_df(relate_ages, run_results, col_name="relate_reage")

    if geva_ages is not None and geva_positions is not None:
        # Merge the GEVA position indices and age estimates
        geva = pd.merge(
            geva_ages["PostMean"],
            geva_positions["Position"],
            how="left",
            left_index=True,
            right_index=True,
        )
        # For GEVA, we use PostMean as the age estimate
        geva = geva.rename(
            columns={"PostMean": "geva", "Position": "position"}
        ).set_index("position")
        print("Number of mutations dated by GEVA", geva.shape[0])
        run_results = pd.merge(
            run_results, geva, how="left", left_index=True, right_index=True
        )

    return run_results


def compare_mutation_msle_noancients(
    ts,
    inferred,
    dated_ts,
    iter_infer,
    dated_ts_iter,
    tsinfer_keep_times,
    tsdate_keep_times,
    tsdate_true_topo,
    sim_error_compatible,
    error_inferred_ts,
    error_dated_ts,
    error_iter_infer,
    error_dated_ts_iter,
):
    """
    Compare mutation accuracy in iterative approach
    """
    for compare_ts in [
        inferred,
        dated_ts,
        iter_infer,
        dated_ts_iter,
        tsinfer_keep_times,
        tsdate_keep_times,
        tsdate_true_topo,
    ]:
        assert np.array_equal(
            ts.tables.sites.position[:], compare_ts.tables.sites.position[:]
        )
    for compare_ts in [
        error_inferred_ts,
        error_dated_ts,
        error_iter_infer,
        error_dated_ts_iter,
    ]:
        assert np.array_equal(
            sim_error_compatible.tables.sites.position[:],
            compare_ts.tables.sites.position[:],
        )

    real_time = tsdate.sites_time_from_ts(ts, unconstrained=False)
    inferred_site_times = tsdate.sites_time_from_ts(inferred, unconstrained=False)
    tsdate_time = tsdate.sites_time_from_ts(dated_ts)
    iteration_time = tsdate.sites_time_from_ts(dated_ts_iter)
    keep_site_times = tsdate.sites_time_from_ts(tsdate_keep_times)
    simulated_topo_time = tsdate.sites_time_from_ts(tsdate_true_topo)
    real_time_error = tsdate.sites_time_from_ts(
        sim_error_compatible, unconstrained=False
    )
    error_inferred_time = tsdate.sites_time_from_ts(
        error_inferred_ts, unconstrained=False
    )
    error_tsdate_time = tsdate.sites_time_from_ts(error_dated_ts)
    error_iteration_time = tsdate.sites_time_from_ts(error_dated_ts_iter)

    run_results = pd.DataFrame(
        [
            mean_squared_log_error(real_time, tsdate_time),
            mean_squared_log_error(real_time, iteration_time),
            mean_squared_log_error(real_time, keep_site_times),
            mean_squared_log_error(real_time, simulated_topo_time),
            pearsonr(real_time, tsdate_time)[0],
            pearsonr(real_time, iteration_time)[0],
            pearsonr(real_time, keep_site_times)[0],
            pearsonr(real_time, simulated_topo_time)[0],
            spearmanr(real_time, inferred_site_times)[0],
            spearmanr(real_time, tsdate_time)[0],
            spearmanr(real_time, iteration_time)[0],
            spearmanr(real_time, keep_site_times)[0],
            spearmanr(real_time, simulated_topo_time)[0],
            ts.kc_distance(inferred, lambda_=0),
            ts.kc_distance(dated_ts, lambda_=1),
            ts.kc_distance(iter_infer, lambda_=0),
            ts.kc_distance(dated_ts_iter, lambda_=1),
            ts.kc_distance(tsdate_keep_times, lambda_=0),
            ts.kc_distance(tsdate_keep_times, lambda_=1),
            ts.kc_distance(tsdate_true_topo, lambda_=1),
            mean_squared_log_error(real_time_error, error_tsdate_time),
            mean_squared_log_error(real_time_error, error_iteration_time),
            pearsonr(real_time_error, error_tsdate_time)[0],
            pearsonr(real_time_error, error_iteration_time)[0],
            spearmanr(real_time_error, error_inferred_time)[0],
            spearmanr(real_time_error, error_tsdate_time)[0],
            spearmanr(real_time_error, error_iteration_time)[0],
            sim_error_compatible.kc_distance(error_inferred_ts, lambda_=0),
            sim_error_compatible.kc_distance(error_dated_ts, lambda_=1),
            sim_error_compatible.kc_distance(error_iter_infer, lambda_=0),
            sim_error_compatible.kc_distance(error_dated_ts_iter, lambda_=1),
        ],
        index=[
            "tsdate_MSLE",
            "iteration_MSLE",
            "keeptime_MSLE",
            "topo_MSLE",
            "tsdate_Pearson",
            "iteration_Pearson",
            "keeptime_Pearson",
            "topo_Pearson",
            "frequency_Spearman",
            "tsdate_Spearman",
            "iteration_Spearman",
            "keeptime_Spearman",
            "topo_Spearman",
            "inferred_KC_0",
            "dated_KC_1",
            "iter_KC_0",
            "iter_KC_1",
            "keep_times_KC_0",
            "keep_times_KC_1",
            "topo_KC_1",
            "tsdate_error_MSLE",
            "iteration_error_MSLE",
            "tsdate_error_Pearson",
            "iteration_error_Pearson",
            "inferred_error_Spearman",
            "tsdate_error_Spearman",
            "iteration_error_Spearman",
            "inferred_error_KC_0",
            "dated_error_KC_1",
            "iter_error_KC_0",
            "iter_error_KC_1",
        ],
    ).T
    return run_results


def compare_mutations_iterative(
    ancient_sample_size,
    ts,
    modern_ts,
    inferred,
    tsdate_dates,
    constrained_ages,
    iter_infer,
    iter_dates,
    tsinfer_keep_times,
    tsdate_keep_times,
    tsdate_true_topo,
):
    """
    Compare mutation accuracy in iterative approach
    """
    simulated_df = utility.get_mut_pos_df(ts, "TrueTime", ts.tables.nodes.time[:])
    tsdate_df = utility.get_mut_pos_df(inferred, "tsdateTime", tsdate_dates)
    constr_df = utility.get_mut_pos_df(inferred, "ConstrainedTime", constrained_ages)
    iter_df = utility.get_mut_pos_df(iter_infer, "IterationTime", iter_dates)
    keep_times_df = utility.get_mut_pos_df(
        tsinfer_keep_times, "keeptimeTime", tsdate_keep_times
    )
    simulated_topo_df = utility.get_mut_pos_df(
        modern_ts, "simulatedTopoTime", tsdate_true_topo
    )

    mut_df = pd.DataFrame(
        index=range(modern_ts.num_mutations),
        columns=[
            "ancient_sample_size",
            "TrueTime",
            "tsdateTime",
            "ConstrainedTime",
            "keeptimeTime",
            "simulatedTopoTime",
            "IterationTime",
        ],
    )
    dfs = [
        simulated_df,
        tsdate_df,
        constr_df,
        iter_df,
        keep_times_df,
        simulated_topo_df,
    ]
    mut_df["ancient_sample_size"] = ancient_sample_size
    run_results = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="inner"
        ),
        dfs,
    )
    msle_results_list = [ancient_sample_size]
    pearsonr_results_list = [ancient_sample_size]
    spearmanr_results_list = [ancient_sample_size]
    for results in [
        run_results["tsdateTime"],
        run_results["ConstrainedTime"],
        run_results["IterationTime"],
        run_results["keeptimeTime"],
        run_results["simulatedTopoTime"],
    ]:
        msle_results_list.append(
            mean_squared_log_error(run_results["TrueTime"], results)
        )
        pearsonr_results_list.append(pearsonr(run_results["TrueTime"], results)[0])
        spearmanr_results_list.append(spearmanr(run_results["TrueTime"], results)[0])
    index = [
        "ancient_sample_size",
        "tsdateTime",
        "ConstrainedTime",
        "IterationTime",
        "tsinfer_keep_time",
        "SimulatedTopoTime",
    ]

    msle_run_results = pd.DataFrame(msle_results_list, index=index).T
    pearsonr_run_results = pd.DataFrame(pearsonr_results_list, index=index).T
    spearmanr_run_results = pd.DataFrame(spearmanr_results_list, index=index).T

    return msle_run_results, pearsonr_run_results, spearmanr_run_results


def construct_tsinfer_name(sim_name, subsample_size, input_seq_error=None):
    """
    Returns a TSinfer filename. In the future we may have a tweakable error parameter
    for tsinfer, which may be different from the actual error injected into the
    simulated samples, so we allow for this here.
    If the file is a subset of the original, this can be added to the
    basename in this function, or later using the
    add_subsample_param_to_name() routine.
    """
    d, f = os.path.split(sim_name)
    suffix = "" if input_seq_error is None else "SQerr{}".format(input_seq_error)
    name = os.path.join(d, "+".join(["tsinfer", f, suffix]))
    if subsample_size is not None and not pd.isnull(subsample_size):
        name = add_subsample_param_to_name(name, subsample_size)
    return name


def run_tsdate(input_fn, Ne, mut_rate, timepoints, method):
    with tempfile.NamedTemporaryFile("w+") as ts_out:
        cmd = [
            sys.executable,
            tsdate_executable,
            input_fn,
            ts_out.name,
            str(Ne),
            "--mutation-rate",
            str(mut_rate),
        ]
        # cmd = ["tsdate", "date", input_fn, ts_out.name, str(Ne)]
        # cmd += ["--mutation-rate", str(mut_rate), "--timepoints", str(timepoints), "--method", str(method)]
        cpu_time, memory_use = time_cmd(cmd)
        dated_ts = tskit.load(ts_out.name)
    return dated_ts, cpu_time, memory_use


def run_tsdate_posterior_ts(ts, Ne, mut_rate, method="inside_outside", priors=None):
    """
    Simple wrapper to get dated tree sequence and posterior NodeGridValues
    """
    dates, posterior, timepoints, eps, nds = tsdate.get_dates(
        ts,
        Ne=Ne,
        mutation_rate=mut_rate,
        method=method,
        priors=priors,
    )
    constrained = tsdate.constrain_ages_topo(ts, dates, eps, nds)
    tables = ts.dump_tables()
    tables.nodes.time = constrained * 2 * Ne
    tables.sort()
    dated_ts = tables.tree_sequence()
    return dated_ts, dates * 2 * Ne, posterior


def get_dated_ts(ts, dates, Ne, eps):
    """
    Simple wrapper to get dated tree sequence from unconstrained dates
    NOTE: dates are assumed to be in generations
    """
    constrained = tsdate.constrain_ages_topo(ts, dates, eps)
    tables = ts.dump_tables()
    tables.nodes.time = constrained
    tables.sort()
    dated_ts = tables.tree_sequence()
    return dated_ts


def tsdate_iter(ts, Ne, mut_rate, method, priors, posterior):
    """
    Rerun tsdate, using posterior of previous run as prior.
    """
    priors.grid_data = posterior.grid_data
    dates, posterior, timepoints, eps, nds = tsdate.get_dates(
        ts,
        Ne=Ne,
        mutation_rate=mut_rate,
        method=method,
        priors=priors,
    )
    constrained = tsdate.constrain_ages_topo(ts, dates, eps, nds)
    tables = ts.dump_tables()
    tables.nodes.time = constrained * 2 * Ne
    tables.sort()
    iter_dated = tables.tree_sequence()
    return iter_dated, dates * 2 * Ne, posterior


# def compare_mutations(ts_list, relate_ages=None, geva_ages=None, geva_positions=None):
#    """
#    Given a list of tree sequences, return a pandas dataframe with the age
#    estimates for each mutation via each method (tsdate, tsinfer + tsdate,
#    relate, geva etc.)
#    """
#
#    # Load tree sequences: simulated, dated (topo), dated(inferred)
#    ts = ts_list[0]
#    dated_ts = ts_list[1]
#    dated_inferred_ts = ts_list[2]
#
#    # Load age of mutations for each tree sequence
#    print("Number of mutations", ts.num_mutations)
#    mut_ages = utility.get_mut_pos_df(ts, "simulated_ts", ts.tables.nodes.time[:])
#    print("Number of mutations with true dates", mut_ages.shape[0])
#    mut_dated_ages = utility.get_mut_pos_df(
#        dated_ts, "tsdate", dated_ts.tables.nodes.time[:]
#    )
#    print("Number of mutations dated by tsdate", mut_dated_ages.shape[0])
#    run_results = pd.merge(
#        mut_ages, mut_dated_ages, how="left", left_index=True, right_index=True
#    )
#    mut_inferred_dated_ages = utility.get_mut_pos_df(
#        dated_inferred_ts, "tsdate_inferred", dated_inferred_ts.tables.nodes.time[:], exclude_root=True
#    )
#    print(
#        "Number of mutations dated by tsinfer + tsdate",
#        mut_inferred_dated_ages.shape[0],
#    )
#    run_results = pd.merge(
#        run_results,
#        mut_inferred_dated_ages,
#        how="left",
#        left_index=True,
#        right_index=True,
#    )
#
#    # If Relate and GEVA were run, load mutation ages as pandas dataframe
#    # Create an "age" column for both
#    if relate_ages is not None:
#        # remove mutations that relate can't date or flipped
#        relate_ages = relate_ages[relate_ages["is_flipped"] == 0]
#        relate_ages = relate_ages[relate_ages["is_not_mapping"] == 0]
#        relate_ages["relate"] = (relate_ages["age_begin"] + relate_ages["age_end"]) / 2
#        relate = relate_ages[["pos_of_snp", "relate"]].copy()
#        relate = relate.rename(columns={"pos_of_snp": "position"}).set_index("position")
#        print("Number of mutations dated by relate", relate.shape[0])
#        run_results = pd.merge(
#            run_results, relate, how="left", left_index=True, right_index=True
#        )
#
#    if geva_ages is not None and geva_positions is not None:
#        # Merge the GEVA position indices and age estimates
#        geva = pd.merge(
#            geva_ages["PostMean"],
#            geva_positions["Position"],
#            how="left",
#            left_index=True,
#            right_index=True,
#        )
#        # For GEVA, we use PostMean as the age estimate
#        geva = geva.rename(
#            columns={"PostMean": "geva", "Position": "position"}
#        ).set_index("position")
#        print(geva.head())
#        print("Number of mutations dated by GEVA", geva.shape[0])
#        run_results = pd.merge(
#            run_results, geva, how="left", left_index=True, right_index=True
#        )
#        print(run_results)
#
#    return run_results


def compare_mutations_tslist(ts_list, dates_list, method_names):
    """
    Given a list of tree sequences and a list of names of how they were generated,
    return a pandas dataframe with the age estimates for each mutation via each
    method (tsdate, tsinfer + tsdate, tsdate iterations)
    """

    # Load tree sequences: simulated, dated (topo), dated(inferred)
    first_ts = ts_list[0]
    results = utility.get_mut_pos_df(first_ts, method_names[0], dates_list[0])
    for ts, dates, method_name in zip(ts_list[1:], dates_list[1:], method_names[1:]):
        # Load age of mutations for each tree sequence
        mut_ages = utility.get_mut_pos_df(ts, method_name, dates)
        results = pd.merge(
            results, mut_ages, how="inner", left_index=True, right_index=True
        )
    return results


def get_kc_distances(ts_list, method_names):
    """
    Get kc_distances between a list of tree sequences with lambda at 0 and 1.
    Simulated tree sequence must be first in the list.
    """

    # Load tree sequences
    first_ts = ts_list[0]
    results_lambda_0 = dict()
    results_lambda_1 = dict()
    print(first_ts.first().num_roots)
    for ts, method_name in zip(ts_list[1:], method_names[1:]):
        print(ts.first().num_roots)
        results_lambda_0[method_name] = first_ts.kc_distance(ts, lambda_=0)
        results_lambda_1[method_name] = first_ts.kc_distance(ts, lambda_=1)
    return pd.DataFrame.from_dict([results_lambda_0, results_lambda_1])


def iteration_tsdate(
    constr_sample_data, constr_sites, Ne, mut_rate, adjust_priors=True
):
    iter_infer = tsinfer.infer(constr_sample_data).simplify()
    priors = tsdate.build_prior_grid(iter_infer)
    if adjust_priors and constr_sites:
        for mut_pos, limit in constr_sites.items():
            infer_mut_pos = np.where(mut_pos == iter_infer.tables.sites.position)[0][0]
            node = (
                iter_infer.tables.mutations.node[infer_mut_pos] - iter_infer.num_samples
            )
            priors.grid_data[node][
                : (np.abs(priors.timepoints * 20000 - limit)).argmin()
            ] = 0
    iter_dates, _, _, _, _ = tsdate.get_dates(
        iter_infer, Ne=Ne, mutation_rate=mut_rate, priors=priors
    )
    return iter_infer, iter_dates * 2 * Ne


def run_tsinfer(
    sample_fn,
    length,
    num_threads=1,
    inject_real_ancestors_from_ts_fn=None,
    rho=None,
    error_probability=None,
):
    with tempfile.NamedTemporaryFile("w+") as ts_out:
        cmd = ["tsinfer", "infer", sample_fn, "-O", ts_out.name]
        # cmd += ["--threads", str(num_threads), ts_out.name]
        if inject_real_ancestors_from_ts_fn:
            logging.debug(
                "Injecting real ancestors constructed from {}".format(
                    inject_real_ancestors_from_ts_fn
                )
            )
            cmd.extend(
                ["--inject-real-ancestors-from-ts", inject_real_ancestors_from_ts_fn]
            )
        cpu_time, memory_use = time_cmd(cmd)
        ts_simplified = tskit.load(ts_out.name)
    return ts_simplified, cpu_time, memory_use


def run_relate(ts, path_to_vcf, mut_rate, Ne, genetic_map_path, working_dir, output):
    """
    Run relate software on tree sequence. Requires vcf of simulated data
    Relate needs to run in its own directory (param working_dir)
    NOTE: Relate's effective population size is "of haplotypes"
    """
    cur_dir = os.getcwd()
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)
    os.chdir(working_dir)
    with tempfile.NamedTemporaryFile("w+") as relate_out:
        subprocess.run(
            [
                os.path.join(cur_dir, relatefileformat_executable),
                "--mode",
                "ConvertFromVcf",
                "--haps",
                output + ".haps",
                "--sample",
                output + ".sample",
                "-i",
                path_to_vcf,
            ]
        )
        cpu_time, memory_use = time_cmd(
            [
                os.path.join(cur_dir, relate_executable),
                "--mode",
                "All",
                "-m",
                str(mut_rate),
                "-N",
                str(Ne),
                "--haps",
                output + ".haps",
                "--sample",
                output + ".sample",
                "--seed",
                "1",
                "-o",
                output,
                "--map",
                os.path.join(cur_dir, genetic_map_path),
                "--memory",
                "32",
            ]
        )
        subprocess.check_output(
            [
                os.path.join(cur_dir, relatefileformat_executable),
                "--mode",
                "ConvertToTreeSequence",
                "-i",
                output,
                "-o",
                output,
            ]
        )
    relate_ts = tskit.load(output + ".trees")

    # Set samples flags to "1"
    table_collection = relate_ts.dump_tables()
    samples = np.repeat(1, ts.num_samples)
    internal = np.repeat(0, relate_ts.num_nodes - ts.num_samples)
    correct_sample_flags = np.array(np.concatenate([samples, internal]), dtype="uint32")
    table_collection.nodes.set_columns(
        flags=correct_sample_flags, time=relate_ts.tables.nodes.time
    )
    relate_ts_fixed = table_collection.tree_sequence()
    relate_ages = pd.read_csv(output + ".mut", sep=";")
    os.chdir(cur_dir)
    return relate_ts_fixed, relate_ages, cpu_time, memory_use


def create_poplabels(ts, output):
    population_names = []
    for population in ts.populations():
        population_names.append(
            (
                json.loads(population.metadata)["id"],
                np.sum(ts.tables.nodes.population[ts.samples()] == population.id) / 2,
            )
        )
    with open(output + ".poplabels", "w") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(["Sample population group sex"])
        sample_id = 0
        for population in population_names:
            for indiv in range(int(population[1])):
                wr.writerow(
                    [str(sample_id) + " " + population[0] + " " + population[0] + " NA"]
                )
                sample_id += 1


def run_relate_pop_size(ts, path_to_files, mutation_rate, output, working_dir):
    """
    Run Relate's EstimatePopulationSize script
    """
    cur_dir = os.getcwd()
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)
    os.chdir(working_dir)
    print(ts, output)
    create_poplabels(ts, output)
    print(ts, path_to_files, mutation_rate, output, working_dir)
    with tempfile.NamedTemporaryFile("w+") as relate_out:
        subprocess.run(
            [
                os.path.join(cur_dir, relate_popsize_executable),
                "-i",
                path_to_files,
                "-m",
                str(mutation_rate),
                "--poplabels",
                output + ".poplabels",
                "-o",
                output,
            ]
        )
        subprocess.check_output(
            [
                os.path.join(cur_dir, relatefileformat_executable),
                "--mode",
                "ConvertToTreeSequence",
                "-i",
                output,
                "-o",
                output,
            ]
        )

    new_ages = pd.read_csv(output + ".mut", sep=";")
    new_relate_ts = tskit.load(output + ".trees")
    os.chdir(cur_dir)
    return new_ages, new_relate_ts


def run_geva(file_name, Ne, mut_rate, rec_rate):
    """
    Perform GEVA age estimation on a given vcf
    """
    print(file_name)
    subprocess.check_output(
        [
            geva_executable,
            "--out",
            file_name,
            "--rec",
            str(rec_rate),
            "--vcf",
            file_name + ".vcf",
        ]
    )
    with open(file_name + ".positions.txt", "wb") as out:
        subprocess.call(
            ["awk", "NR>3 {print last} {last = $3}", file_name + ".marker.txt"],
            stdout=out,
        )
    try:
        cpu_time, memory_use = time_cmd(
            [
                geva_executable,
                "-i",
                file_name + ".bin",
                "--positions",
                file_name + ".positions.txt",
                "--hmm",
                geva_hmm_initial_probs,
                geva_hmm_emission_probs,
                "--Ne",
                str(Ne),
                "--mut",
                str(mut_rate),
                "--maxConcordant",
                "200",
                "--maxDiscordant",
                "200",
                "-o",
                file_name + "_estimation",
            ]
        )
    except subprocess.CalledProcessError as grepexc:
        print(grepexc.output)

    age_estimates = pd.read_csv(
        file_name + "_estimation.sites.txt", sep=" ", index_col="MarkerID"
    )
    keep_ages = age_estimates[
        (age_estimates["Clock"] == "J") & (age_estimates["Filtered"] == 1)
    ]
    return keep_ages, cpu_time, memory_use


def geva_age_estimate(file_name, Ne, mut_rate, rec_rate):
    """
    Perform GEVA age estimation on a given vcf
    """
    file_name = "tmp/" + file_name
    subprocess.check_output(
        [
            geva_executable,
            "--out",
            file_name,
            "--rec",
            str(rec_rate),
            "--vcf",
            file_name + ".vcf",
        ]
    )
    with open(file_name + ".positions.txt", "wb") as out:
        subprocess.call(
            ["awk", "NR>3 {print last} {last = $3}", file_name + ".marker.txt"],
            stdout=out,
        )
    try:
        subprocess.check_output(
            [
                geva_executable,
                "-i",
                file_name + ".bin",
                "--positions",
                file_name + ".positions.txt",
                "--hmm",
                "/Users/anthonywohns/Documents/mcvean_group/age_inference/"
                "tsdate/tools/geva/hmm/hmm_initial_probs.txt",
                "/Users/anthonywohns/Documents/mcvean_group/age_inference/tsdate/tools/"
                "geva/hmm/hmm_emission_probs.txt",
                "--Ne",
                str(Ne),
                "--mut",
                str(mut_rate),
                "--maxConcordant",
                "200",
                "--maxDiscordant",
                "200",
                "-o",
                file_name + "_estimation",
            ]
        )
    except subprocess.CalledProcessError as grepexc:
        print(grepexc.output)

    age_estimates = pd.read_csv(
        file_name + "_estimation.sites.txt", sep=" ", index_col="MarkerID"
    )
    keep_ages = age_estimates[
        (age_estimates["Clock"] == "J") & (age_estimates["Filtered"] == 1)
    ]
    return keep_ages


def constrain_with_ancient(
    sample_data, dates, inferred, ancient_genos=None, ancient_ages=None
):
    """
    Take date estimates and constrain using ancient information
    """

    ancient_ages = np.array(ancient_ages)
    constr_ages = np.zeros_like(sample_data.sites_time[:])
    sites_pos = sample_data.sites_position[:]

    if ancient_ages is not None:
        constr_sites = {}
    else:
        constr_sites = None
    for mut in inferred.mutations():
        constr_ages[sites_pos == mut.position] = dates[mut.node]
        if ancient_ages is not None:
            if np.any(ancient_genos[mut.id] == 1):
                constr_sites[mut.position] = np.max(
                    ancient_ages[ancient_genos[mut.id] == 1]
                )
                if np.max(ancient_ages[ancient_genos[mut.id] == 1]) > dates[mut.node]:
                    constr_ages[sites_pos == mut.position] = np.max(
                        ancient_ages[ancient_genos[mut.id] == 1]
                    )

    constr_sample_data = sample_data.copy()
    constr_sample_data.sites_time[:] = constr_ages
    constr_sample_data.finalise()
    return constr_sample_data, constr_sites


def run_all_methods_compare(
    index,
    ts,
    n,
    Ne,
    mutation_rate,
    recombination_rate,
    time_grid,
    grid_slices,
    estimation_method,
    approximate_prior,
    error_model,
    include_geva,
    seed,
):
    """
    Function to run all comparisons and return dataframe of mutations
    """
    output = "comparison_" + str(index)
    if error_model is not None:
        error_samples = generate_samples(
            ts, "error_comparison_" + str(index), empirical_seq_err_name=error_model
        )
        # return_vcf(samples, "comparison_" + str(index))
        sampledata_to_vcf(error_samples, "comparison_" + str(index))
    else:
        samples = generate_samples(ts, "comparison_" + str(index))
        sampledata_to_vcf(samples, "comparison_" + str(index))
    dated_ts, inferred_ts, dated_inferred_ts = run_tsdate(
        ts,
        n,
        Ne,
        mutation_rate,
        time_grid,
        grid_slices,
        estimation_method,
        approximate_prior,
    )
    if include_geva:
        geva_ages = geva_age_estimate(
            "comparison_" + str(index), Ne, mutation_rate, recombination_rate
        )
        geva_positions = pd.read_csv(
            "tmp/comparison_" + str(index) + ".marker.txt",
            delimiter=" ",
            index_col="MarkerID",
        )
    relate_output = run_relate(
        ts, "comparison_" + str(index), mutation_rate, Ne * 2, output
    )
    if include_geva:
        compare_df = compare_mutations(
            ["simulated_ts", "tsdate", "tsdate_inferred", "geva", "relate"],
            [ts, dated_ts, dated_inferred_ts],
            geva_ages=geva_ages,
            geva_positions=geva_positions,
            relate_ages=relate_output[1],
        )
        tmrca_compare = find_tmrcas_snps(
            {
                "ts": ts,
                "tsdate_true": dated_ts,
                "tsdate_inferred": dated_inferred_ts,
                "relate": relate_output[0],
            }
        )
        # kc_distances = [kc_distance_ts(ts, inferred_ts, 0),
        #                 kc_distance_ts(ts, relate_output[0], 0),
        #                 kc_distance_ts(ts, inferred_ts_round2, 0),
        #                 kc_distance_ts(ts, dated_ts, 1),
        #                 kc_distance_ts(ts, dated_inferred_ts, 1),
        #                 kc_distance_ts(ts, tsdated_inferred_ts_wtimes, 1),
        #                 kd_distance_ts(ts, tsdated_ts_round2),
        #                 kc_distance_ts(ts, relate_output[0], 1)]
    else:
        compare_df = compare_mutations(
            ["simulated_ts", "tsdate", "tsdate_inferred", "geva", "relate"],
            [ts, dated_ts, dated_inferred_ts],
            relate_ages=relate_output[1],
        )
        tmrca_compare = find_tmrcas_snps(
            {
                "ts": ts,
                "tsdate_true": dated_ts,
                "tsdate_inferred": dated_inferred_ts,
                "relate": relate_output[0],
            }
        )
        # kc_distances = [kc_distance_ts(ts, inferred_ts, 0),
        #                 kc_distance_ts(ts, relate_output[0], 0),
        #                 kc_distance_ts(ts, inferred_ts_round2, 0),
        #                 kc_distance_ts(ts, dated_ts, 1),
        #                 kc_distance_ts(ts, dated_inferred_ts, 1),
        #                 kc_distance_ts(ts, tsdated_inferred_ts_wtimes, 1),
        #                 kd_distance_ts(ts, tsdated_ts_round2),
        #                 kc_distance_ts(ts, relate_output[0], 1)]

    return compare_df, tmrca_compare


def run_all_tests(params):
    """
    Runs simulation and all tests for the simulation
    """
    index = int(params[0])
    n = int(params[1])
    Ne = float(params[2])
    length = int(params[3])
    mutation_rate = float(params[4])
    recombination_rate = float(params[5])
    model = params[6]
    time_grid = params[7]
    grid_slices = params[8]
    estimation_method = params[9]
    approximate_prior = params[10]
    error_model = params[11]
    include_geva = params[12]
    seed = float(params[13])

    if model == "neutral":
        ts = run_neutral_sim(n, Ne, length, mutation_rate, recombination_rate, seed)
    elif model == "out_of_africa":
        ts = out_of_africa(n, mutation_rate, recombination_rate, length)
    compare_df, tmrca_compare = run_all_methods_compare(
        index,
        ts,
        n,
        Ne,
        mutation_rate,
        recombination_rate,
        time_grid,
        grid_slices,
        estimation_method,
        approximate_prior,
        error_model,
        include_geva,
        seed,
    )

    return compare_df, tmrca_compare


def run_multiprocessing(function, params, output, num_replicates, num_processes):
    """
    Run multiprocessing of inputted function a specified number of times
    """
    mutation_results = list()
    tmrca_results = list()
    # kc_distances = list()
    if num_processes > 1:
        logging.info(
            "Setting up using multiprocessing ({} processes)".format(num_processes)
        )
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=2) as pool:
            for result in pool.imap_unordered(function, params):
                #  prior_results = pd.read_csv("data/result")
                #  combined = pd.concat([prior_results, result])
                mutation_results.append(result[0])
                tmrca_results.append(result[1])
                # kc_distances.append(result[2])
    else:
        # When we have only one process it's easier to keep everything in the
        # same process for debugging.
        logging.info("Setting up using a single process")
        for result in map(function, params):
            mutation_results.append(result[0])
            tmrca_results.append(result[1])
            # kc_distances.append(result[2])
    master_mutation_df = pd.concat(mutation_results)
    master_tmrca_df = np.column_stack(tmrca_results)
    master_mutation_df.to_csv("data/" + output + "_mutations")
    np.savetxt("data/" + output + "_tmrcas", master_tmrca_df, delimiter=",")
    # print(kc_distances)
    return master_mutation_df


def time_cmd(cmd, stdout=sys.stdout):
    """
    Runs the specified command line (a list suitable for subprocess.call)
    and writes the stdout to the specified file object.
    """
    if sys.platform == "darwin":
        # on OS X, install gtime using `brew install gnu-time`
        time_cmd = "/usr/local/bin/gtime"
    else:
        time_cmd = "/usr/bin/time"
    full_cmd = [time_cmd, "-f%M %S %U"] + cmd

    with tempfile.TemporaryFile() as stderr:
        exit_status = subprocess.call(full_cmd, stderr=stderr)
        stderr.seek(0)
        if exit_status != 0:
            raise ValueError(
                "Error running '{}': status={}:stderr{}".format(
                    " ".join(cmd), exit_status, stderr.read()
                )
            )

        split = stderr.readlines()[-1].split()
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicates", type=int, default=10, help="number of replicates"
    )
    parser.add_argument("num_samples", help="number of samples to simulate")
    parser.add_argument("output", help="name of output files")
    parser.add_argument(
        "-n", "--Ne", type=float, default=10000, help="effective population size"
    )
    parser.add_argument(
        "--length", "-l", type=int, default=1e5, help="Length of the sequence"
    )
    parser.add_argument(
        "-m", "--mutation-rate", type=float, default=None, help="mutation rate"
    )
    parser.add_argument(
        "-r",
        "--recombination-rate",
        type=float,
        default=None,
        help="recombination rate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="neutral",
        help="choose neutral or out of africa model",
    )
    parser.add_argument(
        "-g",
        "--grid-slices",
        type=int,
        default=50,
        help="how many slices/quantiles to pass to time grid",
    )
    parser.add_argument(
        "--estimation-method",
        type=str,
        default="inside_outside",
        help="use inside-outside or maximization method",
    )
    parser.add_argument(
        "-e", "--error-model", type=str, default=None, help="input error model"
    )
    parser.add_argument(
        "-t",
        "--time-grid",
        type=str,
        default="adaptive",
        help="adaptive or uniform time grid",
    )
    parser.add_argument(
        "-a", "--approximate-prior", action="store_true", help="use approximate prior"
    )
    parser.add_argument(
        "--include-geva", action="store_true", help="run comparisons with GEVA"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=123, help="use a non-default RNG seed"
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=1,
        help="number of worker processes, e.g. 40",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    seeds = [rng.randint(1, 2 ** 31) for i in range(args.replicates)]
    inputted_params = [
        int(args.num_samples),
        args.Ne,
        args.length,
        args.mutation_rate,
        args.recombination_rate,
        args.model,
        args.time_grid,
        args.grid_slices,
        args.estimation_method,
        args.approximate_prior,
        args.error_model,
        args.include_geva,
    ]
    params = iter(
        [
            np.concatenate([[index], inputted_params, [seed]])
            for index, seed in enumerate(seeds)
        ]
    )
    mutation_df = run_multiprocessing(
        run_all_tests, params, args.output, args.replicates, args.processes
    )


if __name__ == "__main__":
    main()

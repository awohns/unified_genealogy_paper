import csv
import logging
import os
import subprocess
import sys
import tempfile

import json
import numpy as np
import pandas as pd

import msprime
import tsinfer
import tskit
import stdpopsim

import utility

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


def infer_with_mismatch(
    sample_data,
    path_to_genetic_map,
    ma_mismatch=1,
    ms_mismatch=1,
    precision=15,
    num_threads=1,
    path_compression=True,
    progress_monitor=False,
):
    ancestors = tsinfer.generate_ancestors(
        sample_data, num_threads=num_threads, progress_monitor=progress_monitor
    )
    gmap = msprime.RateMap.read_hapmap(
        path_to_genetic_map, sequence_length=ancestors.sequence_length
    )
    genetic_dists = tsinfer.Matcher.recombination_rate_to_dist(
        gmap, ancestors.sites_position[:]
    )
    recombination = tsinfer.Matcher.recombination_dist_to_prob(genetic_dists)
    recombination[recombination == 0] = 1e-20
    mismatch = np.full(
        len(ancestors.sites_position[:]),
        tsinfer.Matcher.mismatch_ratio_to_prob(1, np.median(genetic_dists), 2),
    )

    ancestors_ts = tsinfer.match_ancestors(
        sample_data,
        ancestors,
        recombination=recombination,
        mismatch=mismatch,
        precision=precision,
        num_threads=num_threads,
        path_compression=path_compression,
        progress_monitor=progress_monitor,
    )
    return tsinfer.match_samples(
        sample_data,
        ancestors_ts,
        recombination=recombination,
        mismatch=mismatch,
        precision=precision,
        num_threads=num_threads,
        path_compression=path_compression,
        progress_monitor=progress_monitor,
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


def remove_ancient_only_muts(ts, modern_samples=None):
    """
    Remove mutations which only appear in ancients, and mutations which are fixed when
    ancients are removed.
    """
    tables = ts.dump_tables()
    tables.migrations.clear()
    ts_nomig = tables.tree_sequence()
    if modern_samples is None:
        modern_samples = np.where(ts.tables.nodes.time[ts.samples()] == 0)[0]
    modern_ts = ts_nomig.simplify(
        samples=modern_samples, keep_unary=True, filter_sites=False
    )

    assert modern_ts.num_sites == ts.num_sites
    del_sites = []
    for tree in modern_ts.trees():
        for site in tree.sites():
            assert len(site.mutations) <= 1  # Only supports infinite sites muts.
            if len(site.mutations) == 0:
                del_sites.append(site.id)
            else:
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
        ts, "simulated_ts", ts.tables.nodes.time, node_selection="msprime"
    )
    print("Number of mutations with true dates", run_results.shape[0])

    for cur_ts, method in zip(ts_list[1:], method_names[1:]):
        # Load age of mutations for each tree sequence
        mut_dated_ages = utility.get_mut_pos_df(
            cur_ts,
            method,
            cur_ts.tables.nodes.time,
            node_selection="arithmetic",
            exclude_root=True,
        )
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
        relate_ages[col_name] = (relate_ages["age_begin"] + relate_ages["age_end"]) / 2
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
        run_results = get_relate_df(
            relate_reinfer, run_results, col_name="relate_iterate"
        )

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


def get_kc_distances(ts_list, method_names):
    """
    Get kc_distances between a list of tree sequences with lambda at 0 and 1.
    Simulated tree sequence must be first in the list.
    """

    # Load tree sequences
    first_ts = ts_list[0]
    results_lambda_0 = dict()
    results_lambda_1 = dict()
    for ts, method_name in zip(ts_list[1:], method_names[1:]):
        results_lambda_0[method_name] = first_ts.kc_distance(ts, lambda_=0)
        results_lambda_1[method_name] = first_ts.kc_distance(ts, lambda_=1)
    return pd.DataFrame.from_dict([results_lambda_0, results_lambda_1])


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
    Run relate software on tree sequence. Requires vcf of simulated data and genetic map.
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
    create_poplabels(ts, output)
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


def run_geva(file_name, Ne, mut_rate, rec_rate=None, genetic_map_path=None):
    """
    Perform GEVA age estimation on a given vcf
    """
    if genetic_map_path is None:
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
    else:
        subprocess.check_output(
            [
                geva_executable,
                "--out",
                file_name,
                "--map",
                genetic_map_path,
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


def papuans_10j19():
    id = "PapuansOutOfAfrica_10J19"
    description = "Out-of-Africa with archaic admixture into Papuans"
    long_description = """
        A ten population model of out-of-Africa, including two pulses of
        Denisovan admixture into Papuans, and several pulses of Neandertal
        admixture into non-Africans.
        Most parameters are from Jacobs et al. (2019), Table S5 and Figure S5.
        This model is an extension of one from Malaspinas et al. (2016), thus
        some parameters are inherited from there.
        """

    # sampling times
    T_DenA = 2203
    T_NeaA = 3793

    populations = [
        # humans
        stdpopsim.Population(id="YRI", description="1000 Genomes YRI (Yorubans)"),
        stdpopsim.Population(
            id="CEU",
            description="1000 Genomes CEU (Utah Residents (CEPH) with Northern and Western European Ancestry",
        ),
        stdpopsim.Population(
            id="CHB", description="1000 Genomes CHB (Han Chinese in Beijing, China)"
        ),
        stdpopsim.Population("Papuan", "Papuans from Indonesia and New Guinea"),
        # archaics
        stdpopsim.Population(
            "DenA", "Altai Denisovan (sampling) lineage", sampling_time=T_DenA
        ),
        stdpopsim.Population(
            "NeaA", "Altai Neandertal (sampling) lineage", sampling_time=T_NeaA
        ),
        stdpopsim.Population(
            "Den1", "Denisovan D1 (introgressing) lineage", sampling_time=None
        ),
        stdpopsim.Population(
            "Den2", "Denisovan D2 (introgressing) lineage", sampling_time=None
        ),
        stdpopsim.Population(
            "Nea1", "Neandertal N1 (introgressing) lineage", sampling_time=1725
        ),
        stdpopsim.Population("Ghost", "Out-of-Africa lineage", sampling_time=None),
    ]
    pop = {p.id: i for i, p in enumerate(populations)}

    citations = [
        stdpopsim.Citation(
            author="Jacobs et al.",
            year=2019,
            doi="https://doi.org/10.1016/j.cell.2019.02.035",
            reasons={stdpopsim.CiteReason.DEM_MODEL},
        ),
        stdpopsim.Citation(
            author="Malaspinas et al.",
            year=2016,
            doi="https://doi.org/10.1038/nature18299",
            reasons={stdpopsim.CiteReason.DEM_MODEL},
        ),
    ]

    # Inherited from Malaspinas et al., which gives the following refs:
    generation_time = 29  # Fenner 2005
    # mutation_rate = 1.25e-8  # per gen per site, Scally & Durbin 2012

    N_YRI = 48433
    N_Ghost = 8516
    N_CEU = 6962
    N_CHB = 9025
    N_Papuan = 8834

    N_DenA = 5083
    N_Den1 = 13249
    N_Den2 = 13249
    N_NeaA = 826
    N_Nea1 = 13249

    pop_meta = {p.id: p.asdict() for p in populations}
    population_configurations = [
        msprime.PopulationConfiguration(initial_size=N_YRI, metadata=pop_meta["YRI"]),
        msprime.PopulationConfiguration(initial_size=N_CEU, metadata=pop_meta["CEU"]),
        msprime.PopulationConfiguration(initial_size=N_CHB, metadata=pop_meta["CHB"]),
        msprime.PopulationConfiguration(
            initial_size=N_Papuan, metadata=pop_meta["Papuan"]
        ),
        msprime.PopulationConfiguration(initial_size=N_DenA, metadata=pop_meta["DenA"]),
        msprime.PopulationConfiguration(initial_size=N_NeaA, metadata=pop_meta["NeaA"]),
        msprime.PopulationConfiguration(initial_size=N_Den1, metadata=pop_meta["Den1"]),
        msprime.PopulationConfiguration(initial_size=N_Den2, metadata=pop_meta["Den2"]),
        msprime.PopulationConfiguration(initial_size=N_Nea1, metadata=pop_meta["Nea1"]),
        msprime.PopulationConfiguration(
            initial_size=N_Ghost, metadata=pop_meta["Ghost"]
        ),
    ]
    assert len(populations) == len(population_configurations)

    # initial migrations
    m_Ghost_Afr = 1.79e-4
    m_Ghost_EU = 4.42e-4
    m_EU_AS = 3.14e-5
    m_AS_Papuan = 5.72e-5
    # older migrations
    m_Eurasian_Papuan = 5.72e-4
    m_Eurasian_Ghost = 4.42e-4

    migration_matrix = [[0] * len(populations) for _ in range(len(populations))]
    migration_matrix[pop["Ghost"]][pop["YRI"]] = m_Ghost_Afr
    migration_matrix[pop["YRI"]][pop["Ghost"]] = m_Ghost_Afr
    migration_matrix[pop["Ghost"]][pop["CEU"]] = m_Ghost_EU
    migration_matrix[pop["CEU"]][pop["Ghost"]] = m_Ghost_EU
    migration_matrix[pop["CEU"]][pop["CHB"]] = m_EU_AS
    migration_matrix[pop["CHB"]][pop["CEU"]] = m_EU_AS
    migration_matrix[pop["CHB"]][pop["Papuan"]] = m_AS_Papuan
    migration_matrix[pop["Papuan"]][pop["CHB"]] = m_AS_Papuan

    # splits
    T_EU_AS_split = 1293
    T_Eurasian_Ghost_split = 1758
    T_Papuan_Ghost_split = 1784
    T_Ghost_Afr_split = 2218
    T_NeaA_Nea1_split = 3375
    T_DenA_Den1_split = 9750
    T_DenA_Den2_split = 12500
    T_Den_Nea_split = 15090
    T_Afr_Archaic_split = 20225

    # bottlenecks
    Tb_Eurasia = 1659
    Tb_Papua = 1685
    Tb_Ghost = 2119
    Nb_Eurasia = 2231
    Nb_Papua = 243
    Nb_Ghost = 1394

    # internal branches
    N_EU_AS = 12971
    N_Ghost_Afr = 41563
    N_NeaA_Nea1 = 13249
    N_Den_Anc = 100  # S10.i p. 31/45
    N_DenA_Den1 = N_Den_Anc
    N_DenA_Den2 = N_Den_Anc
    N_Den_Nea = 13249
    N_Afr_Archaic = 32671

    # admixture pulses
    m_Den_Papuan = 0.04
    p1 = 0.55  # S10.i p. 31/45
    m_Den1_Papuan = p1 * m_Den_Papuan
    m_Den2_Papuan = (1 - p1) * m_Den_Papuan
    m_Nea1_Ghost = 0.024
    m_Nea1_Eurasian = 0.011
    m_Nea1_Papuan = 0.002
    m_Nea1_AS = 0.002

    T_Nea1_Ghost_mig = 1853
    T_Nea1_Eurasian_mig = 1566
    T_Nea1_Papuan_mig = 1412
    T_Nea1_AS_mig = 883
    # Fig. 4B, and S10.h p. 30/45
    T_Den1_Papuan_mig = 29.8e3 / generation_time
    T_Den2_Papuan_mig = 45.7e3 / generation_time

    demographic_events = [
        # human lineage splits
        msprime.MassMigration(
            time=T_EU_AS_split, source=pop["CEU"], destination=pop["CHB"]
        ),
        msprime.PopulationParametersChange(
            time=T_EU_AS_split, initial_size=N_EU_AS, population_id=pop["CHB"]
        ),
        msprime.MassMigration(
            time=T_Eurasian_Ghost_split, source=pop["CHB"], destination=pop["Ghost"]
        ),
        msprime.MassMigration(
            time=T_Papuan_Ghost_split, source=pop["Papuan"], destination=pop["Ghost"]
        ),
        msprime.MassMigration(
            time=T_Ghost_Afr_split, source=pop["Ghost"], destination=pop["YRI"]
        ),
        msprime.PopulationParametersChange(
            time=T_Ghost_Afr_split, initial_size=N_Ghost_Afr, population_id=pop["YRI"]
        ),
        # archaic lineage splits
        msprime.MassMigration(
            time=T_DenA_Den1_split, source=pop["Den1"], destination=pop["DenA"]
        ),
        msprime.PopulationParametersChange(
            time=T_DenA_Den1_split, initial_size=N_DenA_Den1, population_id=pop["DenA"]
        ),
        msprime.MassMigration(
            time=T_DenA_Den2_split, source=pop["Den2"], destination=pop["DenA"]
        ),
        msprime.PopulationParametersChange(
            time=T_DenA_Den2_split, initial_size=N_DenA_Den2, population_id=pop["DenA"]
        ),
        msprime.MassMigration(
            time=T_NeaA_Nea1_split, source=pop["Nea1"], destination=pop["NeaA"]
        ),
        msprime.PopulationParametersChange(
            time=T_NeaA_Nea1_split, initial_size=N_NeaA_Nea1, population_id=pop["NeaA"]
        ),
        msprime.MassMigration(
            time=T_Den_Nea_split, source=pop["NeaA"], destination=pop["DenA"]
        ),
        msprime.PopulationParametersChange(
            time=T_Den_Nea_split, initial_size=N_Den_Nea, population_id=pop["DenA"]
        ),
        msprime.MassMigration(
            time=T_Afr_Archaic_split, source=pop["DenA"], destination=pop["YRI"]
        ),
        msprime.PopulationParametersChange(
            time=T_Afr_Archaic_split,
            initial_size=N_Afr_Archaic,
            population_id=pop["YRI"],
        ),
        # bottlenecks
        msprime.PopulationParametersChange(
            time=Tb_Eurasia, initial_size=Nb_Eurasia, population_id=pop["CHB"]
        ),
        msprime.PopulationParametersChange(
            time=Tb_Papua, initial_size=Nb_Papua, population_id=pop["Papuan"]
        ),
        msprime.PopulationParametersChange(
            time=Tb_Ghost, initial_size=Nb_Ghost, population_id=pop["Ghost"]
        ),
        # migration changes
        msprime.MigrationRateChange(
            time=T_EU_AS_split, rate=0, matrix_index=(pop["CHB"], pop["CEU"])
        ),
        msprime.MigrationRateChange(
            time=T_EU_AS_split, rate=0, matrix_index=(pop["CEU"], pop["CHB"])
        ),
        msprime.MigrationRateChange(
            time=T_EU_AS_split, rate=0, matrix_index=(pop["Papuan"], pop["CHB"])
        ),
        msprime.MigrationRateChange(
            time=T_EU_AS_split, rate=0, matrix_index=(pop["CHB"], pop["Papuan"])
        ),
        msprime.MigrationRateChange(
            time=T_EU_AS_split, rate=0, matrix_index=(pop["Ghost"], pop["CEU"])
        ),
        msprime.MigrationRateChange(
            time=T_EU_AS_split, rate=0, matrix_index=(pop["CEU"], pop["Ghost"])
        ),
        msprime.MigrationRateChange(
            time=T_EU_AS_split,
            rate=m_Eurasian_Papuan,
            matrix_index=(pop["CHB"], pop["Papuan"]),
        ),
        msprime.MigrationRateChange(
            time=T_EU_AS_split,
            rate=m_Eurasian_Papuan,
            matrix_index=(pop["Papuan"], pop["CHB"]),
        ),
        msprime.MigrationRateChange(
            time=T_EU_AS_split,
            rate=m_Eurasian_Ghost,
            matrix_index=(pop["CHB"], pop["Ghost"]),
        ),
        msprime.MigrationRateChange(
            time=T_EU_AS_split,
            rate=m_Eurasian_Ghost,
            matrix_index=(pop["Ghost"], pop["CHB"]),
        ),
        # all migrations off
        msprime.MigrationRateChange(time=Tb_Eurasia, rate=0),
        # admixture pulses
        msprime.MassMigration(
            time=T_Den1_Papuan_mig,
            proportion=m_Den1_Papuan,
            source=pop["Papuan"],
            destination=pop["Den1"],
        ),
        msprime.MassMigration(
            time=T_Den2_Papuan_mig,
            proportion=m_Den2_Papuan,
            source=pop["Papuan"],
            destination=pop["Den2"],
        ),
        msprime.MassMigration(
            time=T_Nea1_Ghost_mig,
            proportion=m_Nea1_Ghost,
            source=pop["Ghost"],
            destination=pop["Nea1"],
        ),
        msprime.MassMigration(
            time=T_Nea1_Eurasian_mig,
            proportion=m_Nea1_Eurasian,
            source=pop["CHB"],
            destination=pop["Nea1"],
        ),
        msprime.MassMigration(
            time=T_Nea1_Papuan_mig,
            proportion=m_Nea1_Papuan,
            source=pop["Papuan"],
            destination=pop["Nea1"],
        ),
        msprime.MassMigration(
            time=T_Nea1_AS_mig,
            proportion=m_Nea1_AS,
            source=pop["CHB"],
            destination=pop["Nea1"],
        ),
    ]

    demographic_events.sort(key=lambda x: x.time)

    return stdpopsim.DemographicModel(
        id=id,
        description=description,
        long_description=long_description,
        populations=populations,
        citations=citations,
        generation_time=generation_time,
        population_configurations=population_configurations,
        migration_matrix=migration_matrix,
        demographic_events=demographic_events,
    )

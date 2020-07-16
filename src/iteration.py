import argparse
import csv
import numpy as np
import json
import pickle
import logging
import pandas as pd
import re
from tqdm import tqdm
import time

import msprime
import tsinfer
import tskit
import stdpopsim

import utility as util
import evaluation
import constants

import tsdate


"""
Infer tree sequences from modern and ancient samples.
Ancient sample ages MUST be specified in the SampleData file using the individuals_time
array. Ancient sample ages should be specified in years. 
Input is a sampledata file with moderns (and optionally ancients).
"""


def iter_infer(
    samples,
    Ne,
    mutation_rate,
    mismatch_rate=None,
    precision=None,
    recombination_rate=None,
    num_threads=1,
    output_fn=None,
    prune_tree=False,
    ignore_oldest_root=False,
    modern_only_first_pass=False,
    chrom=None,
    return_trees=False,
    inferred_ts=None,
    progress=False,
):
    """
    Runs all steps in iterative approach.
    Input is a sampledata file (optionally with ancient samples).
    """

    print(
        "Starting iterative approach with {} samples, {} sites, using {} mismatch_rate," 
        "{} average rho, and {} precision".format(
            samples.num_samples, samples.num_sites, mismatch_rate, np.quantile(recombination_rate, 0.5), precision
        )
    )
    # Step 1: tsinfer first pass
    start = time.time()
    if inferred_ts is None:
        inferred_ts = tsinfer_first_pass(
            samples,
            output_fn=output_fn,
            recombination_rate=recombination_rate,
            mismatch_rate=mismatch_rate,
            precision=precision,
            num_threads=num_threads,
            progress=progress,
        )
    else:
        inferred_ts = tskit.load(inferred_ts)
    end = time.time()
    print("Step 1, Inferring Tree Sequence, done in {} minutes".format((end - start) / 60)
    )


    # Snip centromere and telomeres from inferred_ts if chr in title of sampledata file
    if chrom is not None:
        inferred_ts, centromere = run_snip_centromere_telomeres(inferred_ts, chrom)

    # Step 2: tsdate first pass
    start = time.time()
    tsdate_ages, modern_inferred_ts = tsdate_first_pass(
        inferred_ts,
        samples,
        Ne,
        mutation_rate,
        output_fn=output_fn,
        num_threads=num_threads,
        progress=progress,
    )
    end = time.time()
    print("Step 2, Dating Tree Sequence, {} minutes".format((end - start) / 60))

    # Step 3: Add dates to sampledata file
    start = time.time()
    samples_constrained, constr_mut_pos, constrained_ages = get_dated_sampledata(
        samples, tsdate_ages, modern_inferred_ts, output_fn=output_fn
    )
    end = time.time()
    print(
        "Step 3, Adding times to sampledata file, {} minutes".format((end - start) / 60)
    )

    # Step 4: tsinfer second pass
    start = time.time()
    reinferred_ts = tsinfer_second_pass(
        samples_constrained,
        recombination_rate=recombination_rate,
        mismatch_rate=mismatch_rate,
        precision=precision,
        output_fn=output_fn,
        num_threads=num_threads,
        progress=progress,
    )
    end = time.time()
    print("Step 4, Re-Inferring Tree Sequence, {} minutes".format((end - start) / 60))
    print(
        "Re-Inferred ts has {} nodes, {} edges, {} mutations and {} trees".format(
            reinferred_ts.num_nodes,
            reinferred_ts.num_edges,
            reinferred_ts.num_mutations,
            reinferred_ts.num_trees,
        )
    )

    # Snip centromere and telomeres from inferred_ts if chr in title of sampledata file
    if chrom is not None:
        reinferred_ts, centromere = run_snip_centromere_telomeres(reinferred_ts, chrom)

    # Step 5: tsdate second pass
    start = time.time()
    iter_dates, modern_reinferred_ts = tsdate_second_pass(
        reinferred_ts,
        samples_constrained,
        Ne,
        mutation_rate,
        output_fn=output_fn,
        constr_sites=constr_mut_pos,
        adjust_priors=True,
        num_threads=num_threads,
        progress=progress,
    )
    end = time.time()
    print(
        "Step 5, Dating Re-Inferred Tree Sequence, {} minutes".format(
            (end - start) / 60
        )
    )
    if return_trees:
        inferred_dated_ts = get_dated_ts(modern_inferred_ts, tsdate_ages, Ne, 1e-6)
        reinferred_dated_ts = get_dated_ts(modern_reinferred_ts, iter_dates, Ne, 1e-6)
        return (
            modern_inferred_ts,
            tsdate_ages,
            constrained_ages,
            modern_reinferred_ts,
            iter_dates,
            inferred_dated_ts,
            reinferred_dated_ts,
        )
    else:
        return (
            modern_inferred_ts,
            tsdate_ages,
            constrained_ages,
            modern_reinferred_ts,
            iter_dates,
        )


def tsinfer_first_pass(
    samples,
    recombination_rate=None,
    mismatch_rate=None,
    precision=None,
    output_fn=None,
    modern_only=False,
    num_threads=1,
    progress=False,
):
    """
    Infer tree sequence topology with modern and ancient samples.
    Then simplify so tree sequence only contains moderns.
    """
    if progress:
        progress = tsinfer.cli.ProgressMonitor(1, 0, 0, 0, 1)
        inferred_ts = tsinfer.infer(
            samples,
            recombination_rate=recombination_rate,
            mismatch_rate=mismatch_rate,
            precision=precision,
            simplify=False,
            num_threads=num_threads,
            progress_monitor=progress,
        )
    else:
        inferred_ts = tsinfer.infer(
            samples,
            recombination_rate=recombination_rate,
            mismatch_rate=mismatch_rate,
            precision=precision,
            simplify=False,
            num_threads=num_threads,
        )

    if output_fn is not None:
        inferred_ts.dump(output_fn + ".tsinferred.trees")
    #print("Starting with av rho {} (mean {}, median {},  nonzero min {},  2.5% quantile {}) precision {}").format( 
    print(
        "Inferred ts has {} nodes, {} edges, {} mutations and {} trees".format(
            inferred_ts.num_nodes,
            inferred_ts.num_edges,
            inferred_ts.num_mutations,
            inferred_ts.num_trees,
        )
    )
    return inferred_ts


def tsdate_first_pass(
    inferred_ts,
    samples,
    Ne,
    mutation_rate,
    output_fn=None,
    method="inside_outside",
    num_threads=1,
    progress=False,
):
    """
    Date the inferred tree sequence, simplyging any ancient samples out before dating.
    """
    ancient_indivs = np.where(samples.individuals_time[:] != 0)[0]
    ancient_nodes = np.where(
        np.isin(inferred_ts.tables.nodes.individual[:], ancient_indivs)
    )[0]
    # Assert that all ancient nodes are samples
    assert np.all(np.isin(ancient_nodes, inferred_ts.samples()))
    modern_inferred_ts = inferred_ts.simplify(
        samples=inferred_ts.samples()[~np.isin(inferred_ts.samples(), ancient_nodes)]
    )
    tsdate_ages = tsdate.get_dates(
        modern_inferred_ts,
        Ne,
        mutation_rate,
        method=method,
        num_threads=num_threads,
        progress=progress,
    )

    tsdate_ages = tsdate_ages[0] * 2 * Ne
    tsdate_ages_df = util.get_mut_pos_df(
        modern_inferred_ts, "FirstPassAges", tsdate_ages
    )
    if output_fn is not None:
        tsdate_ages_df.to_csv(output_fn + ".tsdatefirstpass.csv")
        pickle.dump(tsdate_ages, open(output_fn + ".tsdatefirstpass.dates.p", "wb"))
    logging.debug(
        "STEP TWO: Dated inferred tree sequence with {} mutations.".format(
            inferred_ts.num_mutations
        )
    )
    return tsdate_ages, modern_inferred_ts


def get_dated_sampledata(
    sampledata, tsdate_ages, inferred_ts, output_fn=None, centromere=None
):
    """
    Constrain sites in sampledata by date estimates and ancient ages.
    Takes sampledata file, which either has modern and ancient samples OR only moderns
    tsdate_ages is the per-node date estimate for each node in inferred_ts.
    Returns dated sampledata with only moderns.
    """
    # Make dataframe of mutations present in both modern and ancient samples
    # First make ancient mut df from ancient sampledata file
    sampledata_ages = sampledata.individuals_time[:][sampledata.samples_individual[:]]
    ancient_samples_bool = sampledata_ages != 0
    ancient_ages = sampledata_ages[ancient_samples_bool] / constants.GENERATION_TIME
    ancient_genotypes = sampledata.sites_genotypes[:][:, ancient_samples_bool]
    ancient_sites_bool = np.any(ancient_genotypes == 1, axis=1)
    ancient_genotypes = ancient_genotypes[ancient_sites_bool]
    ancient_positions = sampledata.sites_position[:][ancient_sites_bool]
    ancient_alleles = sampledata.sites_alleles[:][ancient_sites_bool]
    ancient_mutations = [
        (pos, np.max(ancient_ages[geno == 1]), alleles[0], alleles[1])
        for pos, geno, alleles in zip(
            ancient_positions, ancient_genotypes, ancient_alleles
        )
    ]
    ancient_mut_df = pd.DataFrame(
        ancient_mutations,
        columns=["Position", "Ancient Bound", "Reference Allele", "Alternative Allele"],
    )
    ancient_mut_df = ancient_mut_df.astype(
        {"Position": "float64", "Ancient Bound": "float64"}
    )

    mutation_table = inferred_ts.tables.mutations
    sites_table = inferred_ts.tables.sites
    modern_mut_df = pd.DataFrame(
        {
            "Position": sites_table.position[mutation_table.site],
            "Estimated Age": tsdate_ages[mutation_table.node],
            "Ancestral Allele": np.array(
                tskit.unpack_strings(
                    sites_table.ancestral_state, sites_table.ancestral_state_offset
                )
            )[mutation_table.site],
            "Derived Allele": tskit.unpack_strings(
                mutation_table.derived_state, mutation_table.derived_state_offset
            ),
        }
    )
    modern_mut_df = modern_mut_df.astype(
        {"Position": "float64", "Estimated Age": "float64"}
    )
    # Now take the oldest mutation at each site
    sort_upper_bound = modern_mut_df.sort_values(
        by=["Estimated Age"], ascending=False, kind="mergesort"
    )
    # Ignore back mutations
    sort_upper_bound = sort_upper_bound[
        sort_upper_bound["Derived Allele"] != sort_upper_bound["Ancestral Allele"]
    ]
    modern_mut_df = sort_upper_bound.groupby("Position", as_index=False).first()
    # Account for snipped centromere
    sampledata_pos = sampledata.sites_position[:]
    if centromere is not None:
        keep_sites = np.logical_and(
            sampledata_pos < centromere[0], sampledata_pos > centromere[1]
        )
        sampledata_alleles = sampledata.sites_alleles[:][keep_sites]
    merged = pd.merge(
        modern_mut_df,
        ancient_mut_df,
        how="left",
        left_on=["Position", "Ancestral Allele", "Derived Allele"],
        right_on=["Position", "Reference Allele", "Alternative Allele"],
    )
    merged["Constrained Age"] = np.fmax(
        merged["Estimated Age"], merged["Ancient Bound"]
    )
    constr_where = np.where(merged["Ancient Bound"] > merged["Estimated Age"])[0]
    logging.debug(
        "STEP THREE: Constraining age estimates with ancient samples. {} mutations were constrained".format(
            len(constr_where)
        )
    )

    other_sites = ~np.isin(sampledata.sites_position[:], merged["Position"])
    if np.sum(other_sites) != sampledata.num_sites:
        sampledata = sampledata.delete(sites=other_sites)
    if output_fn is not None:
        sampledata_copy = sampledata.copy(output_fn + ".constrained.samples")
    else:
        sampledata_copy = sampledata.copy()
    # Convert age in years back to generations
    which_sites = np.isin(merged["Position"], sampledata_copy.sites_position[:])
    sampledata_copy.sites_time[:] = (merged["Constrained Age"][which_sites]) + 1
    sampledata_copy.finalise()
    constr_index = np.where(
        merged["Ancient Bound"][which_sites] >= merged["Estimated Age"][which_sites]
    )[0]
    constr_mut_pos = dict(
        zip(
            merged["Position"][which_sites][constr_index],
            merged["Constrained Age"][which_sites][constr_index],
        )
    )
    # At sites with multiple mutations, only constrain oldest times
    constrained_ages = np.copy(tsdate_ages)
    for site in inferred_ts.sites():
        oldest_mut_age = 0
        oldest_mut_node = 0
        for mut in site.mutations:
            if constrained_ages[mut.node] > oldest_mut_age:
                oldest_mut_age = constrained_ages[mut.node]
                oldest_mut_node = mut.node
        constrained_ages[mut.node] = merged["Constrained Age"][site.id]

    return sampledata_copy, constr_mut_pos, constrained_ages


def ancients_as_ancestors(sample_data, ancestor_data, output_fn=None, progress=False):
    """
    Insert ancient samples in sample_data file as ancestors in ancestor_data.
    """

    def add_ancient_ancestor(ancestors, ancient_age, ancient_haplotype):
        ancestors.add_ancestor(
            0, ancestor_data.num_sites, ancient_age, [], ancient_haplotype
        )
        assert ancient_haplotype.shape[0] == (ancestors.num_sites)

    sample_ages = sample_data.individuals_time[:][sample_data.samples_individual[:]]
    sample_metadata = sample_data.individuals_metadata[:][
        sample_data.samples_individual[:]
    ]
    ancient_ages = sample_ages[sample_ages != 0] / constants.GENERATION_TIME
    ancient_metadata = sample_metadata[sample_ages != 0]
    ancient_samples = np.where(sample_ages != 0)[0]
    # Ancients could be out of order timewise, argsort them
    argsort_ancients = np.argsort(ancient_ages)[::-1]
    add_ancient_indices = np.searchsorted(
        -ancestor_data.ancestors_time[:], -ancient_ages[argsort_ancients]
    )
    ancient_haplos = list()

    for haplo in sample_data.haplotypes():
        if haplo[0] in ancient_samples:
            ancient_haplos.append(haplo[1][sample_data.sites_inference[:]])
    if output_fn is not None:
        output_fn = output_fn + ".ancients_added.ancestors"
    with tsinfer.AncestorData(
        sample_data=sample_data, path=output_fn
    ) as ancestor_data_ancients:
        index = 0
        added_ancients_indices = list()
        added_ancients_metadata = list()
        for cur_index, ancestor in tqdm(
            enumerate(ancestor_data.ancestors()),
            total=ancestor_data.num_ancestors,
            disable=not progress,
            desc="Add ancient ancestors",
        ):
            if cur_index in add_ancient_indices:
                for ancient_index in np.where(cur_index == add_ancient_indices)[0]:
                    # Add ancient sample as ancestor at appropriate time
                    # Use correct index into ancients (so they are sorted)
                    ancient_index = argsort_ancients[ancient_index]
                    add_ancient_ancestor(
                        ancestor_data_ancients,
                        ancient_ages[ancient_index],
                        ancient_haplos[ancient_index],
                    )
                    added_ancients_indices.append(index)
                    added_ancients_metadata.append(ancient_metadata[ancient_index])
                    index += 1
            ancestor_data_ancients.add_ancestor(
                ancestor.start,
                ancestor.end,
                ancestor.time,
                ancestor.focal_sites,
                ancestor.haplotype,
            )
            index += 1
        if cur_index + 1 in add_ancient_indices:
            for ancient_index in np.where(cur_index + 1 == add_ancient_indices)[0]:
                ancient_index = argsort_ancients[ancient_index]
                add_ancient_ancestor(
                    ancestor_data_ancients,
                    ancient_ages[ancient_index],
                    ancient_haplos[ancient_index],
                )
                added_ancients_indices.append(index)
                added_ancients_metadata.append(ancient_metadata[ancient_index])
                index += 1
    assert ancestor_data_ancients.num_ancestors == (
        ancestor_data.num_ancestors + len(ancient_samples)
    ), (
        ancestor_data_ancients.num_ancestors,
        ancestor_data.num_ancestors,
        len(ancient_samples),
    )
    ancestor_times = ancestor_data_ancients.ancestors_time[:]
    assert all(x >= y for x, y in zip(ancestor_times, ancestor_times[1:])), (
        ancestor_times,
        np.diff(ancestor_data.ancestors_time[:]),
        np.diff(ancestor_times),
    )

    return (
        ancestor_data_ancients,
        np.array(added_ancients_indices),
        added_ancients_metadata,
    )


def tsinfer_second_pass(
    samples_constrained,
    recombination_rate=None,
    mismatch_rate=None,
    precision=None,
    output_fn=None,
    num_threads=1,
    progress=False,
):
    """
    When reinfering the tree sequence, we now use the ancient samples as ancestors
    """

    def add_ancestors_flags(ts, added_ancients_indices, added_ancients_metadata):
        for index, node in enumerate(ts.tables.nodes):
            if node.flags & tsinfer.constants.NODE_IS_PC_ANCESTOR:
                added_ancients_indices[added_ancients_indices > index] += 1
        tables = ts.dump_tables()
        flag_array = tables.nodes.flags.copy()
        metadata_array = tskit.unpack_bytes(
            tables.nodes.metadata, tables.nodes.metadata_offset
        )
        for index, ancestor_index in enumerate(added_ancients_indices):
            flag_array[ancestor_index] = np.bitwise_or(
                flag_array[ancestor_index],
                np.bitwise_or(tsinfer.NODE_IS_SAMPLE_ANCESTOR, tskit.NODE_IS_SAMPLE),
            )
            metadata_array[ancestor_index] = json.dumps(
                added_ancients_metadata[index]
            ).encode()
        md, md_offset = tskit.pack_bytes(metadata_array)
        tables.nodes.set_columns(
            flags=flag_array,
            time=tables.nodes.time,
            population=tables.nodes.population,
            individual=tables.nodes.individual,
            metadata=md,
            metadata_offset=md_offset,
        )
        return tables.tree_sequence()

    if progress is True:
        progress = tsinfer.cli.ProgressMonitor(1, 0, 0, 0, 1)
    else:
        progress = None
    # We bin the sites time for efficiency
    binned_sd = bin_sampledata(samples_constrained, output_fn)
    ancestors_data = tsinfer.generate_ancestors(
        binned_sd, num_threads=num_threads, progress_monitor=progress
    )
    if output_fn is not None:
        output_str = output_fn + ".ancestors"
    else:
        output_str = None
    # If there are ancient samples, add them as ancestors
    if np.any(samples_constrained.individuals_time[:] > 0):
        ancients_present = True
    else:
        ancients_present = False
    if ancients_present:
        (
            ancestors_data,
            added_ancient_indices,
            added_ancient_metadata,
        ) = ancients_as_ancestors(
            binned_sd, ancestors_data, output_fn, progress=progress
        )
    extra_params = dict(
        num_threads=num_threads,
        recombination_rate=recombination_rate,
        precision=precision,
        progress_monitor=progress,
    )
    inferred_anc_ts = tsinfer.match_ancestors(
        binned_sd,
        ancestors_data,
        mismatch_rate=mismatch_rate,
        path_compression=False,
        **extra_params,
    )
    if ancients_present:
        inferred_anc_ts = add_ancestors_flags(
            inferred_anc_ts, added_ancient_indices, added_ancient_metadata
        )
    if output_fn is not None:
        inferred_anc_ts.dump(output_fn + ".iter.tsinferred.atrees")
#    if ancients_present:
#        ancient_samples = np.where(binned_sd.individuals_time[:][binned_sd.samples_individual] != 0)[0]
#        print("Matching Samples without the following ancient samples: {}".format(ancient_samples))
#        modern_samples_constrained = binned_sd.delete(samples=ancient_samples)
#    else:
    modern_samples_constrained = binned_sd
    iter_infer = tsinfer.match_samples(
        modern_samples_constrained,
        inferred_anc_ts,
        mismatch_rate=mismatch_rate,
        path_compression=False,
        simplify=False,
        **extra_params,
    )
    if output_fn is not None:
        iter_infer.dump(output_fn + ".iter.tsinferred.trees")

    logging.debug(
        "STEP FOUR: Reinferred tree sequence with {} modern samples and {} ancients.".format(
            np.sum(modern_samples_constrained.individuals_time[:] == 0),
            np.sum(modern_samples_constrained.individuals_time[:] != 0),
        )
    )
    return iter_infer


def tsdate_second_pass(
    inferred_ts,
    samples,
    Ne,
    mut_rate,
    output_fn=None,
    constr_sites=None,
    adjust_priors=True,
    num_threads=1,
    progress=False,
):
    """
    Input is tree sequence with modern and ancient samples.
    Simplify so tree sequence only contains moderns and date.
    """
    modern_samples = np.where(
        samples.individuals_time[:][samples.samples_individual] == 0
    )[0]
    modern_inferred_ts = inferred_ts.simplify(samples=modern_samples)
    if modern_inferred_ts.num_samples > 1000:
        approximate_priors = True
    else:
        approximate_priors = False
    priors = tsdate.build_prior_grid(
        modern_inferred_ts, progress=progress, approximate_priors=approximate_priors
    )
    # If there are no ancient samples, don't adjust priors
    if np.all(samples.individuals_time[:] == 0):
        adjust_priors = False
    if adjust_priors and constr_sites is not None:
        inferred_times = modern_inferred_ts.tables.nodes.time[:]
        for mut_pos, limit in constr_sites.items():
            constrained_mutations_nodes = modern_inferred_ts.tables.mutations.node[
                np.where(
                    modern_inferred_ts.tables.sites.position[
                        modern_inferred_ts.tables.mutations.site
                    ]
                    == mut_pos
                )[0]
            ]
            # Only constrain the oldest mutation at this site
            oldest_mut_age = 0
            oldest_mut_node = 0
            for mut_node in constrained_mutations_nodes:
                if inferred_times[mut_node] > oldest_mut_age:
                    oldest_mut_age = inferred_times[mut_node]
                    oldest_mut_node = mut_node

            # Only constrain mutations which are not above sample nodes
            if inferred_times[mut_node] > 0:
                priors.grid_data[
                    mut_node - modern_inferred_ts.num_samples,
                    : (np.abs(priors.timepoints * 2 * Ne - limit)).argmin(),
                ] = 0
        added_ancestors = np.where(
            modern_inferred_ts.tables.nodes.flags & tsinfer.NODE_IS_SAMPLE_ANCESTOR
        )[0]
        for added_anc in added_ancestors:
            ancient_time = modern_inferred_ts.tables.nodes.time[added_anc]
            priors.grid_data[
                added_anc - modern_inferred_ts.num_samples,
                (np.abs(priors.timepoints * 2 * Ne - ancient_time)).argmin(),
            ] = 1
    iter_dates = tsdate.get_dates(
        modern_inferred_ts,
        Ne,
        mut_rate,
        priors=priors,
        num_threads=num_threads,
        progress=progress,
    )
    tsdate_ages_df = util.get_mut_pos_df(
        modern_inferred_ts, "SecondPassDates", iter_dates[0] * 2 * Ne
    )
    if output_fn is not None:
        tsdate_ages_df.to_csv(output_fn + ".tsdatesecondpass.csv")
        pickle.dump(iter_dates[0] * 2 * Ne, open(output_fn + ".tsdatesecondpass.dates.p", "wb"))
    # Ensure that all dates are greater than the timepoint closest to the lower
    # bound ancient time constraint
    if constr_sites is not None:
        constr_site_timepoint = {
            pos: (priors.timepoints * 2 * Ne)[
                (np.abs(priors.timepoints * 2 * Ne - time)).argmin()
            ]
            for pos, time in constr_sites.items()
        }
        # Remove singletons
        tsdate_ages_df = tsdate_ages_df.loc[iter_dates[0][tsdate_ages_df["Node"]] > 0]
        tsdate_ages_df.to_csv(output_fn + ".tsdatesecondpass.nosingletons.csv")
        assert np.all(
            [
                val >= constr_site_timepoint[pos]
                for pos, val in zip(
                    tsdate_ages_df.index, tsdate_ages_df["SecondPassDates"]
                )
                if pos in constr_site_timepoint
            ]
        ), output_fn
    logging.debug(
        "STEP FIVE: Dated reinferred tree sequence with {} mutations.".format(
            modern_inferred_ts.num_mutations
        )
    )
    iter_dates = iter_dates[0] * 2 * Ne
    return iter_dates, modern_inferred_ts


def bin_sampledata(sampledata, output_fn=None):
    if output_fn is not None:
        sd = sampledata.copy(output_fn + ".binned.samples")
    else:
        sd = sampledata.copy()

    times = sd.sites_time[:]

    for j, variant in enumerate(sd.variants(inference_sites=True)):
        time = variant.site.time
        if time == tsinfer.constants.TIME_UNSPECIFIED:
            counts = tsinfer.formats.allele_counts(variant.genotypes)
            # Non-variable sites have no obvious freq-as-time values
            assert counts.known != counts.derived
            assert counts.known != counts.ancestral
            assert counts.known > 0
            # Time = freq of *all* derived alleles. Note that if n_alleles > 2 this
            # may not be sensible: https://github.com/tskit-dev/tsinfer/issues/228
            times[variant.site.id] = counts.derived / counts.known

    # Round times to the nearest 10, excluding times less than 5
    times[times > 5] = np.round(times[times > 5], -1)
    sd.sites_time[:] = times
    print(
        "Number of samples:",
        sd.num_samples,
        ". Number of discrete times:",
        len(np.unique(sd.sites_time[:])),
    )
    sd.finalise()
    return sd


def get_dated_ts(ts, dates, Ne, eps):
    """
    Simple wrapper to get dated tree sequence from unconstrained dates.
    NOTE: dates are assumed to be in generations.
    """
    constrained = tsdate.constrain_ages_topo(ts, dates, eps)
    tables = ts.dump_tables()
    tables.nodes.time = constrained
    tables.sort()
    dated_ts = tables.tree_sequence()
    return dated_ts


def setup_sample_file(args):
    """
    Return a Thousand Genomes Project sample data file, the
    corresponding recombination rate array, and a prefix to use for files
    """
    filename = args.sampledata_file
    map = args.genetic_map
    if not filename.endswith(".samples"):
        raise ValueError("Sample data file must end with '.samples'")
    sd = tsinfer.load(filename)
    inference_pos = sd.sites_position[:][sd.sites_inference[:]]

    match = re.search(r"(chr\d+)", filename)
    if match is not None:
        chrom = match.group(1)
    else:
        chrom = None
    if chrom or map is not None:
        if map is not None:
            print(f"Using {map} for the recombination map")
            chr_map = msprime.RecombinationMap.read_hapmap(map)
        else:
            print(f"Using {chrom} from HapMapII_GRCh37 for the recombination map")
            map = stdpopsim.get_species("HomSap").get_genetic_map(id="HapMapII_GRCh37")
            if not map.is_cached():
                map.download()
            chr_map = map.get_chromosome_map(chrom)
        inference_distances = physical_to_genetic(chr_map, inference_pos)
        d = np.diff(inference_distances)
        rho = np.concatenate(([0.0], d))
    else:
        inference_distances = inference_pos
        d = np.diff(inference_distances)
        rho = np.concatenate(([0.0], d / sd.sequence_length))

    return sd, rho, filename[: -len(".samples")], chrom


def physical_to_genetic(recombination_map, input_physical_positions):
    map_pos = recombination_map.get_positions()
    map_rates = recombination_map.get_rates()
    map_genetic_positions = np.insert(
        np.cumsum(np.diff(map_pos) * map_rates[:-1]), 0, 0
    )
    return np.interp(input_physical_positions, map_pos, map_genetic_positions)


def run_snip_centromere_telomeres(ts, chrom):
    with open("../all-data/centromeres.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["chrom"] == chrom:
                start = int(row["start"])
                end = int(row["end"])
                break
        else:
            raise ValueError("Did not find row")
    position = ts.tables.sites.position
    s_index = np.searchsorted(position, start)
    e_index = np.searchsorted(position, end)
    # We have a bunch of sites within the centromere. Get the largest
    # distance between these and call these the start and end. Probably
    # pointless having the centromere coordinates as input in the first place,
    # since we're just searching for the largest gap anyway. However, it can
    # be useful in UKBB, since it's perfectly possible that the largest
    # gap between sites isn't in the centromere.
    X = position[s_index : e_index + 1]
    if len(X) > 0:
        j = np.argmax(X[1:] - X[:-1])
        real_start = X[j] + 1
        real_end = X[j + 1]
        print(
            "Snipping inferred tree sequence. Centromere is {} to {}".format(
                real_start, real_end
            )
        )
        ts_snipped = ts.delete_intervals([[real_start, real_end]])
    else:
        real_start = start
        real_end = end
    first_pos = position[0]
    last_pos = position[-1] + 1
    print(
        "First position is {}, last position is {}. Total sequence length is {}".format(
            first_pos, last_pos, ts.get_sequence_length()
        )
    )
    ts_snipped = ts.keep_intervals([[first_pos, last_pos]])
    return ts_snipped, (real_start, real_end)


def main():
    parser = argparse.ArgumentParser(
        description="Process the human data and make data files for plotting."
    )
    parser.add_argument(
        "sampledata_file",
        type=str,
        help="Input sampledata to infer. \
            Be sure that any ancient samples are set to their estimated \
            age in years.",
    )
    parser.add_argument("output", type=str, help="Output dated tree sequence.")
    parser.add_argument("Ne", type=int, help="Estimated effective population size.")
    parser.add_argument("mutation_rate", type=float, help="Estimated mutation rate.")
    parser.add_argument(
        "-m",
        "--genetic-map",
        default=None,
        help="An alternative genetic map to be used for this analysis, in the format"
        "expected by msprime.RecombinationMap.read_hapmap",
    )
    parser.add_argument(
        "--mismatch-rate",
        type=float,
        default=0.1,
        help="The mismatch probability used in tsinfer,"
        " as a fraction of the median recombination probability between sites",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=None,
        help="The precision, as a number of decimal places, which will affect the speed"
        " of the matching algorithm (higher precision = lower speed). If None,"
        " calculate the smallest of the recombination rates or mismatch rates, and"
        " use the negative exponent of that number plus one. E.g. if the smallest"
        " recombination rate is 2.5e-6, use precision = 6+3 = 7",
    )

    parser.add_argument(
        "--inferred-ts",
        type=str,
        default=None,
        help="Path to inferred ts (first pass) \
            If user already has an inferred ts to perform iteration on. \
            This causes the first step to be skipped",
    )
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")

    args = parser.parse_args()
    sampledata, rho, prefix, chrom = setup_sample_file(args)
    av_rho = np.quantile(rho, 0.5)

    if rho is not None:
        av_rho = np.quantile(rho, 0.5)
        mismatch_rate = av_rho * args.mismatch_rate
    else:
        mismatch_rate = None


    if args.precision is None:
        # Smallest nonzero recombination rate
        min_rho = int(np.ceil(-np.min(np.log10(rho[rho > 0]))))
        # Smallest mean
        av_min = int(np.ceil(-np.log10(mismatch_rate)))
        precision = max(min_rho, av_min) + 3
    else:
        precision = args.precision

    print(np.quantile(rho, 0.5))
    iter_infer(
        sampledata,
        args.Ne,
        args.mutation_rate,
        output_fn=args.output,
        recombination_rate=rho,
        mismatch_rate=mismatch_rate,
        precision=precision,
        inferred_ts=args.inferred_ts,
        chrom=chrom,
        progress=args.progress,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()

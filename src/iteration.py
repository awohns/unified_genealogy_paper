import argparse
import numpy as np
import json
import logging
import pandas as pd
import sys

import tsinfer
import tskit

import utility as util
import evaluation

sys.path.insert(0, "/home/wilderwohns/tsdate_paper/tsdate/")
import tsdate

logging.basicConfig(filename="iteration.log", filemode="w", level=logging.DEBUG)

"""
Function to infer tree sequences from modern and ancient samples.
Constrains mutations with ancient samples.
"""


def iter_infer(
    samples,
    output_fn,
    Ne,
    mutation_rate,
    num_threads,
    prune_tree=False,
    ignore_oldest_root=False,
    modern_only_first_pass=False,
    progress=False,
):
    """
    Runs all steps in iterative approach.
    """

    # Step 1: tsinfer first pass
    inferred_ts, modern_only_samples = tsinfer_first_pass(
        samples, output_fn, modern_only_first_pass, num_threads, progress
    )

    # Step 2: tsdate first pass
    tsdate_ages = tsdate_first_pass(
        inferred_ts, Ne, mutation_rate, output_fn, num_threads, progress
    )

    # Step 3: Constrain with ancients
    samples_constrained, constrained_ages, constr_mut_pos = get_ancient_constraints(
        samples, modern_only_samples, tsdate_ages, inferred_ts, output_fn
    )

    # Step 4: tsinfer second pass
    reinferred_ts = tsinfer_second_pass(
        samples_constrained, samples, output_fn, num_threads, progress
    )

    # Step 5: tsdate second pass
    iter_dates = tsdate_second_pass(
        reinferred_ts,
        Ne,
        mutation_rate,
        output_fn,
        constr_sites=constr_mut_pos,
        adjust_priors=True,
        progress=progress,
    )

    return inferred_ts, tsdate_ages, constrained_ages, reinferred_ts, iter_dates


def tsinfer_first_pass(samples, output_fn, modern_only, num_threads, progress):
    """
    Infer tree sequence topology with modern and ancient samples.
    Then simplify so tree sequence only contains moderns.
    """
    modern_samples = np.where(samples.individuals_time[:] == 0)[0]
    num_modern_samples = modern_samples.shape[0]
    num_ancient_samples = np.sum(samples.individuals_time[:] != 0)
    if modern_only:
        samples = samples.delete(samples=np.where(samples.individuals_time[:] != 0)[0])

    inferred_ts = run_tsinfer(
        samples, simplify=True, num_threads=num_threads, progress=progress
    )
    if not modern_only:
        (
            inferred_ts_simplified,
            removed_mutation_counts,
        ) = evaluation.remove_ancients_mutations(inferred_ts, modern_samples)
        logging.debug(
            "STEP ONE: Inferred tree sequence with {} modern samples and {} \
                    ancients.".format(
                len(modern_samples), num_ancient_samples
            )
        )
        logging.debug(
            "{} mutations were removed when removing ancients. {} were only present \
                    in ancients. {} became fixed in moderns when ancients were removed \
                    and {} became singeltons when ancients were removed.".format(
                (inferred_ts.num_mutations - inferred_ts_simplified.num_mutations),
                removed_mutation_counts[0],
                removed_mutation_counts[1],
                removed_mutation_counts[2],
            )
        )

        inferred_ts_simplified.dump(output_fn + ".tsinferred.trees")
        modern_only_samples = tsinfer.formats.SampleData.from_tree_sequence(
            inferred_ts_simplified, use_times=True
        )
        inferred_ts = inferred_ts_simplified
    else:
        modern_only_samples = tsinfer.formats.SampleData.from_tree_sequence(
            inferred_ts, use_times=True
        )
    assert modern_only_samples.num_samples == num_modern_samples
    return inferred_ts, modern_only_samples


def tsdate_first_pass(inferred_ts, Ne, mutation_rate, output_fn, num_threads, progress):
    """
    Date the inferred tree sequence (with only modern samples)
    """
    tsdate_ages = tsdate.get_dates(inferred_ts, Ne, mutation_rate, progress=progress)
    tsdate_ages = tsdate_ages[0] * 2 * Ne
    tsdate_ages_df = util.get_mut_pos_df(inferred_ts, "FirstPassAges", tsdate_ages)
    tsdate_ages_df.to_csv(output_fn + ".tsdatefirstpass.csv")
    logging.debug(
        "STEP TWO: Dated inferred tree sequence with {} mutations.".format(
            inferred_ts.num_mutations
        )
    )
    return tsdate_ages


def get_ancient_constraints(
    sampledata, modern_sampledata, tsdate_ages, inferred_ts, output_fn, centromere
):
    """
    Constrain sites in sampledata by date estimates and ancient ages.
    Takes sampledata file, which either has modern and ancient samples OR only ancient samples
    modern_sampledata, which only has modern samples. tsdate_ages is the per-node date estimate
    for each node in inferred_ts.
    Returns dated sampledata with only moderns.
    """
    # Make dataframe of mutations present in both modern and ancient samples
    sampledata_ages = sampledata.individuals_time[:][sampledata.samples_individual[:]]
    ancient_samples_bool = sampledata_ages != 0
    ancient_ages = sampledata_ages[ancient_samples_bool]
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
        {"Position": "int32", "Ancient Bound": "float64"}
    )
    mutation_table = inferred_ts.tables.mutations
    sites_table = inferred_ts.tables.sites
    modern_mut_df = pd.DataFrame(
        {
            "Position": sites_table.position[mutation_table.site],
            "Estimated Age": 20000 * 28 * tsdate_ages[mutation_table.node],
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
    modern_pos = modern_sampledata.sites_position[:]
    keep_sites = np.logical_and(modern_pos < centromere[0], modern_pos > centromere[1])
    modern_sampledata_alleles = modern_sampledata.sites_alleles[:][keep_sites]
    #    assert np.array_equal(modern_mut_df["Position"], modern_pos[keep_sites])
    #    assert np.array_equal(modern_mut_df["Ancestral Allele"], [allele[0] for allele in modern_sampledata_alleles])
    #    assert np.array_equal(modern_mut_df["Derived Allele"], [allele[1] for allele in modern_sampledata_alleles])
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

    other_sites = ~np.isin(modern_sampledata.sites_position[:], merged["Position"])
    deleted_sampledata = modern_sampledata.delete(sites=other_sites)
    sample_data_copy = deleted_sampledata.copy(output_fn + ".constrained.samples")
    # Conver age in years back to generations
    which_sites = np.isin(merged["Position"], sample_data_copy.sites_position[:])
    sample_data_copy.sites_time[:] = (merged["Constrained Age"][which_sites] / 28) + 1
    sample_data_copy.finalise()
    #    constrained_ages = np.copy(tsdate_ages)
    #    constrained_ages[inferred_ts.tables.mutations.node] = merged["Constrained Age"]

    #
    #    mut_ages = np.copy(tsdate_ages[inferred_ts.tables.mutations.node])
    #    ancient_samples_bool = sampledata.individuals_time[:][sampledata.samples_individual[:]] != 0
    #    ancient_samples_age = sampledata.individuals_time[:][sampledata.samples_individual[:]][ancient_samples_bool]
    #    genotypes = sampledata.sites_genotypes[:]
    #    genotypes = genotypes[ancient_sites_bool][:, ancient_samples_bool]
    #    positions = modern_sampledata.sites_position[:]
    #    ancient_sites = np.any(genotypes == 1, axis=1)
    #    lower_bound = np.array(
    #        [np.max(ancient_samples_age[geno == 1]) for geno in genotypes[ancient_sites]]
    #    )
    #    # Find maximum of estimated tsdate age and the ancient constraints for each mutation
    #    constrained_ages = np.maximum(
    #        mut_ages[modern_sites_bool][ancient_sites], lower_bound,
    #    )
    #    constr_where = np.where(lower_bound > mut_ages[modern_sites_bool][ancient_sites])[0]
    #    logging.debug(
    #        "STEP THREE: Constraining age estimates with ancient samples. {} mutations were constrained".format(
    #            len(constr_where)
    #        )
    #    )
    #
    #    mut_ages[ancient_sites] = constrained_ages
    #    constr_mut_pos = dict(
    #        zip(
    #            positions[ancient_sites][constr_where],
    #            mut_ages[ancient_sites][constr_where],
    #        )
    #    )
    #    sample_data_copy = modern_sampledata.copy(output_fn + ".constrained.samples")
    #    sample_data_copy.sites_time[:] = mut_ages
    #    sample_data_copy.finalise()
    #    constrained_ages = np.copy(tsdate_ages)
    #    constrained_ages[inferred_ts.tables.mutations.node] = mut_ages

    return sample_data_copy, merged


def get_ancients_sampledata(sampledata, modern_sampledata):
    """
    Takes as input SampleData file with ancient and modern samples.
    Compare with sampledata that only has moderns. Remove any modern samples 
    from combined sampledata, also remove any mutations only appearing in 
    combined sampledata. 
    Return sampledata file with only ancients
    """
    sample_ages = sampledata.individuals_time[:][sampledata.samples_individual[:]]
    modern_samples = np.where(sample_ages == 0)[0]
    ancient_samples_bool = sample_ages == 0
    ancient_sites_notin_modern = np.where(
        ~np.isin(sampledata.sites_position[:], modern_sampledata.sites_position[:])
    )[0]
    ancient_sampledata = sampledata.delete(
        samples=modern_samples, sites=ancient_sites_notin_modern
    )
    assert np.all(
        np.isin(
            ancient_sampledata.sites_position[:], modern_sampledata.sites_position[:]
        )
    )
    return ancient_sampledata


def ancients_as_ancestors(sampledata, ancestor_data, constrained_samples, output_path):
    """
    Insert ancient samples in sample_data file as ancestors in ancestor_data.
    """

    def add_ancient_ancestor(ancestors, ancient_age, ancient_haplotype):
        ancestors.add_ancestor(
            0, ancestor_data.num_sites, ancient_age, [], ancient_haplotype
        )
        assert ancient_haplotype.shape[0] == (ancestors.num_sites)

    # Get age of ancient samples
    ancient_samples = get_ancients_sampledata(sampledata, constrained_samples)
    # We assume the ages in individuals_time are in years, so convert to generations
    ancient_ages = (
        sampledata.individuals_time[:][ancient_samples.samples_individual[:]] / 28
    )
    assert np.all(ancient_ages > 0)
    ancient_metadata = ancient_samples.individuals_metadata[:][
        ancient_samples.samples_individual[:]
    ]
    # Find what time ancient samples should be added as ancestors
    add_ancient_indices = np.searchsorted(
        -ancestor_data.ancestors_time[:], -ancient_ages
    )
    # Make ancient haplotypes consistent with sites in ancestordata. Add missing data to
    # ancient haplotypes when it is missing a site in the ancestordata
    assert ancestor_data.num_sites == constrained_samples.num_inference_sites
    sites_inference = constrained_samples.sites_inference[:]
    ancient_sites_subset = np.isin(
        ancient_samples.sites_position[:],
        constrained_samples.sites_position[:][sites_inference],
    )
    # Find positions in ancestors that are not present in ancients, add them as missing data
    pos_to_add = ancestor_data.sites_position[:][
        ~np.isin(
            constrained_samples.sites_position[:][sites_inference],
            ancient_samples.sites_position[:][ancient_sites_subset],
        )
    ]
    insertion_point = np.searchsorted(
        ancient_samples.sites_position[:][ancient_sites_subset], pos_to_add
    )
    inserted_missing_data = np.full(insertion_point.shape[0], tskit.MISSING_DATA)
    ancient_haplos = list()
    for haplo in ancient_samples.haplotypes():
        ancient_haplo = np.insert(
            haplo[1][ancient_sites_subset], insertion_point, inserted_missing_data
        )
        ancient_haplos.append(ancient_haplo)
        assert len(ancient_haplo) == ancestor_data.num_sites
    # Check that alleles of ancient sample are consistent with sampledata

    with tsinfer.AncestorData(
        sample_data=constrained_samples, path=output_path
    ) as ancestor_data_ancients:
        index = 0
        added_ancients_indices = list()
        added_ancients_metadata = list()
        for cur_index, ancestor in enumerate(ancestor_data.ancestors()):
            if cur_index in add_ancient_indices:
                for ancient_index in np.where(cur_index == add_ancient_indices)[0]:
                    # Add ancient sample as ancestor at appropriate time
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
                add_ancient_ancestor(
                    ancestor_data_ancients,
                    ancient_ages[ancient_index],
                    ancient_haplos[ancient_index],
                )
                added_ancients_indices.append(index)
                added_ancients_metadata.append(ancient_metadata[ancient_index])
                index += 1

    assert ancestor_data_ancients.num_ancestors == (
        ancestor_data.num_ancestors + ancient_samples.num_samples
    ), (
        ancestor_data_ancients.num_ancestors,
        ancestor_data.num_ancestors,
        ancient_samples.num_samples,
    )
    return (
        ancestor_data_ancients,
        np.array(added_ancients_indices),
        added_ancients_metadata,
    )


def tsinfer_second_pass(
    samples_constrained,
    ancient_samples,
    recombination_rate,
    ma_rate,
    ms_rate,
    precision,
    output_fn,
    num_threads,
    progress,
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
                tskit.NODE_IS_SAMPLE, tsinfer.NODE_IS_SAMPLE_ANCESTOR
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

    ancestors_dated = run_generate_ancestors(
        samples_constrained, num_threads=num_threads, progress=progress
    )
    (
        ancestors_w_ancients,
        added_ancient_indices,
        added_ancient_metadata,
    ) = ancients_as_ancestors(
        ancient_samples,
        ancestors_dated,
        samples_constrained,
        output_fn + ".ancient_added.ancestors",
    )
    # round ages for speed in match ancestors
    ancestors_w_ancients_copy = ancestors_w_ancients.copy(
        output_fn + ".ancient_added.ages_rounded.ancestors"
    )
    ancestors_w_ancients_copy.ancestors_time[:] = np.round(
        ancestors_w_ancients.ancestors_time[:]
    )
    ancestors_w_ancients_copy.finalise()
    ancestors_ts = run_match_ancestors(
        samples_constrained,
        ancestors_w_ancients_copy,
        num_threads=num_threads,
        precision=precision,
        recombination_rate=recombination_rate,
        mutation_rate=ma_rate,
        progress=progress,
        path_compression=False,
    )
    ancestors_ts.dump(output_fn + ".atrees")
    ancestors_ts = add_ancestors_flags(
        ancestors_ts, added_ancient_indices, added_ancient_metadata
    )
    iter_infer = run_match_samples(
        samples_constrained, ancestors_ts, num_threads=num_threads, progress=progress
    )
    iter_infer.dump(output_fn + ".iter.tsinferred.trees")
    modern_samples = np.where(iter_infer.tables.nodes.time[iter_infer.samples()] == 0)[
        0
    ]
    #    (
    #        iter_infer_simplified,
    #        removed_mutation_counts,
    #    ) = evaluation.remove_ancients_mutations(iter_infer, modern_samples)
    logging.debug(
        "STEP FOUR: Reinferred tree sequence with {} modern samples and {} ancients.".format(
            np.sum(samples_constrained.individuals_time[:] == 0),
            np.sum(samples_constrained.individuals_time[:] != 0),
        )
    )
    #    logging.debug(
    #        "{} mutations were removed when removing ancients. {} were only present in ancients. {} became fixed in moderns when ancients were removed and {} became singeltons when ancients were removed.".format(
    #            (iter_infer.num_mutations - iter_infer_simplified.num_mutations),
    #            removed_mutation_counts[0],
    #            removed_mutation_counts[1],
    #            removed_mutation_counts[2],
    #        )
    #    )
    return iter_infer


def tsdate_second_pass(
    inferred_ts,
    Ne,
    mut_rate,
    output_fn,
    constr_sites=None,
    adjust_priors=True,
    progress=False,
):
    """
    Run tsdate on tree inferred after iteration.
    """
    inferred_ts = inferred_ts.simplify()
    priors = tsdate.build_prior_grid(inferred_ts, progress=progress)
    if adjust_priors and constr_sites is not None:
        for mut_pos, limit in constr_sites.items():
            infer_site_pos = (
                inferred_ts.tables.sites.position[inferred_ts.tables.mutations.site]
                == mut_pos
            )
            mut_node = (
                inferred_ts.tables.mutations.node[infer_site_pos]
                - inferred_ts.num_samples
            )
            priors.grid_data[
                mut_node, : (np.abs(priors.timepoints * 2 * Ne - limit)).argmin()
            ] = 0
        added_ancestors = np.where(
            inferred_ts.tables.nodes.flags == tsinfer.NODE_IS_SAMPLE_ANCESTOR
        )[0]
        for added_anc in added_ancestors:
            ancient_time = inferred_ts.tables.nodes.time[added_anc]
            priors.grid_data[
                added_anc - inferred_ts.num_samples,
                (np.abs(priors.timepoints * 2 * Ne - ancient_time)).argmin(),
            ] = 1
    iter_dates = tsdate.get_dates(
        inferred_ts, Ne, mut_rate, priors=priors, progress=progress
    )
    tsdate_ages_df = util.get_mut_pos_df(
        inferred_ts, "SecondPassDates", iter_dates[0] * 2 * Ne
    )
    tsdate_ages_df.to_csv(output_fn + ".tsdatesecondpass.csv")
    # Ensure that all dates are greater than the timepoint closest to the lower
    # bound ancient time constraint
    if constr_sites is not None:
        constr_site_timepoint = {
            pos: (priors.timepoints * 2 * Ne)[
                (np.abs(priors.timepoints * 2 * Ne - time)).argmin()
            ]
            for pos, time in constr_sites.items()
        }
        tsdate_ages_df_nodup = tsdate_ages_df.loc[
            ~tsdate_ages_df.index.duplicated(keep=False)
        ]
        assert np.all(
            [
                val >= constr_site_timepoint[pos]
                for pos, val in zip(
                    tsdate_ages_df_nodup.index, tsdate_ages_df_nodup["SecondPassDates"]
                )
                if pos in constr_site_timepoint
            ]
        ), output_fn
    logging.debug(
        "STEP FIVE: Dated reinferred tree sequence with {} mutations.".format(
            inferred_ts.num_mutations
        )
    )
    return iter_dates


def run_tsinfer(sample_data, simplify, num_threads, progress=False):
    progress_monitor = tsinfer.cli.ProgressMonitor(enabled=progress, match_samples=True)
    return tsinfer.infer(
        sample_data,
        simplify=True,
        num_threads=num_threads,
        progress_monitor=progress_monitor,
    )


def run_generate_ancestors(sample_data, num_threads, progress=False):
    progress_monitor = tsinfer.cli.ProgressMonitor(enabled=progress, match_samples=True)
    return tsinfer.generate_ancestors(
        sample_data, num_threads=num_threads, progress_monitor=progress_monitor
    )


def run_match_ancestors(
    sample_data,
    ancestors_data,
    num_threads,
    precision=None,
    mutation_rate=None,
    recombination_rate=None,
    path_compression=True,
    progress=False,
):
    progress_monitor = tsinfer.cli.ProgressMonitor(
        enabled=progress, match_ancestors=True
    )
    return tsinfer.match_ancestors(
        sample_data,
        ancestors_data,
        num_threads=num_threads,
        precision=precision,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        path_compression=path_compression,
        progress_monitor=progress_monitor,
    )


def run_match_samples(sample_data, ancestors_ts, num_threads, progress=False):
    progress_monitor = tsinfer.cli.ProgressMonitor(enabled=progress, match_samples=True)
    return tsinfer.match_samples(
        sample_data,
        ancestors_ts,
        num_threads=num_threads,
        simplify=True,
        progress_monitor=progress_monitor,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process the human data and make data files for plotting."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input sampledata to infer. \
            Be sure that ancient samples (if included) are set to their estimated \
            calendar age",
    )
    parser.add_argument("output", type=str, help="Output dated tree sequence.")
    parser.add_argument("Ne", type=int, help="Estimated effective population size.")
    parser.add_argument("mutation_rate", type=float, help="Estimated mutation rate.")
    parser.add_argument("--num-threads", type=int, default=16)

    args = parser.parse_args()
    iter_infer(args.input, args.output, args.Ne, args.mutation_rate, args.num_threads)


if __name__ == "__main__":
    main()

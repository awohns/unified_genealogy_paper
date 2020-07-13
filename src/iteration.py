import argparse
import numpy as np
import json
import logging
import pandas as pd
from tqdm import tqdm

import tsinfer
import tskit

import utility as util
import evaluation

import tsdate

logging.basicConfig(filename="iteration.log", filemode="w", level=logging.DEBUG)

"""
Infer tree sequences from modern and ancient samples.
Input is a sampledata file with moderns (and optionally ancients).
"""


def iter_infer(
    samples,
    Ne,
    mutation_rate,
    ma_mis=None,
    ms_mis=None,
    recombination_rate=None,
    num_threads=1,
    output_fn=None,
    prune_tree=False,
    ignore_oldest_root=False,
    modern_only_first_pass=False,
    return_trees=False,
    progress=False,
):
    """
    Runs all steps in iterative approach.
    Input is a sampledata file (optionally with ancient samples).
    """

    # Step 1: tsinfer first pass
    inferred_ts = tsinfer_first_pass(
        samples, output_fn, modern_only_first_pass, num_threads, progress
    )

    # Step 2: tsdate first pass
    tsdate_ages, modern_inferred_ts = tsdate_first_pass(
        inferred_ts,
        samples,
        Ne,
        mutation_rate,
        output_fn,
        num_threads=num_threads,
        progress=progress,
    )

    # Step 3: Constrain with ancients
    samples_constrained, constr_mut_pos, constrained_ages = get_ancient_constraints(
        samples, tsdate_ages, modern_inferred_ts, output_fn=output_fn
    )

    # Step 4: tsinfer second pass
    reinferred_ts = tsinfer_second_pass(
        samples_constrained,
        recombination_rate=recombination_rate,
        ma_mis=ma_mis,
        ms_mis=ms_mis,
        output_fn=output_fn,
        num_threads=num_threads,
        progress=progress,
    )

    # Step 5: tsdate second pass
    iter_dates = tsdate_second_pass(
        reinferred_ts,
        Ne,
        mutation_rate,
        output_fn=output_fn,
        constr_sites=constr_mut_pos,
        adjust_priors=True,
        num_threads=num_threads,
        progress=progress,
    )
    if return_trees:
        inferred_dated_ts = get_dated_ts(inferred_ts, tsdate_ages, Ne, 1e-6)
        reinferred_dated_ts = get_dated_ts(reinferred_ts, iter_dates, Ne, 1e-6)
        return inferred_ts, tsdate_ages, constrained_ages, reinferred_ts, iter_dates, inferred_dated_ts, reinferred_dated_ts
    else:
        return inferred_ts, tsdate_ages, constrained_ages, reinferred_ts, iter_dates


def tsinfer_first_pass(
    samples, output_fn=None, modern_only=False, num_threads=1, progress=False
):
    """
    Infer tree sequence topology with modern and ancient samples.
    Then simplify so tree sequence only contains moderns.
    """
    #modern_samples = np.where(samples.individuals_time[:] == 0)[0]
    #num_modern_samples = modern_samples.shape[0]
    #num_ancient_samples = np.sum(samples.individuals_time[:] != 0)
    #if modern_only:
    #    # If first inference does not include ancients, remove them from sampledata file
    #    samples = samples.delete(samples=np.where(samples.individuals_time[:] != 0)[0])
    if progress:
        progress=tsinfer.cli.ProgressMonitor(1, 0, 0, 0, 1)
        inferred_ts = tsinfer.infer(
            samples, simplify=True, num_threads=num_threads, progress_monitor=progress
        )
    else:
        inferred_ts = tsinfer.infer(
            samples, simplify=True, num_threads=num_threads
        )

    if output_fn is not None:
        inferred_ts.dump(output_fn + ".tsinferred.trees")
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
    ancient_nodes = np.where(np.isin(inferred_ts.tables.nodes.individual[:], ancient_indivs))[0]
    # Assert that all ancient nodes are samples
    assert np.all(np.isin(ancient_nodes, inferred_ts.samples()))
    modern_inferred_ts = inferred_ts.simplify(samples=inferred_ts.samples()[~np.isin(inferred_ts.samples(),
        ancient_nodes)])
    tsdate_ages = tsdate.get_dates(modern_inferred_ts, Ne, mutation_rate, method=method, num_threads=num_threads, progress=progress)

    tsdate_ages = tsdate_ages[0] * 2 * Ne
    tsdate_ages_df = util.get_mut_pos_df(modern_inferred_ts, "FirstPassAges", tsdate_ages)
    if output_fn is not None:
        tsdate_ages_df.to_csv(output_fn + ".tsdatefirstpass.csv")
    logging.debug(
        "STEP TWO: Dated inferred tree sequence with {} mutations.".format(
            inferred_ts.num_mutations
        )
    )
    return tsdate_ages, modern_inferred_ts


def get_ancient_constraints(
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

    other_sites = ~np.isin(sampledata.sites_position[:], merged["Position"])
    # print(np.sum(np.isin(sampledata.sites_position[:].astype("int32"), merged["Position"])))
    # print(np.sum(other_sites), sampledata.num_sites, merged["Position"].shape)
    if np.sum(other_sites) != sampledata.num_sites:
        sampledata = sampledata.delete(sites=other_sites)
    if output_fn is not None:
        sampledata_copy = sampledata.copy(output_fn + ".constrained.samples")
    else:
        sampledata_copy = sampledata.copy()
    # Conver age in years back to generations
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

    #sites_inference = modern_samples.sites_inference[:]
    #ancient_sites_inference = np.isin(
    #    sample_data.sites_position[:], modern_samples.sites_position[:][sites_inference]
    #)
    #assert np.sum(ancient_sites_inference) == ancestor_data.num_sites
    sample_ages = sample_data.individuals_time[:][sample_data.samples_individual[:]]
    sample_metadata = sample_data.individuals_metadata[:][
        sample_data.samples_individual[:]
    ]
    ancient_ages = sample_ages[sample_ages != 0]
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
        for cur_index, ancestor in tqdm(enumerate(ancestor_data.ancestors()),
            total=ancestor_data.num_ancestors, disable=not progress,
                desc="Add ancient ancestors"):
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
        added_ancients_metadata)

def tsinfer_second_pass(
    samples_constrained,
    recombination_rate=None,
    ma_mis=None,
    ms_mis=None,
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
            flag_array[ancestor_index] = tsinfer.NODE_IS_SAMPLE_ANCESTOR
            metadata_array[ancestor_index] = json.dumps(added_ancients_metadata[index]).encode()
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
    binned_sd = bin_sampledata(samples_constrained, output_fn)
    ancestors_data = tsinfer.generate_ancestors(
        binned_sd, num_threads=num_threads, progress_monitor=progress
    )
    if output_fn is not None:
        output_str = output_fn + ".ancestors"
    else:
        output_str = None
    (
        ancestors_w_ancients,
        added_ancient_indices,
        added_ancient_metadata,
    ) = ancients_as_ancestors(
        binned_sd, ancestors_data, output_fn, progress=progress
    )
    extra_params = dict(
            num_threads=num_threads,
            recombination_rate=recombination_rate,
            precision=precision,
            progress_monitor=progress
    )
    inferred_anc_ts = tsinfer.match_ancestors( 
        binned_sd,
        ancestors_w_ancients,
        mismatch_rate=ma_mis,
        path_compression=False,
        **extra_params
    )

    inferred_anc_ts = add_ancestors_flags(
        inferred_anc_ts, added_ancient_indices, added_ancient_metadata
    )
    if output_fn is not None:
        inferred_anc_ts.dump(output_fn + ".iter.tsinferred.atrees")

    modern_samples_constrained = binned_sd.delete(samples=np.where(binned_sd.individuals_time[:][binned_sd.samples_individual] != 0)[0])
    iter_infer = tsinfer.match_samples(
        modern_samples_constrained,
        inferred_anc_ts,
        mismatch_rate=ms_mis,
        path_compression=False,
        **extra_params
    )
    if output_fn is not None:
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
            np.sum(binned_sd.individuals_time[:] == 0),
            np.sum(binned_sd.individuals_time[:] != 0),
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
    output_fn=None,
    constr_sites=None,
    adjust_priors=True,
    num_threads=1,
    progress=False,
):
    """
    Infer tree sequence topology with modern and ancient samples.
    Then simplify so tree sequence only contains moderns.
    """
    inferred_ts = inferred_ts.simplify()
    if inferred_ts.num_samples > 1000:
        approximate_priors=True
    else:
        approximate_priors=False
    priors = tsdate.build_prior_grid(inferred_ts, progress=progress, approximate_priors=approximate_priors)
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
            # Remove mutations above sample nodes
            mut_node = mut_node[mut_node > 0]
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
        inferred_ts, Ne, mut_rate, priors=priors, num_threads=num_threads, progress=progress
    )
    tsdate_ages_df = util.get_mut_pos_df(
        inferred_ts, "SecondPassDates", iter_dates[0] * 2 * Ne
    )
    if output_fn is not None:
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
    iter_dates = iter_dates[0] * 2 * Ne
    return iter_dates


def bin_sd_times(sampledata, output_fn=None):
    if output_fn is not None:
        sd = sampledata.copy(output_fn)
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
    
    sd.sites_time[:] = np.around(times * sd.num_samples)/sd.num_samples
    print(
        "Number of samples:",
        sd.num_samples,
        ". Number of discrete times:",
        len(np.unique(sd.sites_time[:])))
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

import argparse
import numpy as np
import sys

import msprime
import tsinfer
import tskit

import utility as util
import constants

sys.path.insert(0, "/home/wilderwohns/tsdate_paper/tsdate/")
import tsdate


def iter_infer(samples, output_fn, Ne, mutation_rate, num_threads,
        progress=False):
    """
    Function to infer tree sequences from modern and ancient samples.
    """
#    if first_pass_noancients:
#        # Remove any mutations only found in ancients
#        samples_pruned = remove_ancient_mutations(samples, output_fn)
#        # Remove ancients and infer tree sequence from only modern samples
#        modern_samples = make_modern_samples(samples_pruned, output_fn)
#        # Infer the ts based on sample data
#        inferred_ts = run_tsinfer(
#            samples, simplify=True, num_threads=num_threads
#        ).simplify()
#    else:
    modern_samples = np.where(samples.individuals_time[:] == 0)[0]
    inferred_ts = run_tsinfer(
        samples, simplify=True, num_threads=num_threads
    ).simplify(samples=modern_samples, filter_sites=True)

    inferred_ts.dump(output_fn + ".tsinferred.trees")
    # tsdate runs
    # tsdated_ts = tsdate.date(inferred_ts, Ne, mutation_rate, progress=True)
    # Date the inferred tree sequence (with only modern samples)
    tsdate_ages = tsdate.get_dates(inferred_ts, Ne, mutation_rate, progress=progress)
    modern_samples = tsinfer.formats.SampleData.from_tree_sequence(inferred_ts, use_times=True)
    tsdate_ages = tsdate_ages[0] * 2 * Ne

    # Constrain and reinfer
    samples_constrained, constrained_mut_ages, constr_mut_pos = get_ancient_constraints(
        samples, modern_samples, tsdate_ages[inferred_ts.tables.mutations.node]
    )
    constr_where = np.where(
        constrained_mut_ages != tsdate_ages[inferred_ts.tables.mutations.node]
    )[0]
    constrained_ages = np.copy(tsdate_ages)
    constrained_ages[inferred_ts.tables.mutations.node] = constrained_mut_ages

    # When reinfering the tree sequence, we now use the ancient samples as ancestors
    ancestors_data = run_generate_ancestors(
        samples_constrained, num_threads=num_threads
    )
    samples_pruned_modern_sites_inf = modern_samples.copy(output_fn + "modern_sites.samples")
    samples_pruned_modern_sites_inf.sites_inference[
        :
    ] = samples_constrained.sites_inference[:]
    samples_pruned_modern_sites_inf.finalise()
    ancestors_w_ancients, added_ancient_indices = ancients_as_ancestors(
        samples_constrained, ancestors_data, output_fn + ".ancestors"
    )
    ancestors_ts = run_match_ancestors(
        samples_constrained, ancestors_w_ancients, num_threads=num_threads
    )
    ancestors_ts = add_ancestors_flags(ancestors_ts, added_ancient_indices)
    iter_infer = run_match_samples(
        samples_constrained, ancestors_ts, num_threads=num_threads
    ).simplify()
    # iter_infer = tsinfer.infer(samples_constrained).simplify()
    iter_infer.dump(output_fn + ".iter.tsinferred.trees")
    #iter_dates = tsdate.get_dates(iter_infer, Ne, mutation_rate)
    modern_samples = np.where(iter_infer.tables.nodes.time[iter_infer.samples()] == 0)[0]
    iter_dates = iteration_tsdate(iter_infer.simplify(samples=modern_samples), Ne, mutation_rate, constr_sites=constr_mut_pos, adjust_priors=True, progress=progress)
    # iter_dated_ts.dump(output_fn + ".iter.tsdated.trees")

    return inferred_ts, tsdate_ages, constrained_ages, iter_infer, iter_dates


def add_ancestors_flags(ts, added_ancients_indices):
    for index, node in enumerate(ts.tables.nodes):
        if node.flags & tsinfer.constants.NODE_IS_PC_ANCESTOR:
            added_ancients_indices[added_ancients_indices > index] += 1 
    tables = ts.dump_tables()
    flag_array = tables.nodes.flags.copy()
    for index in added_ancients_indices:
        flag_array[index] = tsinfer.NODE_IS_SAMPLE_ANCESTOR
    tables.nodes.set_columns(flags=flag_array, time=tables.nodes.time,
                           population=tables.nodes.population, individual=tables.nodes.individual,
                           metadata=tables.nodes.metadata, metadata_offset=tables.nodes.metadata_offset)
    return tables.tree_sequence()


def iteration_tsdate(inferred_ts, Ne, mut_rate, constr_sites=None, adjust_priors=True, progress=False):
    """
    Run tsdate on tree inferred after iteration.
    """
    priors = tsdate.build_prior_grid(inferred_ts)
    if adjust_priors and constr_sites is not None:
        for mut_pos, limit in constr_sites.items():
            infer_site_pos = inferred_ts.tables.sites.position[inferred_ts.tables.mutations.site] == mut_pos
            mut_node = inferred_ts.tables.mutations.node[infer_site_pos] - inferred_ts.num_samples
            priors.grid_data[mut_node, : (np.abs(priors.timepoints * 2 * Ne - limit)).argmin()] = 0
        added_ancestors = np.where(inferred_ts.tables.nodes.flags == tsinfer.NODE_IS_SAMPLE_ANCESTOR)[0]
        for added_anc in added_ancestors:
            ancient_time = inferred_ts.tables.nodes.time[added_anc]
            priors.grid_data[added_anc - inferred_ts.num_samples,
                    (np.abs(priors.timepoints * 2 * Ne - ancient_time)).argmin()] = 1
    iter_dates = tsdate.get_dates(inferred_ts, Ne, mut_rate, priors=priors, progress=progress)
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


def run_match_ancestors(sample_data, ancestors_data, num_threads, progress=False):
    progress_monitor = tsinfer.cli.ProgressMonitor(enabled=False, match_samples=True)
    return tsinfer.match_ancestors(
        sample_data,
        ancestors_data,
        num_threads=num_threads,
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


def get_ancient_constraints(sampledata, modern_sampledata, tsdate_ages):
    """
    Constrain sites in sampledata by date estimates and ancient ages.
    Return dated sampledata with only moderns.
    """
    modern_sites_bool = np.isin(
        modern_sampledata.sites_position[:], sampledata.sites_position[:]
    )
    ancient_sites_bool = np.isin(
        sampledata.sites_position[:], modern_sampledata.sites_position[:]
    )
    ancient_samples_bool = sampledata.individuals_time[:] != 0
    ancient_samples_age = sampledata.individuals_time[:][ancient_samples_bool]
    genotypes = sampledata.sites_genotypes[:]
    genotypes = genotypes[ancient_sites_bool][:, ancient_samples_bool]
    positions = modern_sampledata.sites_position[:]
    ancient_sites = np.any(genotypes == 1, axis=1)
    # mut_ages, _ = util.get_mut_ages(dated_ts, dated_ts.tables.nodes.time[:])
    mut_ages = tsdate_ages
    lower_bound = np.array(
        [np.max(ancient_samples_age[geno == 1]) for geno in genotypes[ancient_sites]]
    )
    # Find the maximum of the estimated tsdate age and the ancient constraints for each mutation
    constrained_ages = np.maximum(
        mut_ages[modern_sites_bool][ancient_sites], lower_bound,
    )
    constr_where = np.where(lower_bound > mut_ages[modern_sites_bool][ancient_sites])[0]
    #print("Constrained at following:", constr_where)
    #print(
    #    constrained_ages[constr_where],
    #    lower_bound[constr_where],
    #    mut_ages[modern_sites_bool][ancient_sites][constr_where],
    #)
    mut_ages[ancient_sites] = constrained_ages
    #print(mut_ages[ancient_sites][constr_where], positions[ancient_sites][constr_where])
    constr_mut_pos = dict(zip(positions[ancient_sites][constr_where], mut_ages[ancient_sites][constr_where]))
    assert np.array_equal(mut_ages, tsdate_ages)
    # sample_data_copy = sampledata.copy()
    sample_data_copy = modern_sampledata.copy("samples_constrained")
    sample_data_copy.sites_time[:] = mut_ages
    sample_data_copy.finalise()
    return sample_data_copy, mut_ages, constr_mut_pos


def remove_ancient_mutations(samples, output_fn):
    """
    Remove mutations which only appear in ancient samples.
    TODO: Remove mutations which become fixed in moderns.
    """
    modern_samples_bool = samples.individuals_time[:] == 0
    with tsinfer.SampleData(
        path=output_fn + ".pruned.samples",
        sequence_length=samples.sequence_length,
        num_flush_threads=2,
    ) as samples_pruned:
        for pop in samples.populations_metadata[:]:
            samples_pruned.add_population(pop)
        pop_metadata = samples.populations_metadata[:]
        for indiv in samples.individuals():
            samples_bool = samples.samples_individual[:] == indiv.id
            population_id = samples.samples_population[:][samples_bool]
            # All population labels for this individual should be the same
            assert np.all(population_id == population_id[0])
            samples_pruned.add_individual(
                metadata=indiv.metadata,
                time=indiv.time,
                population=population_id[0],
                ploidy=np.sum(samples_bool),
            )
        for var in samples.variants():
            modern_genos = var.genotypes[modern_samples_bool]
            if np.any(modern_genos == 1) and not np.all(modern_genos == 1):
                samples_pruned.add_site(var.site.position, var.genotypes, var.alleles)
    return samples_pruned


def make_modern_samples(samples, output_fn):
    # Remove ancients and infer tree sequence from only modern samples
    with tsinfer.SampleData(
        path=output_fn + ".modern.samples",
        sequence_length=samples.sequence_length,
        num_flush_threads=2,
    ) as modern_samples:
        for pop in samples.populations_metadata[:]:
            pop_id = modern_samples.add_population(pop)
        for indiv in samples.individuals():
            if indiv.time == 0:
                samples_bool = samples.samples_individual[:] == indiv.id
                population_id = samples.samples_population[:][samples_bool]
                # All population labels for this individual should be the same
                assert np.all(population_id == population_id[0])
                modern_samples.add_individual(
                    metadata=indiv.metadata,
                    time=indiv.time,
                    population=population_id[0],
                    ploidy=np.sum(samples_bool),
                )

        modern_samples_bool = samples.individuals_time[:] == 0
        for var in samples.variants():
            if np.any(var.genotypes[modern_samples_bool] == 1):
                modern_samples.add_site(
                    var.site.position, var.genotypes[modern_samples_bool], var.alleles
                )
    return modern_samples


def ancients_as_ancestors(sample_data, ancestor_data, output_path):
    """
    Insert ancient samples in sample_data file as ancestors in ancestor_data.
    """
    def add_ancient_ancestor(ancestors, ancient_age, ancient_haplotype):
        ancestors.add_ancestor(
            0,
            ancestor_data.num_sites,
            ancient_age,        
            [],
            ancient_haplotype
        )
        assert ancient_haplotype.shape[0] == (
            ancestors.num_sites
        )


    sites_inference = sample_data.sites_inference[:]
    assert np.sum(sites_inference) == ancestor_data.num_sites
    sample_ages = sample_data.individuals_time[:]
    ancient_ages = sample_ages[sample_ages != 0]
    ancient_samples = np.where(sample_ages != 0)[0]
    add_ancient_indices = np.searchsorted(
        -ancestor_data.ancestors_time[:], -ancient_ages
    )
    ancient_haplos = list()

    for haplo in sample_data.haplotypes():
        if haplo[0] in ancient_samples:
            ancient_haplos.append(haplo[1][sites_inference])

    with tsinfer.AncestorData(
        sample_data=sample_data, path=output_path
    ) as ancestor_data_ancients:
        index = 0
        added_ancients_indices = list()
        for cur_index, ancestor in enumerate(ancestor_data.ancestors()):
            if cur_index in add_ancient_indices:
                for ancient_index in np.where(cur_index == add_ancient_indices)[0]:
                    # Add ancient sample as ancestor at appropriate time
                    add_ancient_ancestor(ancestor_data_ancients, ancient_ages[ancient_index], ancient_haplos[ancient_index])
                    added_ancients_indices.append(index)
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
                add_ancient_ancestor(ancestor_data_ancients, ancient_ages[ancient_index], ancient_haplos[ancient_index])
                added_ancients_indices.append(index)
                index += 1

            
    assert ancestor_data_ancients.num_ancestors == (
        ancestor_data.num_ancestors + len(ancient_samples)
    ), (ancestor_data_ancients.num_ancestors, ancestor_data.num_ancestors, len(ancient_samples))
    return ancestor_data_ancients, np.array(added_ancients_indices)


def main():
    parser = argparse.ArgumentParser(
        description="Process the human data and make data files for plotting."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input sampledata to infer. \
            Be sure that ancient samples (if included) are set to their estimated calendar age",
    )
    parser.add_argument("output", type=str, help="Output dated tree sequence.")
    parser.add_argument("Ne", type=int, help="Estimated effective population size.")
    parser.add_argument("mutation_rate", type=float, help="Estimated mutation rate.")
    parser.add_argument("--num-threads", type=int, default=16)

    args = parser.parse_args()
    iter_infer(args.input, args.output, args.Ne, args.mutation_rate, args.num_threads)


if __name__ == "__main__":
    main()

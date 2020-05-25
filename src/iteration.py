import argparse
import numpy as np

import msprime
import tsinfer
import tskit
import tsdate

import utility as util
import constants


def iter_infer(input_modern_samples_fn, output_fn, Ne, mutation_rate, num_threads):
    """
    Function to infer tree sequences from modern and ancient samples.
    """
    samples = tsinfer.load(input_modern_samples_fn)
    # Remove any mutations only found in ancients
    samples_pruned = remove_ancient_mutations(samples, output_fn)

    # Remove ancients and infer tree sequence from only modern samples
    modern_samples = make_modern_samples(samples_pruned, output_fn)
    # Infer the ts based on sample data
    inferred_ts = tsinfer.infer(modern_samples)
    inferred_ts = inferred_ts.simplify()
    inferred_ts.dump(output_fn + ".tsinferred.trees")

    # tsdate runs
    tsdated_ts = tsdate.date(inferred_ts, Ne, mutation_rate)

    # Constrain and reinfer
    samples_constrained = get_ancient_constraints(
        samples_pruned, modern_samples, tsdated_ts
    )

    # When reinfering the tree sequence, we now use the ancient samples as ancestors
    ancestors_data = run_generate_ancestors(
        samples_constrained, num_threads=num_threads
    )
    ancestors_w_ancients = ancients_as_ancestors(samples_pruned, ancestors_data)
    ancestors_ts = run_match_ancestors(
        samples_pruned, ancestors_w_ancients, num_threads=num_threads
    )
    iter_infer = run_match_samples(
        samples_pruned, ancestors_ts, num_threads=num_threads
    )
    # iter_infer = tsinfer.infer(samples_constrained).simplify()
    iter_infer.dump(output_fn + ".iter.tsinferred.trees")
    iter_dated_ts = tsdate.date(iter_infer, Ne, mutation_rate)
    iter_dated_ts.dump(output_fn + "iter.tsdated.trees")


def run_generate_ancestors(sample_data, num_threads):
    progress_monitor = tsinfer.cli.ProgressMonitor(enabled=True, match_samples=True)
    return tsinfer.generate_ancestors(
        sample_data, num_threads=num_threads, progress_monitor=progress_monitor
    )


def run_match_ancestors(sample_data, ancestors_data, num_threads):
    progress_monitor = tsinfer.cli.ProgressMonitor(enabled=True, match_samples=True)
    return tsinfer.match_ancestors(
        sample_data,
        ancestors_data,
        num_threads=num_threads,
        progress_monitor=progress_monitor,
    )


def run_match_samples(sample_data, ancestors_ts, num_threads):
    progress_monitor = tsinfer.cli.ProgressMonitor(enabled=True, match_samples=True)
    return tsinfer.generate_ancestors(
        sample_data,
        ancestors_ts,
        num_threads=num_threads,
        simplify=True,
        progress_monitor=progress_monitor,
    )


# def make_modern_samples(samples, output_fn):
#    # Remove ancients and infer tree sequence from only modern samples
#    with tsinfer.SampleData(
#        path=output_fn + ".modern.samples",
#        sequence_length=samples.sequence_length,
#        num_flush_threads=2,
#    ) as modern_samples:
#        population_id_map = {}
#        for pop in samples.populations_metadata[:]:
#            if pop:
#                pop_id = modern_samples.add_population(pop)
#                try:
#                    population_id_map[pop["name"]] = pop_id
#                except:
#                    raise ValueError("Population must have field 'name'")
#        pop_metadata = samples.populations_metadata[:]
#        for indiv in samples.individuals():
#            if indiv.time == 0:
#                samples_bool = samples.samples_individual[:] == indiv.id
#                modern_samples.add_individual(
#                    metadata=indiv.metadata,
#                    population=population_id_map[
#                        pop_metadata[samples.samples_population[samples_bool]]["name"]
#                    ],
#                    ploidy=np.sum(samples_bool),
#                )
#        modern_samples_bool = samples.individuals_time[:] == 0
#        for var in samples.variants():
#            if np.any(var.genotypes[modern_samples_bool] == 1):
#                modern_samples.add_site(
#                    var.site.position, var.genotypes[modern_samples_bool], var.alleles
#                )
#    return modern_samples


def get_ancient_constraints(sampledata, modern_sampledata, dated_ts):
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
    ancient_sites = np.any(genotypes == 1, axis=1)
    mut_ages, _ = util.get_mut_ages(dated_ts, dated_ts.tables.nodes.time[:])
    lower_bound = np.array(
        [np.max(ancient_samples_age[geno == 1]) for geno in genotypes[ancient_sites]]
    )
    # Find the maximum of the estimated tsdate age and the ancient constraints for each mutation
    constrained_ages = np.maximum(
        mut_ages[modern_sites_bool][ancient_sites] * constants.GENERATION_TIME,
        lower_bound,
    )
    mut_ages[ancient_sites] = constrained_ages
    # sample_data_copy = sampledata.copy()
    sample_data_copy = modern_sampledata.copy()
    sample_data_copy.sites_time[:] = mut_ages
    sample_data_copy.finalise()
    return sample_data_copy


def remove_ancient_mutations(samples, output_fn):
    """
    Remove mutations which only appear in ancient samples 
    """
    modern_samples_bool = samples.individuals_time[:] == 0
    with tsinfer.SampleData(
        path=output_fn + ".modern.samples",
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
            if np.any(var.genotypes[modern_samples_bool] == 1):
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


def ancients_as_ancestors(sample_data, ancestors_data):
    assert sample_data.num_sites == ancestors_data.num_sites
    sample_ages = sample_data.individuals_time[:]
    ancient_ages = sample_ages[sample_ages != 0]
    ancient_samples = np.where(sample_ages != 0)
    assert ancestors_data.num_ancestors == (
        sample_data.num_samples - len(ancient_samples)
    )
    add_ancient_indices = np.searchsorted(
        -ancestors_data.ancestors_time[:], -ancient_ages
    )
    ancient_haplos = dict()

    for haplo in sample_data.haplotypes():
        if haplo[0] in ancient_samples:
            ancient_haplos[haplo[0]] = haplo[1]

    with tsinfer.AncestorData(
        sample_data=sample_data, path=path
    ) as ancestor_data_ancients:
        index = 0
        added_ancestors_indices = list()
        for cur_index, ancestor in enumerate(ancestors_data.ancestors()):
            if cur_index in add_ancient_indices:
                for ancient_index in np.where(cur_index == add_ancient_indices)[0]:
                    ancestor_data_ancients.add_ancestor(
                        0,
                        ancestors_data.num_sites,
                        ancient_ages[ancient_index],
                        [],
                        ancient_haplos[ancient_index],
                    )
                    added_ancestors_indices.append(index)
                    assert ancient_haplos[ancient_index].shape == (
                        ancestors_data.num_sites
                    )
                    index += 1
                ancestor_data_ancients.add_ancestor(
                    ancestor.start,
                    ancestor.end,
                    ancestor.time,
                    ancestor.focal_sites,
                    ancestor.haplotype,
                )
                index += 1
    assert ancestor_data_ancients.num_ancestors == sample_data.num_samples
    return ancestor_data_ancients.finalise()


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

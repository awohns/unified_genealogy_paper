import argparse
import msprime
import tsinfer
import tskit
import numpy as np

import utility as util


def iter_infer(input_modern_samples_fn, output_fn, Ne, mutation_rate):
    """
    Function to infer tree sequences from modern and ancient samples.
    """
    samples = tsinfer.load(input_modern_samples_fn)
    # Remove ancients and infer tree sequence from only modern samples
    with tsinfer.SampleData(
        path=output_fn + ".modern.samples",
        sequence_length=samples.sequence_length,
        num_flush_threads=2,
    ) as modern_samples:
        population_id_map = {}
        for pop in samples.populations_metadata[:]:
            if pop:
                pop_id = modern_samples.add_population(pop)
                try:
                    population_id_map[pop["name"]] = pop_id
                except:
                    raise ValueError("Population must have field 'name'")
        pop_metadata = samples.populations_metadata[:]
        for indiv in modern_samples.individuals():
            if indiv.time == 0:
                samples_bool = samples.samples_individual[:] == indiv.id
                modern_samples.add_individual(
                    metadata=indiv.metadata,
                    population=population_id_map[
                        pop_metadata[samples.samples_population[samples_bool]]["name"]
                    ],
                    ploidy=np.sum(samples_bool),
                )
        modern_samples_bool = samples.individuals_time[:] == 0
        for var in samples.variants():
            if np.any(var.genotypes[modern_samples_bool] == 1):
                modern_samples.add_site(
                    var.site.position, var.genotypes[modern_samples_bool], var.alleles
                )

    # Infer the ts based on sample data
    inferred_ts, tsinfer_cpu, tsinfer_memory = tsinfer.infer(modern_samples)
    inferred_ts = inferred_ts.simplify()
    inferred_ts.dump(output_fn + ".tsinferred.trees")

    # tsdate runs
    tsdate_dates = evaluation.run_tsdate_get_dates(inferred_ts, Ne, mutation_rate)

    # Constrain and reinfer
    samples_constrained = get_ancient_constraints(samples)
    iter_infer = tsinfer.infer(samples_constrained).simplify()
    iter_infer.dump(output_fn + ".iter.tsinferred.trees")
    iter_dated_ts = evaluation.run_tsdate(iter_infer, Ne, mutation_rate)
    iter_dated_ts.dump(output_fn + "iter.tsdated.trees")


def get_ancient_constraints(sampledata, dated_ts):
    ancient_samples_bool = sampledata.individuals_time[:] != 0
    ancient_samples_age = sampledata.individuals_time[:][ancient_samples_bool]
    genotypes = sampledata.sites_genotypes[:][:, ancient_samples_bool]
    ancient_sites = np.any(genotypes == 1, axis=1)
    mut_ages, _ = util.get_mut_ages(dated_ts, dated_ts.tables.nodes.time[:])
    # positions = sampledata.sites_position[:][np.any(genotypes == 1, axis=1)]
    lower_bound = np.array(
        [
            np.max(ancient_samples_age[geno == 1])
            for pos, geno in genotypes[ancient_sites]
        ]
    )
    # Find the maximum of the estimated tsdate age and the ancient constraints for each mutation
    constrained_ages = np.maximum(
        mut_ages[ancient_sites] * constants.GENERATION_TIME, lower_bound
    )
    mut_ages[ancient_sites] = constrained_ages
    sample_data_copy = samples.copy()
    sample_data_copy.sites_time[:] = mut_ages
    sample_data_copy.finalise()
    return sample_data_copy


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

    args = parser.parse_args()
    iter_infer(args.input, args.output, args.Ne, args.mutation_rate)


if __name__ == "__main__":
    main()

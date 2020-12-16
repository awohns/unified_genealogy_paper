import numpy as np
import pandas as pd


def make_seq_errors_genotype_model(g, error_probs):
    """
    Given an empirically estimated error probability matrix, resample for a particular
    variant. Determine variant frequency and true genotype (g0, g1, or g2),
    then return observed genotype based on row in error_probs with nearest
    frequency. Treat each pair of alleles as a diploid individual.
    """
    m = g.shape[0]
    frequency = np.sum(g) / m
    closest_row = (np.abs(error_probs["freq"] - frequency)).idxmin()
    closest_freq = error_probs.iloc[closest_row - 1].values[1:]

    w = np.copy(g)

    # Make diploid (iterate each pair of alleles)
    genos = np.reshape(w, (-1, 2))

    # Record the true genotypes (0,0=>0; 1,0=>1; 0,1=>2, 1,1=>3)
    count = np.sum(np.array([1, 2]) * genos, axis=1)

    base_genotypes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    genos[count == 0, :] = base_genotypes[
        np.random.choice(
            4, sum(count == 0), p=closest_freq[[0, 1, 1, 2]] * [1, 0.5, 0.5, 1]
        ),
        :,
    ]
    genos[count == 1, :] = base_genotypes[[0, 1, 3], :][
        np.random.choice(3, sum(count == 1), p=closest_freq[[3, 4, 5]]), :
    ]
    genos[count == 2, :] = base_genotypes[[0, 2, 3], :][
        np.random.choice(3, sum(count == 2), p=closest_freq[[3, 4, 5]]), :
    ]
    genos[count == 3, :] = base_genotypes[
        np.random.choice(
            4, sum(count == 3), p=closest_freq[[6, 7, 7, 8]] * [1, 0.5, 0.5, 1]
        ),
        :,
    ]

    return np.reshape(genos, -1)


def add_errors(sample_data, ancestral_allele_error=0, random_seed=None, **kwargs):
    if random_seed is not None:
        np.random.seed(random_seed)
    if sample_data.num_samples % 2 != 0:
        raise ValueError("Must have an even number of samples to inject error")
    error_probs = pd.read_csv("data/EmpiricalErrorPlatinum1000G.csv", index_col=0)
    n_variants = 0
    aa_error_by_site = np.zeros(sample_data.num_sites, dtype=np.bool)
    if ancestral_allele_error > 0:
        assert ancestral_allele_error <= 1
        n_bad_sites = round(ancestral_allele_error * sample_data.num_sites)
        # This gives *exactly* a proportion aa_error or bad sites
        # NB - to to this probabilitistically, use np.binomial(1, e, ts.num_sites)
        aa_error_by_site[0:n_bad_sites] = True
        np.random.shuffle(aa_error_by_site)
    new_sd = sample_data.copy(**kwargs)
    genotypes = new_sd.data["sites/genotypes"][:]  # Could be big
    alleles = new_sd.data["sites/alleles"][:]

    for i, (ancestral_allele_error, v) in enumerate(
        zip(aa_error_by_site, sample_data.variants())
    ):
        if ancestral_allele_error and len(v.site.alleles) == 2:
            genotypes[i, :] = 1 - v.genotypes
            alleles[i] = list(reversed(alleles[i]))
        genotypes[i, :] = make_seq_errors_genotype_model(genotypes[i, :], error_probs)
    new_sd.data["sites/genotypes"][:] = genotypes
    new_sd.data["sites/alleles"][:] = alleles
    new_sd.finalise()
    return new_sd

# Creating Tree Sequences from Human Data

This directory contains all code necessary to produce the unified tree sequence
of the 1000 Genomes Project, Human Genome Diversity Project, Simons Genome
Diversity Project, the four high coverage archaic genomes, and
the Afanasievo family, using the full dataset of ancient samples
from the Reich Laboratory to constrain age estimates. A Makefile is used to produce these files
as well as all intermediate files.

We assume that all commands are run **within** this directory.

Note: the Afanasievo datasets are downloaded from [here](https://reichdata.hms.harvard.edu/pub/datasets/release/wohn_2021_phasedAfanasievo/).


## Requirements

The pipelines require Python 3 and the software built using the Makefile in the `tools`
directory.

The Python package requirements can be installed using the ``requirements.txt`` file 
in the repository root.

## 1000 Genomes

To build a dated tree sequence for 1000 Genomes chromosome 20, run:

```
make tgp_chr20.dated.trees
```

The Makefile will proceed with the following steps to infer the tree sequence:
1. Download Chromosome 20 variant data
2. Produce a ``.samples`` file (``tsinfer`` input format). It takes some time to convert from a VCF file to the `.samples` file.
3. Infer tree sequence topology (`.trees` file)
4. Date tree sequence (`.dated.trees` file)

The number of threads used in steps of the pipeline that support 
threading can be controlled using the ``NUM_THREADS`` make 
variable, i.e.

```
make NUM_THREADS=20 tgp_chr20.dated.trees
```

will tell ``tsinfer`` to use 20 threads where appropriate.


## Unified Tree Sequence

To build the dated tree sequence of all datasets, run:

```
make hgdp_tgp_sgdp_high_cov_ancients_chr20.dated.trees
```

The pipeline for creating the unified tree sequence is generic, and so any chromosome can be built
by running, e.g., ``make hgdp_tgp_sgdp_high_cov_ancients_dated_chr3.trees``

The Makefile will proceed with the following steps to infer the unified tree sequence:
1. Download variant information
2. LiftOver some datasets to GRCh38
3. Produce a ``.samples`` file for each dataset
4. Merge ``.samples`` files of modern datsets
5. Infer a tree sequence of modern individuals with ``tsinfer``
6. Date tree sequence with ``tsdate``
7. Constrain date estimates with ancient samples
8. Reinfer tree sequence with modern and ancient samples



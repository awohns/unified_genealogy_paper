# Creating Tree Sequences from Human Data

This direcotry includes all code to produce the unified tree sequence
of the 1000 Genomes Project, Human Genome Diversity Project, Simons Genome
Diversity Project, the four high coverage archaic genomes,
the Afanasievo family, and the full dataset of ancient samples
from the Reich Laboratory. A Makefile is used to produce these files
as well as all intermediate files.

Outline of process for producing dated 1000 Genomes Tree Sequence
1. Download Chromosome 20 variant data
2. Produce a ``.samples'' file (``tsinfer'' input format)
3. Infer tree sequence
4. Date tree sequence

Outline of process for producing the unified tree sequence for each chromosome:
1. Download variant information
2. LiftOver to GRCh38 if necessary
3. Produce a ``.samples'' file
4. Merge ``.samples'' files
5. Infer a tree sequence with ``tsinfer''
6. Date tree sequence with ``tsdate''
7. Constrain date estimates with ancient samples
8. Reinfer tree sequence with modern and ancient samples

We assume that all commands are run **within** this directory.

## Requirements

The pipelines require:

- Python 3
- bcftools
- vcftools
- samtools
- tabix
- convertf
- Picard

The Python package requirements are listed in the ``requirements.txt`` file 
in the repository root.

## 1000 Genomes

To build a dated tree sequence for 1000 Genomes chromosome 20, run:

```
$ make 1kg_chr20.dated.trees
```

This will download the data, create the intermediate files, 
output the tree sequence ``.trees`` file, and then date the tree sequence.
Converting from VCF to the ``.samples`` file will take some time.

The number of threads used in steps of the pipeline that support 
threading can be controlled using the ``NUM_THREADS`` make 
variable, i.e.

```
$ make NUM_THREADS=20 1kg_chr20.dated.trees
```

will tell ``tsinfer`` to use 20 threads where appropriate.

The pipeline for 1000 genomes is generic, and so any chromosome can be built
by running, e.g., ``make 1kg_chr1.dated.trees``.

## Unified Tree Sequence

To build the dated tree sequence of all datasets, run:

```
$ make hgdp_1kg_sgdp_high_cov_ancients_dated_chr20.missing_binned.dated.trees
```

The number of threads and which chromosome to infer can be controlled as described
above.

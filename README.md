# Code for "A unified genealogy of modern and ancient genomes"

This repo contains code used in ["A unified genealogy of modern and ancient genomes"](https://www.biorxiv.org/content/10.1101/2021.02.16.431497v2).
This includes:
* pipelines to generate a unified, dated tree sequence from multiple datasets
* simulation based validation of [tsinfer](https://tsinfer.readthedocs.io/) and
[tsdate](https://tsdate.readthedocs.io/en/latest/)
* empirical validation of inferred tree sequences using ancient genomes
* code to produce all non-schematic figures in the paper, as well as the interactive figure and supplementary video

These analyses are placed in subdirectories as follows:
* `all-data` contains all code for downloading and preparing real data, as well as the pipeline for creating tree sequences
* `src` contains scripts to run analyses and validation on both real and simulated data, as well as code for plotting figures
* `data` contains reference data needed for some analyses. It is also where the results of analyses performed on tree sequences inferred from real data are stored. Finally, it is also where simulations evaluating mismatch parameters in tsinfer are performed
* `simulated-data` is where the results of all simulation-based analyses are saved
* `tools` contains methods used to compare the accuracy of `tsinfer` and `tsdate` as well as tools to process variant data files

### Getting Started
You must first clone this repo, including submodules as follows:

```
git clone --recurse-submodules https://github.com/awohns/unified_genealogy_paper.git
cd unified_genealogy_paper
```

### Required Software

#### Installing required python modules

Please use Python >= 3.8 and <= 3.10 (`numba' currently fails to install under Python 3.11).

First, ``cartopy`` (required for generating figures with maps) and ``ffmpeg`` (required to create the movie)  must be installed
with conda as follows:

```
conda install -c conda-forge cartopy
conda install -c conda-forge ffmpeg
```

All other required Python packages are listed in the ``requirements.txt`` file. These can be 
installed with

```
python -m pip install -r requirements.txt
```


#### Required software for preparing real data

We require [BCFtools, SAMtools, and HTSlib](http://www.htslib.org/download/), as well as
[convertf](https://github.com/argriffing/eigensoft/tree/master/CONVERTF), [Picard](https://broadinstitute.github.io/picard/),
[eigensoft](https://github.com/argriffing/eigensoft) and [plink](http://zzz.bwh.harvard.edu/plink/download.shtml)
to prepare the variant data files from
the 1000 Genomes Project, Human Genome Diversity Project, Simons Genome Diversity Project, and ancient DNA
datasets for use with `tsinfer` and `tsdate`.
We compare our methods with [Genealogical Estimation of Variant Age (GEVA)](https://github.com/pkalbers/geva) and
[Relate](https://myersgroup.github.io/relate/index.html).
All of these tools are kept in the ``tools`` directory and can be downloaded and built using 

```
cd tools
make all
```

### Inferring Tree Sequences from Real Data

Please see the [README](all-data/README.md) in the ``all-data`` directory for details on inferring tree sequences from real data.


### Running Simulations

To generate data required for simulation-based figures and to plot the figures themselves, follow this general process: generate data, run analyses, and plot the results. 
For example, the following will generate Figure 1c:

```
python src/run_evaluation.py tsdate_neutral_sims --setup
python src/run_evaluation.py tsdate_neutral_sims --inference
python src/plot.py tsdate_neutral_sims
```

The first command runs the simulations for each evaluation. The simulations are stored in the `simulated-data` directory 
The second command performs inference on the results of the simulations. This will take multiple
days to run for all simulations in the paper. The results are stored as
csv files in the `simulated-data` directory. The third command plots the csv files and saves the resulting figures to the
`figures` directory.


### Running all evaluations

To produce all the simulation data in our paper, run the following, in order:

```
python src/run_evaluation.py --setup all 
python src/run_evaluation.py --infer all # will take multiple days
```

You can speed up the inference step by using multiple processors, specified using the `-p` flag.
For instance, on a 64 core machine, using all cores:

```
python src/run_evaluation.py --infer all -p 64
```

The mismatch simulations are produced separately, using the Makefile in the `data` directory:

```
cd data
make mismatch
```

### Analyzing inferred tree sequences

Once you have inferred tree sequences and the results are in the ``all-data`` directory, run functions in ``src/analyze_data.py`` to generate data for non-simulation based figures. The figures themselves are plotted using ``src/plot.py``

Before running these analyses, for certain plots you will need to download other required files (allele age estimates from `Relate` and `GEVA` for fig S18="tgp_muts_frequency"; the 1000 genomes chr20 mask file for for fig S6="ld_quality_by_mutations"). You can do this using the following:

```
cd data
make relate_ages
make geva_ages.csv.gz
make chr20_mask
```

You can then run all analyses using real data with:

```
python src/analyze_data.py all
```

### Plotting figures

When the above steps have been completed, figures (including the movie) can be plotted using:

```
python src/plot.py all
```

The interactive figure is created separately using the following command:

```
python src/interactive_plot.py
```
#### Reproducing plots from downloaded data

Note that if you simply want to produce figures similar to those in our published paper, you can download the
tree sequences from Zenodo (https://zenodo.org/record/5512994), and perform plotting steps separately,
e.g. within a terminal

```
cd all-data
# can do this for different chromosome arms: here we use chr20_q as an example
wget https://zenodo.org/record/5512994/files/hgdp_tgp_sgdp_high_cov_ancients_chr20_q.dated.trees.tsz
python -m tszip -d hgdp_tgp_sgdp_high_cov_ancients_chr20_q.dated.trees.tsz  # uncompress the download

# Make a world density map (see python src/plot.py for plots available)
# Make any csv files needed for a specific plot (see python src/analyze_data.py --help for options)
python src/analyze_data.py hgdp_sgdp_ancients_ancestral_geography --chrom 20_q
python src/plot.py world_density --chrom 20_q  # creates files in figures/

# Make a world locations map
python src/analyze_data.py average_pop_ancestors_geography --chrom 20_q
python src/plot.py population_ancestors --chrom 20_q  # creates files in figures/

```

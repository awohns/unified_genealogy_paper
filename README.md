# Code for "A unified genealogy of modern and ancient genomes"

This repo contains code used in ["A unified genealogy of modern and ancient genomes"](https://www.biorxiv.org/content/10.1101/2021.02.16.431497v1).
This includes:
* simulation based validation of [tsinfer](https://tsinfer.readthedocs.io/) and
[tsdate](https://tsdate.readthedocs.io/en/latest/)
* pipelines to generate a unified, dated tree sequence from multiple datasets
* empirical validation of the tree sequence using ancient genomes
* code to produce all non-schematic figures in the paper, as well as the interactive figure and supplementary video

These analyses are placed in subdirectories as follows:
* `all-data` contains all code for downloading and preparing real data, as well as the pipeline for creating tree sequences
* `src` contains scripts to run analyses and validation on both real and simulated data
* `data` contains reference data needed for some analyses and is where the results of analyses performed on the real inferred tree sequences are stored. It is also where simulations evaluating mismatch parameters in tsinfer are performed.
* `simulated-data` contains the results of all simulation-based analyses
* `tools` contains methods used to compare the accuracy of `tsinfer` and `tsdate` as well as tools to process variant data files

You must first clone this repo, including submodules as follows:

```
git clone --recurse-submodules https://github.com/awohns/unified_genealogy_paper.git
cd unified_genealogy_paper
```

#### Required Software

### Installing required python modules

The Python packages required are listed in the ``requirements.txt`` file. These can be 
installed with

```
python -m pip install -r requirements.txt
```

Please use Python 3.8 (`numba' fails to install using pip with Python 3.9).

``cartopy`` (required for generating figures with maps) and ``ffmpeg`` (required to create the movie)  must be installed
separately with conda as follows:

```
conda install -c conda-forge cartopy
conda install -c conda-forge ffmpeg
```

### Required software for preparing real data

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

#### Inferring Tree Sequences from Real Data

Please see the [README](all-data/README.md) in the ``all-data`` directory for details on inferring the combined tree sequence. 


#### Running Simulations

To generate data required for simulation-based figures and to plot the figures themselves, follow this general process: generate data, run analyses, and plot the results. 
The following example will generate Figure 1c:

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

To produce all the simulation data in our paper, run the following, in order

```
python src/run_evaluation.py --setup all 
python src/run_evaluation.py --infer all
```

You can speed up the inference step by using multiple processors, specified using the `-p` flag.
For instance, on a 64 core machine, using all cores:

```
python src/run_evaluation.py infer all -p 64 all # will take a few days
```

The mismatch simulations are produced separately, using the Makefile in the `data` directory:

```
cd data
make mismatch
```

### Analyzing inferred tree sequences

Once you have inferred tree sequences and the results are in the ``all-data`` directory, run functions in ``src/analyze_data.py`` to generate data for non-simulation based figures. The figures themselves are plotted using ``src/plot.py``

Before running these analyses, download other required files (such as allele age estimates from `Relate` and `GEVA`) using the following:

```
cd data
make all
```

You can then run all analyses using real data with:

```
python src/analyze_data.py all
```

### Plotting figures

The final figures can then be plotted using

```
python src/plot.py all
```



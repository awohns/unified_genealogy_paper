# Code for "A unified genealogy of modern and ancient genomes reveals human history"

This repo contains code used in "A unified genealogy of modern and ancient genomes reveals human history."
This includes:
* simulation based validation of [tsinfer](https://tsinfer.readthedocs.io/) and
[tsdate](https://tsdate.readthedocs.io/en/latest/)
* pipelines to generate a unified, dated tree sequence from multiple datasets
* empirical validation of the tree sequence using ancient genomes
* code to produce all non-schematic figures in the paper, as well as the interactive figure and supplementary video

These analyses are placed in subdirectories as follows:
* `all-data` contains all code for downloading and preparing real data, as well as pipeline for creating tree sequences
* `src` contains scripts to run analyses and validation on both real and simulated data
* `data` contains reference data needed for some analyses and is where the results of analyses performed on the real inferred tree sequences are stored
* `simulated-data` contains the results of all simulation-based analyses
* `tools` contains methods used to compare the accuracy of tsinfer and tsdate as well as tools to process variant data files

#### Required Software

### Installing required python modules

The Python packages required are listed in the ``requirements.txt`` file. These can be 
installed with

```
$ python -m pip install -r requirements.txt
```

if you are using pip. Conda may also be used to install these dependencies.

### Installing other tools for simulation-based evaluation

We compare our methods with [Genealogical Estimation of Variant Age (GEVA)](https://github.com/pkalbers/geva) and
[Relate](https://myersgroup.github.io/relate/index.html). These are kept in the ``tools`` directory and can be
downloaded and built using 

```
$ make relate
$ make geva
```

### Required software for preparing real data

We require [BCFtools, SAMtools, and HTSlib](http://www.htslib.org/download/), as well as
[convertf](https://github.com/argriffing/eigensoft/tree/master/CONVERTF) to prepare the variant data files from
the 1000 Genomes Project, Human Genome Diversity Project, Simons Genome Diversity Project, and ancient DNA
datasets for use with `tsinfer` and `tsdate`.

```
$ cd tools
$ make all
```

`bcftools`, `htslib`, and `samtools` need to be added to your path to allow code in `all-data` to run seamlessly.

#### Running Simulations

To generate data required for simulation-based figures and to plot the figures themselves, follow this general process: generate data, run analyses, and plot the results. 
The following is an example this code will generate Figure 1c:

`python src/run_evaluation.py tsdate_neutral_simulated_mutation_accuracy --setup`
`python src/run_evaluation.py tsdate_neutral_simulated_mutation_accuracy --inference`
`python src/plot.py tsdate_simulated_accuracy

The first command runs the simulations for each evaluation. It may take multiple days to run all simulations if
multithreading is not being used. The simulations are stored in the `simulated-data` directory and will take multiple
gigabytes of disk space. The second command performs inference on the results of the simulations. The results are stored as
csv files in the `simulated-data` directory. The third command plots the csv files and saves the resulting figures to the
`figures` directory.


### Running all evaluations

To produce all the simulation data in our paper, run the following, in order

```
python src/run_evaluation.py --setup all 
python src/run_evaluation.py --infer all
```

You can speed up the evaluations by using multiple processors, specified using the `-p` flag.
For instance, on a 64 core machine, using all cores:

```
python src/run_evaluation.py setup -p 64 all # will take a few hours
python src/run_evaluation.py infer -p 64 all # will take a few days (mostly to run ARGweaver)
```

The final figures can then be plotted using

```
python src/plot.py all
```

## Inferring Tree Sequences from Real Data

You will need the `cyvcf2` Python module to read VCF files. Once the requirements above have been installed you should simply be able to do:

```
$ python -m pip install cyvcf2 # only for human data analysis: needs to be installed *after* numpy
```

Please see the [README](all-data/README.md) in the ``all-data`` directory for further details. 

### Analyzing inferred tree sequences

Once you have inferred tree sequences and the results are in the ``all-data`` directory, run functions in ``src/analyze_data.py`` to generate data for non-simulation based figures. The figures themselves are plotted using ``src/plot.py``

For example, the following code will generate Figure 2 (this can take a long time to run, even when multithreaded!). 
```
$ python src/analyze_data.py all_mrcas
$ python src/plot.py tmrca_clustermap
$ python src/plot.py inset_tmrca_histograms
```

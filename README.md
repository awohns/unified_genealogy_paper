# Code for "Efficient Dating of Ancestors in Genealogical Networks with Modern and Ancient Samples"

This repo is the home for code used in "Efficient Dating of Ancestors in Genealogical Networks
with Modern and Ancient Samples." Analyses include: validation of tsdate's accuracy on simulated and real
data, applications of tree sequence inference and dating algorithms to various datasets, 
and code to produce all non-schematic plots in the paper.

The all-data directory contains all code for downloading and preparing data.

src contains scripts to run analyses.

tools contains methods used to compare the accuracy of tsinfer and tsdate.

The general pattern to generate data, run analyses, and plot the resulting analysis 
is as follows (this code will generate the data for Figure 2):

python src/run_evaluation.py simulate_vanilla_ancient --setup

python src/run_evaluation.py simulate_vanilla_ancient --inference --process 10

The following procedure will generate files in the all-data directory:
1. Download each dataset using the bash scripts in each directory (max-planck-data/Chagyrskaya etc.)
2. Generate a SampleData file for each dataset using make Dataset.samples
3. Merge SampleData files using python tsutil.py combine-sampledata sampledatalist combinedsampledata

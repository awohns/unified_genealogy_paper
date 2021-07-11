SEED = 1
NUM_PROCS = 60
SPECIES = HomSap
CONTIG = chr20

downloads: | relate_ages geva_ages.csv.gz mismatch chr20_mask

relate_ages: allele_ages_AFR allele_ages_AMR allele_ages_EAS allele_ages_EUR allele_ages_SAS

chr20_mask: 20160622.chr20.mask.fasta

allele_ages_AFR:
	curl https://zenodo.org/record/3234689/files/allele_ages_AFR.zip?download=1 -o allele_ages_AFR.zip
	unzip allele_ages_AFR.zip

allele_ages_AMR:
	curl https://zenodo.org/record/3234689/files/allele_ages_AMR.zip?download=1 -o allele_ages_AMR.zip
	unzip allele_ages_AMR.zip

allele_ages_EAS:
	curl https://zenodo.org/record/3234689/files/allele_ages_EAS.zip?download=1 -o allele_ages_EAS.zip
	unzip allele_ages_EAS.zip

allele_ages_EUR:
	curl https://zenodo.org/record/3234689/files/allele_ages_EUR.zip?download=1 -o allele_ages_EUR.zip
	unzip allele_ages_EUR.zip

allele_ages_SAS:
	curl https://zenodo.org/record/3234689/files/allele_ages_SAS.zip?download=1 -o allele_ages_SAS.zip
	unzip allele_ages_SAS.zip

geva_ages.csv.gz:
	curl https://human.genome.dating/bulk/atlas.chr20.csv.gz -o geva_ages.csv.gz

20160622.chr20.mask.fasta:
	curl http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/working/20160622_genome_mask_GRCh38/StrictMask/20160622.chr20.mask.fasta.gz -o 20160622.chr20.mask.fasta.gz
	gunzip 20160622.chr20.mask.fasta.gz

############
# Mismatch #
############

# Default simulation settings
MODEL = OutOfAfrica_3G09
NUM_SAMPLES = 1500

OOA = $(MODEL)_$(CONTIG)_n$(NUM_SAMPLES)_seed$(SEED)

# Default real data settings
# Subset the real sample files down to a limited number of sites (these should not span the centromere)
START_SITE = 1000000
END_SITE = 1100000

DETAILS = $(CONTIG)_$(START_SITE)-$(END_SITE)

mismatch: $(OOA)_results_plus_RF.csv $(OOA)_ae0.01_results_plus_RF.csv \
	1kg_$(DETAILS)_results.csv hgdp_$(DETAILS)_results.csv

clean:
	rm -f $(MODEL)* 1kg_$(DETAILS)* hgdp_$(DETAILS)*

#### SIMULATIONS ####
#
# Generate the polytomy-split RF distances - this can take a few days
# The -k flag means that we should use (keep) previously run files, so
# if this has been run already, it shouldn't take long
$(OOA)_results_plus_RF.csv: $(OOA)_results.csv
	python3 ../tsinfer-benchmarking/analysis/RFcalc.py $(OOA).trees \
	-s 123 -vv -k -o 20 -p $(NUM_PROCS) $(OOA)_rma*.trees
	python3 ../tsinfer-benchmarking/analysis/add_RF.py \
	$(OOA)_results.csv > $(OOA)_results_plus_RF.csv

$(OOA)_ae0.01_results_plus_RF.csv: $(OOA)_ae0.01_results.csv
	python3 ../tsinfer-benchmarking/analysis/RFcalc.py $(OOA).trees \
	-s 123 -vv -k -o 20 -p $(NUM_PROCS) $(OOA)_ae0.01_rma*.trees
	python3 ../tsinfer-benchmarking/analysis/add_RF.py \
	$(OOA)_ae0.01_results.csv > $(OOA)_ae0.01_results_plus_RF.csv
	
# Make the simulation-without-error file: can take over a day or so on a 60 proc machine
$(OOA)_results.csv:
	python3 ../tsinfer-benchmarking/analysis/test_mismatch_rates.py -s $(SEED) \
	$(SPECIES):$(CONTIG):$(MODEL):$(NUM_SAMPLES) -vv -k -p $(NUM_PROCS)

# Make the simulation-with-error file: can take over a day or so on a 60 proc machine
$(OOA)_ae0.01_results.csv:
	python3 ../tsinfer-benchmarking/analysis/test_mismatch_rates.py -s $(SEED) \
	$(SPECIES):$(CONTIG):$(MODEL):$(NUM_SAMPLES) -vv -k -e 0.01 -p $(NUM_PROCS)


#### REAL DATA ####
#

1kg_$(DETAILS)_results.csv: 1kg_$(DETAILS).samples
	python3 ../tsinfer-benchmarking/analysis/test_mismatch_rates.py \
	1kg_$(DETAILS).samples -vv -k -p $(NUM_PROCS)

1kg_$(DETAILS).samples:
	python3 -c 'import tsinfer; sd=tsinfer.load("../all-data/1kg_$(CONTIG).samples"); \
	sd.subset(sites=range($(START_SITE), $(END_SITE)), path="1kg_$(DETAILS).samples")'

hgdp_$(DETAILS)_results.csv: hgdp_$(DETAILS).samples
	python3 ../tsinfer-benchmarking/analysis/test_mismatch_rates.py \
	hgdp_$(DETAILS).samples -vv -k -p $(NUM_PROCS) \
	-m ../all-data/recomb-hg38/genetic_map_GRCh38_$(CONTIG).txt

hgdp_$(DETAILS).samples:
	python3 -c 'import tsinfer; sd=tsinfer.load("../all-data/hgdp_$(CONTIG).samples"); \
	sd.subset(sites=range($(START_SITE), $(END_SITE)), path="hgdp_$(DETAILS).samples")'

clean:
	rm -fR allele_ages_* 1kg_* hgdp_* $(OOA)* geva_ages* chr20_mask.fasta
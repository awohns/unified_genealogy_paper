all: | relate_ages geva_ages

relate_ages: 
	curl https://zenodo.org/record/3234689/files/allele_ages_AFR.zip?download=1 -o allele_ages_AFR.zip
	unzip allele_ages_AFR.zip
	curl https://zenodo.org/record/3234689/files/allele_ages_AMR.zip?download=1 -o allele_ages_AMR.zip
	unzip allele_ages_AMR.zip
	curl https://zenodo.org/record/3234689/files/allele_ages_EAS.zip?download=1 -o allele_ages_EAS.zip
	unzip allele_ages_EAS.zip
	curl https://zenodo.org/record/3234689/files/allele_ages_EUR.zip?download=1 -o allele_ages_EUR.zip
	unzip allele_ages_EUR.zip
	curl https://zenodo.org/record/3234689/files/allele_ages_SAS.zip?download=1 -o allele_ages_SAS.zip
	unzip allele_ages_SAS.zip

geva_ages:
	curl https://human.genome.dating/bulk/atlas.chr20.csv.gz -o geva_ages.csv.gz

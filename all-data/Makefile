NUM_THREADS ?= 0

# Requirements: bcftools, tabix, convertf, python3.
# See requirements.txt for Python package requirements.
#
help:
	@echo Makefile to create dated tree sequences used in paper

all: 1kg_chr20.dated.trees hgdp_1kg_sgdp_high_cov_ancients_dated_chr20.trees

%.bcf.csi: %.bcf
	bcftools index $(patsubst %.bcf.csi,%.bcf,$@)

%.vcf.gz.csi: %.vcf.gz
	bcftools index $(patsubst %.vcf.gz.csi,%.vcf.gz,$@)

# Save all intermediate files
.SECONDARY:


####################################################
# Standard pipeline for samples file to .dated.trees
####################################################

%.missing_binned.samples: %.samples
	python3 bin_missing.py $^ $@

%.trees: %.samples recomb-hg38/
	python3 ../src/run_inference.py $< -t ${NUM_THREADS} -A 1 -S 1 -m recomb-hg38/genetic_map_GRCh38_
	python3 tsutil.py simplify $*.nosimplify.trees $@

%.dated.trees: %.trees
	python3 -m tsdate preprocess $^ $*.preprocessed.trees
	python3 -m tsdate date $*.preprocessed.trees $@ 10000 -m 1e-8 -p -t ${NUM_THREADS} --ignore-oldest

%.dated.samples: %.samples %.dated.trees
	python3 tsutil.py dated_samples $^

%.binned.samples: %.dated.samples
	python3 bin_dates.py $^ $@

%.dated.samples: %.samples %.modern.dated.trees %.modern.dates.p
	python3 get_dated_sampledata.py $^


#############################################
# Download all prerequisite files
# #############################################

%_download: hg38.fa hg19ToHg38.over.chain.gz homo_sapiens_ancestor_GRCh38.tar.gz \
	homo_sapiens_ancestor_GRCh37_e71.tar.bz2 1kg_samples.ped 1kg_%_genotypes.vcf.gz \
	1kg_GRCh38_%_genotypes.vcf.gz sgdp_samples.txt sgdp_%_genotypes.vcf.gz \
	hgdp_samples.txt hgdp_genotypes.vcf.gz denisovan.%_mq25_mapab100.vcf.gz \
	vindija.%_mq25_mapab100.vcf.gz altai.%_mq25_mapab100.vcf.gz \
	ust_ishim.%_mq25_mapab100.vcf.gz chagyrskaya.%.noRB.vcf.gz \
	lbk.%_mq25_mapab100.vcf.gz loshbour.%_mq25_mapab100.vcf.gz v42.4.1240K.tar
	@echo Downloaded variant data used to create tree sequences


#############################################
# hg38.fa reference genome
#############################################

hg38.fa:
	curl https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz -o hg38.fa.gz
	gunzip -c hg38.fa.gz > hg38.fa
	java -jar ../tools/picard.jar CreateSequenceDictionary \ 
	      R=hg38.fa \ 
	      O=hg38.dict


#############################################
# hg19 to hg39 LiftOver File
#############################################

hg19ToHg38.over.chain.gz:
	curl https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz -o $@


#############################################
# Ancestral states from Ensembl
#############################################

# HGDP is in GRCh38, and tgp has a GRCh38 liftover available. Others we can lift over. 
# So we download the ancestral states for GRCh38. 

# Recorded in the sample file provenance.
REFERENCE_NAME=GRCh38

ANCESTRAL_STATES_PREFIX=homo_sapiens_ancestor_GRCh38
ANCESTRAL_STATES_TARBALL=${ANCESTRAL_STATES_PREFIX}.tar.gz
ANCESTRAL_STATES_URL=ftp://ftp.ensembl.org/pub/release-100/fasta/ancestral_alleles/${ANCESTRAL_STATES_TARBALL}

${ANCESTRAL_STATES_TARBALL}:
	curl ${ANCESTRAL_STATES_URL} -o ${ANCESTRAL_STATES_TARBALL}

${ANCESTRAL_STATES_PREFIX}/README: ${ANCESTRAL_STATES_TARBALL}
	rm -fR ${ANCESTRAL_STATES_PREFIX}
	tar -xvzf ${ANCESTRAL_STATES_TARBALL}
	# Update access times or we'll keep rebuilding this rule. Have to make sure 
	# that the README we touch is older than the actual fa files.
	touch $@
	touch ${ANCESTRAL_STATES_PREFIX}/*.fa

chr%_ancestral_states.fa: ${ANCESTRAL_STATES_PREFIX}/README
	ln -sf ${ANCESTRAL_STATES_PREFIX}/homo_sapiens_ancestor_$*.fa $@

chr%_ancestral_states.fa.fai: chr%_ancestral_states.fa
	samtools faidx $^

# Other datasets are in GRCh37
# Download the ancestral states for GRCh37. 

# Recorded in the sample file provenance.
REFERENCE_NAME_37=GRCh37

ANCESTRAL_STATES_PREFIX_37=homo_sapiens_ancestor_GRCh37_e71
ANCESTRAL_STATES_TARBALL_37=${ANCESTRAL_STATES_PREFIX_37}.tar.bz2
ANCESTRAL_STATES_URL_37=ftp://ftp.ensembl.org/pub/release-75/fasta/ancestral_alleles/${ANCESTRAL_STATES_TARBALL_37}

${ANCESTRAL_STATES_TARBALL_37}:
	curl ${ANCESTRAL_STATES_URL_37} -o ${ANCESTRAL_STATES_TARBALL_37}

${ANCESTRAL_STATES_PREFIX_37}/README: ${ANCESTRAL_STATES_TARBALL_37}
	rm -fR ${ANCESTRAL_STATES_PREFIX_37}
	tar -jxvf ${ANCESTRAL_STATES_TARBALL_37}
	# Update access times or we'll keep rebuilding this rule. Have to make sure 
	# that the README we touch is older than the actual fa files.
	touch $@
	touch ${ANCESTRAL_STATES_PREFIX_37}/*.fa

chr%_ancestral_states_37.fa: ${ANCESTRAL_STATES_PREFIX_37}/README
	ln -sf ${ANCESTRAL_STATES_PREFIX_37}/homo_sapiens_ancestor_$*.fa $@

chr%_ancestral_states_37.fa.fai: chr%_ancestral_states_37.fa
	samtools faidx $^

###########################
# GRCh38 Recombination Maps
###########################

recomb-hg38/:
	wget http://csg.sph.umich.edu/locuszoom/download/recomb-hg38.tar.gz
	tar -xvzf recomb-hg38.tar.gz
	./modify_genetic_map.sh

#############################################
# 1000 Genomes data.
#############################################

GENOTYPES_VCF_BASE=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
GENOTYPES_BCF_BASE=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/bcf_files

1kg_samples.ped:
	curl http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_g1k.ped \
		-o $@

1kg_%_genotypes.vcf.gz:
	curl ${GENOTYPES_VCF_BASE}/ALL.$*.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz -o $@
	tabix -p vcf -f $@

1kg_%.samples: 1kg_%_genotypes.vcf.gz %_ancestral_states_37.fai 1kg_samples.ped
	python3 convert.py 1kg -p \
		1kg_$*_genotypes.vcf.gz \
		$*_ancestral_states_37.fa \
		-m 1kg_samples.ped \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads=${NUM_THREADS} \
		$@ > $@.report

1kg_chr20.trees: 1kg_chr20.samples
	python3 ../src/run_inference.py $^ -t ${NUM_THREADS} -A 0.1 -S 0.1


#############################################
# 1000 Genomes GRCh38 data.
#############################################

GENOTYPES_BASE_GRCH38=ftp://ftp.sra.ebi.ac.uk/vol1/ERZ822/ERZ822766/

1kg_GRCh38_%_genotypes.vcf.gz:
	curl ${GENOTYPES_BASE_GRCH38}ALL.$*.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz -o $@
	tabix -p vcf $@ 

1kg_GRCh38_%.samples: 1kg_GRCh38_%_genotypes.vcf.gz %_ancestral_states.fa.fai 1kg_samples.ped
	python3 convert.py 1kg -p \
		1kg_GRCh38_$*_genotypes.vcf.gz \
		$*_ancestral_states.fa \
		-m 1kg_samples.ped \
		--ancestral-states-url=${ANCESTRAL_STATES_URL} \
		--reference-name=${REFERENCE_NAME} \
		--num-threads ${NUM_THREADS} \
		$@  > $@.report


#############################################
# SGDP data.
#############################################

SGDP_GENOTYPES_BASE=https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/phased_data/PS2_multisample_public

sgdp_samples.txt:
	curl https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/SGDP_metadata.279public.21signedLetter.samples.txt -o $@

sgdp_%_genotypes.vcf.gz:
	curl ${SGDP_GENOTYPES_BASE}/cteam_extended.v4.PS2_phase.public.$*.vcf.gz -o $@
	curl ${SGDP_GENOTYPES_BASE}/cteam_extended.v4.PS2_phase.public.$*.vcf.gz.csi -o $@.csi

sgdp_%_genotypes.bcf: sgdp_%_genotypes.vcf.gz
	# Remove the S_Naxi-2 individual because (a) it doesn't have any metadata in the 
	# file we're using and (b) it has a massively elevated sample edge count if we 
	# leave it in.
	bcftools view -s '^S_Naxi-2' $^ -O b -o $@

sgdp_%_genotypes_GRCh38.vcf.gz: sgdp_%_genotypes.vcf.gz hg38.fa hg19ToHg38.over.chain.gz
	gunzip -c sgdp_$*_genotypes.vcf.gz > sgdp_$*.vcf
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' sgdp_$*.vcf > sgdp_$*.withchr.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=sgdp_$*.withchr.vcf O=sgdp_GRCh38_$*.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=sgdp_GRCh38_$*.rejected_variants.vcf R=hg38.fa
	rm sgdp_$*.withchr.vcf
	bgzip -c sgdp_GRCh38_$*.vcf > sgdp_$*_genotypes_GRCh38.all.vcf.gz 
	rm sgdp_GRCh38_$*.vcf
	tabix -p vcf sgdp_$*_genotypes_GRCh38.all.vcf.gz
	bcftools view sgdp_$*_genotypes_GRCh38.all.vcf.gz --regions $* -O z -o $@
	tabix -p vcf $@
	rm sgdp_$*_genotypes_GRCh38.all.vcf.gz

sgdp_%_genotypes_GRCh38.bcf: sgdp_%_genotypes_GRCh38.vcf.gz
	# Remove the S_Naxi-2 individual because (a) it doesn't have any metadata in the 
	# file we're using and (b) it has a massively elevated sample edge count if we 
	# leave it in.
	bcftools view -s '^S_Naxi-2' $^ -O b -o $@

sgdp_%.samples: sgdp_%_genotypes.bcf.csi %_ancestral_states_37.fai sgdp_samples.txt
	python3 convert.py sgdp -p \
		sgdp_$*_genotypes.bcf \
		$*_ancestral_states_37.fa \
		-m sgdp_samples.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads=1 \
		$@  > $@.report

sgdp_GRCh38_%.samples: sgdp_%_genotypes_GRCh38.bcf.csi %_ancestral_states.fa.fai sgdp_samples.txt
	python3 convert.py sgdp -p \
		sgdp_$*_genotypes_GRCh38.bcf \
		$*_ancestral_states.fa \
		-m sgdp_samples.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL} \
		--reference-name=${REFERENCE_NAME} \
		--num-threads=1 \
		$@  > $@.report


#############################################
# HGDP Data 
#############################################

hgdp_samples.txt:
		curl ftp://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/metadata/hgdp_wgs.20190516.metadata.txt -o $@

HGDP_GENOTYPES_BASE=ftp://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/statphase/

hgdp_genotypes.vcf.gz:
		curl ${HGDP_GENOTYPES_BASE}/hgdp_wgs.20190516.statphase.autosomes.vcf.gz -o $@
			curl ${HGDP_GENOTYPES_BASE}/hgdp_wgs.20190516.statphase.autosomes.vcf.gz.tbi -o $@.tbi

hgdp_genotypes.%.phased.GRCh38.vcf.gz: hgdp_genotypes.vcf.gz
		tabix -h $^ ${*} | bgzip -c > $@
			tabix -p vcf $@

hgdp_genotypes.%.phased.GRCh38.bcf: hgdp_genotypes.%.phased.GRCh38.vcf.gz
		bcftools view $^ -O b -o hgdp_genotypes.${*}.phased.GRCh38.bcf
			bcftools index hgdp_genotypes.${*}.phased.GRCh38.bcf

hgdp_%.samples: hgdp_genotypes.%.phased.GRCh38.vcf.gz %_ancestral_states.fa.fai hgdp_samples.txt 
	python3 convert.py hgdp -p \
		hgdp_genotypes.$*.phased.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		-m hgdp_samples.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL} \
		--reference-name=${REFERENCE_NAME} \
		--num-threads=${NUM_THREADS} \
		$@ > $@.report

	
#############################################
# Max Planck Data 
#############################################

FILE_SUFFIX=_mq25_mapab100.vcf.gz
CHAGYRSKAYA_SUFFIX=.noRB.vcf.gz

denisovan.%_mq25_mapab100.vcf.gz:
	curl http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Denisova/${*}${FILE_SUFFIX} -o $@

vindija.%_mq25_mapab100.vcf.gz:
	curl http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Vindija33.19/${*}${FILE_SUFFIX} -o $@

altai.%_mq25_mapab100.vcf.gz:
	curl http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Altai/${*}${FILE_SUFFIX} -o $@

ust_ishim.%_mq25_mapab100.vcf.gz:
	curl http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Ust_Ishim/${*}${FILE_SUFFIX} -o $@ 

chagyrskaya.%.noRB.vcf.gz:
	curl http://ftp.eva.mpg.de/neandertal/Chagyrskaya/VCF/${*}${CHAGYRSKAYA_SUFFIX} -o $@

lbk.%_mq25_mapab100.vcf.gz:
	curl http://cdna.eva.mpg.de/neandertal/Vindija/VCF/LBK/${*}${FILE_SUFFIX} -o $@

loshbour.%_mq25_mapab100.vcf.gz:
	curl http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Loschbour/${*}${FILE_SUFFIX} -o $@


altai.%_mq25_mapab100.GRCh38.vcf.gz: altai.%_mq25_mapab100.vcf.gz hg38.fa hg19ToHg38.over.chain.gz
	gunzip -c altai.$*_mq25_mapab100.vcf.gz > altai.$*_mq25_mapab100.vcf	
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' altai.$*_mq25_mapab100.vcf > altai.$*_mq25_mapab100.withchr.vcf
	rm altai.$*_mq25_mapab100.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=altai.$*_mq25_mapab100.withchr.vcf O=altai.$*_mq25_mapab100.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=altai.$*_mq25_mapab100.GRCh38.rejected_variants.vcf R=hg38.fa
	rm altai.$*_mq25_mapab100.withchr.vcf
	bgzip -c altai.$*_mq25_mapab100.GRCh38.vcf > altai.$*_mq25_mapab100.GRCh38.all.vcf.gz 
	tabix -p vcf altai.$*_mq25_mapab100.GRCh38.all.vcf.gz
	bcftools view altai.$*_mq25_mapab100.GRCh38.all.vcf.gz --regions $* -O z -o $@
	tabix -p vcf $@
	rm altai.$*_mq25_mapab100.GRCh38.vcf

altai_GRCh38_%.samples: altai.%_mq25_mapab100.GRCh38.vcf.gz %_ancestral_states.fa.fai altai_metadata.txt
	python3 convert.py max-planck -p \
		altai.$*_mq25_mapab100.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		-m altai_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL} \
		--reference-name=${REFERENCE_NAME} \
		--num-threads ${NUM_THREADS} \
		--target-samples=hgdp_1kg_sgdp_$*.samples \
		$@ > $@.report

altai_%.samples: altai.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fai altai_metadata.txt
	python3 convert.py max-planck -p \
		altai.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m altai_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads ${NUM_THREADS} \
		$@ > $@.report

chagyrskaya.%.noRB.GRCh38.vcf.gz: chagyrskaya.%.noRB.vcf.gz hg38.fa hg19ToHg38.over.chain.gz
	gunzip -c chagyrskaya.$*.noRB.vcf.gz > chagyrskaya.$*.noRB.vcf	
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' chagyrskaya.$*.noRB.vcf > chagyrskaya.$*.noRB.withchr.vcf
	rm chagyrskaya.$*.noRB.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=chagyrskaya.$*.noRB.withchr.vcf O=chagyrskaya.$*.noRB.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=chagyrskaya.$*.noRB.GRCh38.rejected_variants.vcf R=hg38.fa
	rm chagyrskaya.$*.noRB.withchr.vcf
	bgzip -c chagyrskaya.$*.noRB.GRCh38.vcf > chagyrskaya.$*.noRB.GRCh38.all.vcf.gz 
	tabix -p vcf chagyrskaya.$*.noRB.GRCh38.all.vcf.gz
	bcftools view chagyrskaya.$*.noRB.GRCh38.all.vcf.gz --regions $* -O z -o $@
	tabix -p vcf $@
	rm chagyrskaya.$*.noRB.GRCh38.vcf

chagyrskaya_%.samples: chagyrskaya.%.noRB.vcf.gz %_ancestral_states_37.fai chagyrskaya_metadata.txt
	python3 convert.py max-planck -p \
		chagyrskaya.$*.noRB.vcf.gz \
		$*_ancestral_states_37.fa \
		-m chagyrskaya_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads ${NUM_THREADS} \
		$@ > $@.report

chagyrskaya_GRCh38_%.samples: chagyrskaya.%.noRB.GRCh38.vcf.gz %_ancestral_states.fa.fai chagyrskaya_metadata.txt
	python3 convert.py max-planck -p \
		chagyrskaya.$*.noRB.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		-m chagyrskaya_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL} \
		--reference-name=${REFERENCE_NAME} \
		--num-threads ${NUM_THREADS} \
		--target-samples=hgdp_1kg_sgdp_$*.samples \
		$@ > $@.report

denisovan.%_mq25_mapab100.GRCh38.vcf.gz: denisovan.%_mq25_mapab100.vcf.gz hg38.fa hg19ToHg38.over.chain.gz
	gunzip -c denisovan.$*_mq25_mapab100.vcf.gz > denisovan.$*_mq25_mapab100.vcf
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' denisovan.$*_mq25_mapab100.vcf > denisovan.$*_mq25_mapab100.withchr.vcf
	rm denisovan.$*_mq25_mapab100.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=denisovan.$*_mq25_mapab100.withchr.vcf O=denisovan.$*_mq25_mapab100.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=denisovan.$*_mq25_mapab100.GRCh38.rejected_variants.vcf R=hg38.fa
	rm denisovan.$*_mq25_mapab100.withchr.vcf
	bgzip -c denisovan.$*_mq25_mapab100.GRCh38.vcf > denisovan.$*_mq25_mapab100.GRCh38.all.vcf.gz 
	tabix -p vcf denisovan.$*_mq25_mapab100.GRCh38.all.vcf.gz
	bcftools view denisovan.$*_mq25_mapab100.GRCh38.all.vcf.gz --regions $* -O z -o $@
	tabix -p vcf $@
	rm denisovan.$*_mq25_mapab100.GRCh38.vcf
	rm denisovan.$*_mq25_mapab100.GRCh38.all.vcf.gz

denisovan_GRCh38_%.samples: denisovan.%_mq25_mapab100.GRCh38.vcf.gz %_ancestral_states.fa.fai denisovan_metadata.txt hgdp_1kg_sgdp_%.samples
	python3 convert.py max-planck -p \
		denisovan.$*_mq25_mapab100.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		-m denisovan_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL} \
		--reference-name=${REFERENCE_NAME} \
		--num-threads=${NUM_THREADS} \
		--target-samples=hgdp_1kg_sgdp_$*.samples \
		$@ > $@.report

denisovan_%.samples: denisovan.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fai denisovan_metadata.txt
	python3 convert.py max-planck -p \
		denisovan.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m denisovan_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads ${NUM_THREADS} \
		$@ > $@.report

vindija.%_mq25_mapab100.GRCh38.vcf.gz: vindija.%_mq25_mapab100.vcf.gz hg38.fa hg19ToHg38.over.chain.gz
	gunzip -c vindija.$*_mq25_mapab100.vcf.gz > vindija.$*_mq25_mapab100.vcf	
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' vindija.$*_mq25_mapab100.vcf > vindija.$*_mq25_mapab100.withchr.vcf
	rm vindija.$*_mq25_mapab100.vcf 
	java -jar ../tools/picard.jar LiftoverVcf I=vindija.$*_mq25_mapab100.withchr.vcf O=vindija.$*_mq25_mapab100.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=vindija.$*_mq25_mapab100.GRCh38.rejected_variants.vcf R=hg38.fa
	rm vindija.$*_mq25_mapab100.withchr.vcf
	bgzip -c vindija.$*_mq25_mapab100.GRCh38.vcf > vindija.$*_mq25_mapab100.GRCh38.all.vcf.gz 
	tabix -p vcf vindija.$*_mq25_mapab100.GRCh38.all.vcf.gz
	bcftools view vindija.$*_mq25_mapab100.GRCh38.all.vcf.gz --regions $* -O z -o $@
	tabix -p vcf $@
	rm vindija.$*_mq25_mapab100.GRCh38.vcf

vindija_%.samples: vindija.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fai vindija_metadata.txt
	python3 convert.py max-planck -p \
		vindija.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m vindija_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads ${NUM_THREADS} \
		$@ > $@.report

vindija_GRCh38_%.samples: vindija.%_mq25_mapab100.GRCh38.vcf.gz %_ancestral_states.fa.fai vindija_metadata.txt
	python3 convert.py max-planck -p \
		vindija.$*_mq25_mapab100.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		-m vindija_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL} \
		--reference-name=${REFERENCE_NAME} \
		--num-threads ${NUM_THREADS} \
		--target-samples=hgdp_1kg_sgdp_$*.samples \
		$@ > $@.report

ust_ishim_%.samples: ust_ishim.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fai ust_ishim_metadata.txt
	python3 convert.py max-planck -p \
		ust_ishim.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m ust_ishim_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads ${NUM_THREADS} \
		$@ > $@.report

loshbour_%.samples: loshbour.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fai loshbour_metadata.txt
	python3 convert.py max-planck -p \
		loshbour.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m loshbour_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads ${NUM_THREADS} \
		$@ > $@.report

lbk_%.samples: lbk.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fai lbk_metadata.txt
	python3 convert.py max-planck -p \
		lbk.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m lbk_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads ${NUM_THREADS} \
		$@ > $@.report


#############################################
# Afanasievo Data
#############################################
VCF_SUFFIX=.phased.detailed.filtered

AfanasievoFamily_%.phased.detailed.filtered.GRCh38.vcf.gz: AfanasievoFamily_%${VCF_SUFFIX}.vcf.gz hg38.fa hg19ToHg38.over.chain.gz
	gunzip -c $< > AfanasievoFamily_$*${VCF_SUFFIX}.vcf
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' AfanasievoFamily_$*${VCF_SUFFIX}.vcf > AfanasievoFamily_$*${VCF_SUFFIX}.withchr.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=AfanasievoFamily_$*${VCF_SUFFIX}.withchr.vcf O=AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=AfanasievoFamily_$*${VCF_SUFFIX}.phased.GRCh38.rejected_variants.vcf R=hg38.fa
	rm AfanasievoFamily_$*${VCF_SUFFIX}.vcf
	rm AfanasievoFamily_$*${VCF_SUFFIX}.withchr.vcf
	bgzip -c AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf > AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf.gz
	tabix -p vcf AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf.gz
	bcftools view AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf.gz --regions $* -O z > $@
	tabix -p vcf $@
	rm AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf*

afanasievo_GRCh38_%.samples: AfanasievoFamily_%.phased.detailed.filtered.GRCh38.vcf.gz %_ancestral_states.fa.fai
	python3 convert.py afanasievo -p \
		AfanasievoFamily_$*.phased.detailed.filtered.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		--ancestral-states-url=${ANCESTRAL_STATES_URL} \
		--reference-name=${REFERENCE_NAME} \
		--num-threads ${NUM_THREADS} \
		--target-samples=hgdp_1kg_sgdp_$*.samples \
		$@ > $@.report


#############################################
# 1240k Data
#############################################

REICH_PREFIX=v42.4.1240K
REICH_TARBALL=${REICH_PREFIX}.tar
REICH_URL=https://reichdata.hms.harvard.edu/pub/datasets/amh_repo/curated_releases/V42/V42.4/SHARE/public.dir/${REICH_TARBALL}

v42.4.1240K.tar:
	curl ${REICH_URL} -o $@

v42.4.1240K.geno v42.4.1240K.anno: v42.4.1240K.tar
	tar -xvf ${REICH_TARBALL}
	touch v42.4.1240K.geno
	touch v42.4.1240K.anno

v42.4.1240K.vcf.gz: v42.4.1240K.geno
	../tools/eigensoft/src/convertf -p par.PACKEDANCESTRYMAP.PACKEDPED
	mv ${REICH_PREFIX}.pedsnp ${REICH_PREFIX}.bim
	mv ${REICH_PREFIX}.pedind ${REICH_PREFIX}.fam
	plink --bfile ${REICH_PREFIX} --recode vcf --out ${REICH_PREFIX}
	bgzip -c ${REICH_PREFIX}.vcf > $@
	tabix -p vcf $@
	rm ${REICH_PREFIX}.vcf

v42.4.1240K_chr%.vcf.gz: v42.4.1240K.vcf.gz
	bcftools view ${REICH_PREFIX}.vcf.gz --regions $* -O z -o $@

reich_%.samples: v42.4.1240K_%.vcf.gz %_ancestral_states_37.fai v42.4.1240K.anno
	python3 convert.py 1240k -p \
		v42.4.1240K_%.vcf.gz \
		$*_ancestral_states_37.fa \
		-m v42.4.1240K.anno \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
		--reference-name=${REFERENCE_NAME_37} \
		--num-threads=${NUM_THREADS} \
		$@ > $@.report

reich_ancients_%.samples: reich_%.samples
	python3 tsutil.py remove-moderns-reich $^ $@

v42.4.1240K_GRCh38_%.vcf.gz: v42.4.1240K_%.vcf.gz hg38.fa hg19ToHg38.over.chain.gz
	gunzip -c v42.4.1240K_$*.vcf.gz > v42.4.1240K_$*.vcf
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' v42.4.1240K_$*.vcf > v42.4.1240K_$*.withchr.vcf
	java -jar ../tools/picard.jar LiftoverVcf \
		INPUT=v42.4.1240K_$*.withchr.vcf \
		OUTPUT=v42.4.1240K_GRCh38_$*.vcf \
		CHAIN=hg19ToHg38.over.chain.gz \
		REJECT=v42.4.1240K_GRCh38_$*.rejected_variants.vcf \
		R=hg38.fa \
		RECOVER_SWAPPED_REF_ALT=true
	bgzip -c v42.4.1240K_GRCh38_$*.vcf > v42.4.1240K_GRCh38_$*.all.vcf.gz 
	rm v42.4.1240K_GRCh38_$*.vcf
	tabix -p vcf v42.4.1240K_GRCh38_$*.all.vcf.gz
	bcftools view v42.4.1240K_GRCh38_$*.all.vcf.gz --regions $* -O z > $@
	tabix -p vcf $@
	rm v42.4.1240K_GRCh38_$*.all.vcf.gz 

reich_GRCh38_%.samples: v42.4.1240K_GRCh38_%.vcf.gz %_ancestral_states.fa.fai v42.4.1240K.anno
	python3 convert.py 1240k -p \
		v42.4.1240K_GRCh38_$*.vcf.gz \
		$*_ancestral_states.fa \
		-m v42.4.1240K.anno \
		--ancestral-states-url=${ANCESTRAL_STATES_URL} \
		--reference-name=${REFERENCE_NAME} \
		--num-threads=${NUM_THREADS} \
		$@ > $@.report

reich_ancients_GRCh38_%.samples: reich_GRCh38_%.samples
	python3 tsutil.py remove-moderns-reich $^ $@


##############################################
# Iterative Approach for Unified Tree Sequence
##############################################

hgdp_1kg_sgdp_%.samples: hgdp_%.samples 1kg_GRCh38_%.samples sgdp_GRCh38_%.samples
	python3 tsutil.py merge-sampledata-files --input-sampledata $^ --output $@

hgdp_1kg_sgdp_high_cov_ancients_%.samples: hgdp_1kg_sgdp_%.samples afanasievo_GRCh38_%.samples denisovan_GRCh38_%.samples vindija_GRCh38_%.samples chagyrskaya_GRCh38_%.samples altai_GRCh38_%.samples
	python3 tsutil.py make-sampledata-compatible --input-sampledata $^
	python3 tsutil.py merge-sampledata-files --input-sampledata $< afanasievo_GRCh38_$*.subset.samples denisovan_GRCh38_$*.subset.samples vindija_GRCh38_$*.subset.samples chagyrskaya_GRCh38_$*.subset.samples altai_GRCh38_$*.subset.samples --output $@

hgdp_1kg_sgdp_all_ancients_%.samples: hgdp_1kg_sgdp_high_cov_ancients_%.samples reich_ancients_GRCh38_%.samples
	python3 tsutil.py make-sampledata-compatible --input-sampledata $^
	python3 tsutil.py merge-sampledata-files --input-sampledata $< reich_ancients_GRCh38_$*.subset.samples --output $@

hgdp_1kg_sgdp_high_cov_ancients_dated_%.samples: hgdp_1kg_sgdp_high_cov_ancients_%.samples hgdp_1kg_sgdp_all_ancients_%.samples hgdp_1kg_sgdp_%.missing_binned.dated.trees
	python3 tsutil.py combined-ts-dated-samples --high-cov hgdp_1kg_sgdp_high_cov_ancients_$*.samples --all-samples hgdp_1kg_sgdp_all_ancients_$*.samples --dated-ts hgdp_1kg_sgdp_$*.missing_binned.dated.trees --output $@ > hgdp_1kg_sgdp_high_cov_ancients_dated_$*_constrained_variants.txt

hgdp_1kg_sgdp_high_cov_ancients_dated_%.trees: hgdp_1kg_sgdp_high_cov_ancients_dated_%.samples recomb-hg38/
	python3 bin_dates.py $< hgdp_1kg_sgdp_high_cov_ancients_dated_$*.binned.samples
	python3 ../src/run_inference.py hgdp_1kg_sgdp_high_cov_ancients_dated_$*.binned.samples -t ${NUM_THREADS} -A 1 -S 1 -m recomb-hg38/genetic_map_GRCh38_
	python3 tsutil.py simplify hgdp_1kg_sgdp_high_cov_ancients_dated_$*.binned.nosimplify.trees $@

clean:
	rm -f 1kg_samples.ped sgdp_samples.txt *.vcf* *.samples*


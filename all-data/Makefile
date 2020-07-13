NUM_THREADS ?= 0

# Requirements: bcftools, tabix, python3.
# See requirements.txt for Python package requirements.
#
help:
	@echo WRITE SOME HELP

all: 1kg_chr20.samples sgdp_chr20.samples


simplebgen: setup.py simplebgenmodule.c
	python3 setup.py build_ext --inplace

%.bcf.csi: %.bcf
	bcftools index $(patsubst %.bcf.csi,%.bcf,$@)

%.vcf.gz.csi: %.vcf.gz
	bcftools index $(patsubst %.vcf.gz.csi,%.vcf.gz,$@)

# Any implicit targets that are built will be deleted unless we make them as 'precious'.
.PRECIOUS: \
	%.ancestors %.ancestors.trees %.nosimplify.trees %.nosimplify.nopc.trees %.trees %.trees.gz\
	%.bcf.csi %.vcf.gz.csi \
	chr%_ancestral_states.fa \
	1kg_%_genotypes.bcf 1kg_%.samples

#############################################
# Standard pipeline for samples file to .trees
#############################################

%.ancestors: %.samples
	python3 -m tsinfer ga -vp -t ${NUM_THREADS} $^

%.ancestors.trees: %.ancestors
	python3 -m tsinfer ma -vp -t ${NUM_THREADS} $*.samples

%.nosimplify.trees: %.ancestors.trees
	python3 -m tsinfer ms -vp -t ${NUM_THREADS} $*.samples -O $@ --no-simplify

%.nosimplify.nopc.trees: %.ancestors.trees
	python3 -m tsinfer ms -vp -t ${NUM_THREADS} $*.samples -O $@ --no-simplify --no-path-compression

%.trees: %.nosimplify.trees
	python3 tsutil.py simplify $^ $@

%.trees.gz: %.trees
	gzip -c $^ > $@

%.trees.tsz: %.trees
	tszip -k $^ 

%.trees.bcf: %.trees
	msp vcf -P 2 $^ | bcftools view - -O b -o $@

%.snipped.trees: %.trees ${CENTROMERES_CSV}
	python3 tsutil.py snip-centromere $< $@ $* ${CENTROMERES_CSV}

#############################################
# Centromere locations for GRCh38 (aka hg38) from UCSC
#############################################
CENTROMERES_CSV=centromeres.csv
${CENTROMERES_CSV}:
	 curl http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz > cytoband.txt.gz
	 echo "chrom,start,end" > ${CENTROMERES_CSV}
	 # Start and end coordinates are on different lines, so we merge them.
	 zcat cytoband.txt.gz | grep acen | sort | paste -d " " - - \
		 | cut -f 1,2,7 --output-delim="," >> ${CENTROMERES_CSV}

#############################################
# Centromere locations for GRCh37 (aka hg19) from UCSC
# See https://www.biostars.org/p/2349/
#############################################
CENTROMERES_GRCH37_CSV=centromeres_GRCh37.csv
${CENTROMERES_GRCH37_CSV}:
	 curl http://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/cytoBand.txt.gz > cytoband.txt.gz
	 echo "chrom,start,end" > ${CENTROMERES_GRCH37_CSV}
	 # Start and end coordinates are on different lines, so we merge them.
	 zcat cytoband.txt.gz | grep acen | sort | paste -d " " - - \
		 | cut -f 1,2,7 --output-delim="," >> ${CENTROMERES_GRCH37_CSV}

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

#############################################
# 1000 Genomes data.
#############################################

GENOTYPES_BASE=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/bcf_files

1kg_samples.ped:
	curl http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_g1k.ped \
		-o $@
1kg_%_genotypes.bcf:
	curl ${GENOTYPES_BASE}/ALL.$*.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf -o $@

1kg_%_allsites_new.samples: 1kg_%_genotypes.bcf.csi %_ancestral_states_37.fa 1kg_samples.ped
	python3 convert.py 1kg -p \
		1kg_$*_genotypes.bcf \
		$*_ancestral_states_37.fa \
		-m 1kg_samples.ped \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@  > $@.report

1kg_chr20.snipped.trees.gnn.csv: 1kg_chr20.snipped.trees
	python3 tsutil.py compute-1kg-gnn $^ $@ --num-threads=16

#############################################
# 1000 Genomes GRCh38 data.
#############################################

GENOTYPES_BASE_GRCH38=ftp://ftp.sra.ebi.ac.uk/vol1/ERZ822/ERZ822766/

1kg_GRCh38_%_genotypes.vcf.gz:
	curl ${GENOTYPES_BASE_GRCH38}/ALL.$*.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz -o $@
	tabix -p vcf $@ 

1kg_GRCh38_%_allsites.samples: 1kg_GRCh38_%_genotypes.vcf.gz %_ancestral_states.fa 1kg_samples.ped
	python3 convert.py 1kg -p \
		1kg_GRCh38_$*_genotypes.vcf.gz \
		$*_ancestral_states.fa \
		-m 1kg_samples.ped \
		--ancestral-states-url=${ANCESTRAL_STATES_URL}\
		--reference-name=${REFERENCE_NAME}\
		--num-threads 20\
		$@  > $@.report

1kg_GRCh38_chr20.snipped.trees.gnn.csv: 1kg_chr20.snipped.trees
	python3 tsutil.py compute-1kg-gnn $^ $@ --num-threads=16

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

sgdp_%.samples: sgdp_%_genotypes.bcf.csi %_ancestral_states_37.fa sgdp_samples.txt
	python3 convert.py sgdp -p \
		sgdp_$*_genotypes.bcf \
		$*_ancestral_states_37.fa \
		-m sgdp/sgdp_samples.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@  > $@.report

sgdp_%.snipped.trees.gnn.csv: sgdp_%.snipped.trees
	python3 tsutil.py compute-sgdp-gnn $^ $@ --num-threads=16

#############################################
# HGDP Data 
#############################################

hgdp/hgdp_samples.txt:
		curl ftp://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/metadata/hgdp_wgs.20190516.metadata.txt -o $@

HGDP_GENOTYPES_BASE=ftp://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/statphase/

hgdp/hgdp_genotypes.vcf.gz:
		curl ${HGDP_GENOTYPES_BASE}/hgdp_wgs.20190516.statphase.autosomes.vcf.gz -o $@
			curl ${HGDP_GENOTYPES_BASE}/hgdp_wgs.20190516.statphase.autosomes.vcf.gz.tbi -o $@.tbi

hgdp/hgdp_genotypes.%.phased.GRCh38.vcf.gz:
		tabix -h hgdp/hgdp_genotypes.vcf.gz ${*} | bgzip -c > $@
			tabix -p vcf $@

hgdp/hgdp_genotypes.%.phased.GRCh38.bcf: hgdp/hgdp_genotypes.%.phased.GRCh38.vcf.gz
		bcftools view $^ -O b -o hgdp/hgdp_genotypes.${*}.phased.GRCh38.bcf
			bcftools index hgdp/hgdp_genotypes.${*}.phased.GRCh38.bcf

hgdp_%.samples: hgdp/hgdp_genotypes.%.phased.GRCh38.bcf %_ancestral_states.fa hgdp/hgdp_samples.txt 
	python convert.py hgdp -p \
		hgdp/hgdp_genotypes.$*.phased.GRCh38.bcf \
		$*_ancestral_states.fa \
		-m hgdp/hgdp_samples.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL}\
		--reference-name=${REFERENCE_NAME}\
		$@ > $@.report

hgdp_missing_data_%.samples: hgdp/hgdp_genotypes.%.phased.GRCh38.bcf %_ancestral_states.fa hgdp/hgdp_samples.txt 
	python convert.py hgdp -p \
		hgdp/hgdp_genotypes.$*.phased.GRCh38.bcf \
		$*_ancestral_states.fa \
		-m hgdp/hgdp_samples.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL}\
		--reference-name=${REFERENCE_NAME}\
		--num-threads 20\
		$@ > $@.report

hgdp_%.snipped.trees.gnn.csv: hgdp_%.snipped.trees
	python3 tsutil.py compute-hgdp-gnn $^ $@ --num-threads=16
	
#############################################
# Max Planck Data 
#############################################

ARCHAIC_PATH=max_planck_data

altai.%_mq25_mapab100.GRCh38.vcf.gz: max_planck_data/altai/altai.%_mq25_mapab100.vcf.gz
	gunzip -c ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.vcf.gz > ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.vcf	
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.vcf > ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.withchr.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.withchr.vcf O=${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.rejected_variants.vcf R=hg38.fa
	bgzip -c ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.vcf > ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.vcf.gz 
	rm ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.vcf
	tabix -p vcf ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.vcf.gz

vindija.%_mq25_mapab100.GRCh38.vcf.gz: max_planck_data/vindija/vindija.%_mq25_mapab100.vcf.gz
	gunzip -c ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.vcf.gz > ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.vcf	
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.vcf > ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.withchr.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.withchr.vcf O=${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.rejected_variants.vcf R=hg38.fa
	bgzip -c ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.vcf > ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.vcf.gz 
	rm ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.vcf
	tabix -p vcf ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.vcf.gz

vindija_%.samples: max_planck_data/vindija/vindija.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/vindija/vindija_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/vindija/vindija.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/vindija/vindija_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

vindija_GRCh38_%.samples: ${ARCHAIC_PATH}/vindija/vindija.%_mq25_mapab100.GRCh38.vcf.gz %_ancestral_states.fa max_planck_data/vindija/vindija_metadata.txt
	python convert.py max-planck -p \
		${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		-m ${ARCHAIC_PATH}/vindija/vindija_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL}\
		--reference-name=${REFERENCE_NAME}\
		--num-threads 40\
		$@ > $@.report

altai_GRCh38_%.samples: max_planck_data/altai/altai.%_mq25_mapab100.GRCh38.vcf.gz %_ancestral_states.fa max_planck_data/altai/altai_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/altai/altai.$*_mq25_mapab100.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		-m max_planck_data/altai/altai_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL}\
		--reference-name=${REFERENCE_NAME}\
		--num-threads 40\
		$@ > $@.report

altai_%.samples: max_planck_data/altai/altai.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/altai/altai_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/altai/altai.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/altai/altai_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

chagyrskaya.%.noRB.GRCh38.vcf.gz: ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.%.noRB.vcf.gz
	gunzip -c ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.vcf.gz > ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.vcf	
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.vcf > ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.withchr.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.withchr.vcf O=${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.rejected_variants.vcf R=hg38.fa
	bgzip -c ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.vcf > ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.vcf.gz 
	rm ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.vcf
	tabix -p vcf ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.vcf.gz

chagyrskaya_%.samples: max_planck_data/chagyrskaya/chagyrskaya.%.noRB.vcf.gz %_ancestral_states_37.fa max_planck_data/chagyrskaya/chagyrskaya_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/chagyrskaya/chagyrskaya.$*.noRB.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/chagyrskaya/chagyrskaya_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

chagyrskaya_GRCh38_%.samples: max_planck_data/chagyrskaya/chagyrskaya.%.noRB.GRCh38.vcf.gz %_ancestral_states.fa max_planck_data/chagyrskaya/chagyrskaya_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		-m max_planck_data/chagyrskaya/chagyrskaya_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL}\
		--reference-name=${REFERENCE_NAME}\
		--num-threads 40\
		$@ > $@.report

denisovan.%_mq25_mapab100.GRCh38.vcf.gz: max_planck_data/denisovan/denisovan.%_mq25_mapab100.vcf.gz
	gunzip -c ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.vcf.gz > ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.vcf	
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.vcf > ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.withchr.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.withchr.vcf O=${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.rejected_variants.vcf R=hg38.fa
	bgzip -c ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.vcf > ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.vcf.gz 
	rm ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.vcf
	tabix -p vcf ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.vcf.gz

denisovan_GRCh38_%.samples: ${ARCHAIC_PATH}/denisovan/denisovan.%_mq25_mapab100.GRCh38.vcf.gz %_ancestral_states.fa max_planck_data/denisovan/denisovan_metadata.txt
	python convert.py max-planck -p \
		${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		-m ${ARCHAIC_PATH}/denisovan/denisovan_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL}\
		--reference-name=${REFERENCE_NAME}\
		--num-threads 40\
		$@ > $@.report

denisovan_%.samples: max_planck_data/denisovan/denisovan.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/denisovan/denisovan_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/denisovan/denisovan.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/denisovan/denisovan_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

ust_ishim_%.samples: max_planck_data/ust_ishim/ust_ishim.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/ust_ishim/ust_ishim_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/ust_ishim/ust_ishim.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/ust_ishim/ust_ishim_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

loshbour_%.samples: max_planck_data/loshbour/loshbour.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/loshbour/loshbour_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/loshbour/loshbour.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/loshbour/loshbour_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

lbk_%.samples: max_planck_data/lbk/lbk.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/lbk/lbk_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/lbk/lbk.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/lbk/lbk_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

#############################################
# Afanasievo Data
#############################################
AFANASIEVO_PREFIX=afanasievo/Afanasievo

afanasievo.%.0.999.phased.GRCh38.vcf.gz: ${AFANASIEVO_PREFIX}.%.0.999.phased.vcf.gz
	gunzip -c ${AFANASIEVO_PREFIX}.$*.0.999.phased.vcf.gz > ${AFANASIEVO_PREFIX}.$*.0.999.phased.vcf	
	awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${AFANASIEVO_PREFIX}.$*.0.999.phased.vcf > ${AFANASIEVO_PREFIX}.$*.0.999.phased.withchr.vcf
	java -jar ../tools/picard.jar LiftoverVcf I=${AFANASIEVO_PREFIX}.$*.0.999.phased.withchr.vcf O=${AFANASIEVO_PREFIX}.$*.0.999.phased.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${AFANASIEVO_PREFIX}.$*.0.999.phased.GRCh38.rejected_variants.vcf R=hg38.fa
	rm ${AFANASIEVO_PREFIX}.$*.0.999.phased.vcf
	rm ${AFANASIEVO_PREFIX}.$*.0.999.phased.withchr.vcf
	bgzip -c ${AFANASIEVO_PREFIX}.$*.0.999.phased.GRCh38.vcf > ${AFANASIEVO_PREFIX}.$*.0.999.phased.GRCh38.vcf.gz 
	rm ${AFANASIEVO_PREFIX}.$*.0.999.phased.GRCh38.vcf
	tabix -p vcf ${AFANASIEVO_PREFIX}.$*.0.999.phased.GRCh38.vcf.gz

afanasievo_%.samples: ${AFANASIEVO_PREFIX}.%.0.999.phased.vcf.gz %_ancestral_states_37.fa
	python convert.py afanasievo -p \
		${AFANASIEVO_PREFIX}.$*.0.999.phased.vcf.gz \
		$*_ancestral_states_37.fa \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

afanasievo_GRCh38.%.samples: ${AFANASIEVO_PREFIX}.%.0.999.phased.GRCh38.vcf.gz %_ancestral_states.fa
	python convert.py afanasievo -p \
		${AFANASIEVO_PREFIX}.$*.0.999.phased.GRCh38.vcf.gz \
		$*_ancestral_states.fa \
		--ancestral-states-url=${ANCESTRAL_STATES_URL}\
		--reference-name=${REFERENCE_NAME}\
		--num-threads 20\
		$@ > $@.report

#afanasievo_new_shapeit_%.samples: ${AFANASIEVO_PREFIX}Family_shapeit_%.covered.phased.filtered.vcf.gz %_ancestral_states_37.fa
afanasievo_phased4wilder.%.samples: afanasievo/phased4Wilder/AfanasievoFamily_%.covered.phased.filtered.vcf.gz %_ancestral_states_37.fa
	python convert.py afanasievo -p \
		afanasievo/phased4Wilder/AfanasievoFamily_$*.covered.phased.filtered.vcf.gz \
		$*_ancestral_states_37.fa \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report


#afanasievo_new_%.samples: afanasievo/phased4Wilder/AfanasievoFamily_%.covered.phased.filtered.vcf.gz %_ancestral_states_37.fa
afanasievo_new_phased_oldtsinfer_times.%.samples: afanasievo/phasing/afanasievo.phased.vcf %_ancestral_states_37.fa
	python convert.py afanasievo -p \
		afanasievo/phasing/afanasievo.phased.vcf.gz \
		$*_ancestral_states_37.fa \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

afanasievo_new_phased.%.samples: afanasievo/phasing/afanasievo.phased.vcf %_ancestral_states_37.fa
	python convert.py afanasievo -p \
		afanasievo/phasing/afanasievo.phased.vcf.gz \
		$*_ancestral_states_37.fa \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

afanasievo_new_beagle_phased.%.samples: afanasievo/phasing/AfanasievoBeaglePhased.vcf.gz %_ancestral_states_37.fa
	python convert.py afanasievo -p \
		afanasievo/phasing/AfanasievoBeaglePhased.vcf.gz \
		$*_ancestral_states_37.fa \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report


#############################################
# 1240k Data
#############################################

#v42.4.1240K_%.vcf: 
#plink --file 1240k/v42.4.1240K_$* --alleleACGT --recode vcf --out 1240k/v42.4.1240K_$*

reich_%.samples: 1240k/v42.4.1240K_%.vcf.gz %_ancestral_states_37.fa 1240k/v42.4.1240K.anno
	python convert.py 1240k -p \
		1240k/v42.4.1240K_chr20.vcf.gz \
		$*_ancestral_states_37.fa \
		-m 1240k/v42.4.1240K.anno \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report


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
# Centromere locations for GRCh37 (aka hg19) from UCSC
# See https://www.biostars.org/p/2349/
#############################################
CENTROMERES_CSV=centromeres.csv
${CENTROMERES_CSV}:
	 curl http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz > cytoband.txt.gz
	 echo "chrom,start,end" > ${CENTROMERES_CSV}
	 # Start and end coordinates are on different lines, so we merge them.
	 zcat cytoband.txt.gz | grep acen | sort | paste -d " " - - \
		 | cut -f 1,2,7 --output-delim="," >> ${CENTROMERES_CSV}

#############################################
# Ancestral states from Ensembl
#############################################

# HGDP is in GRCh38, and tgp has a GRCh38 alignment available. Others we lift over. 
# So we download the ancestral states for GRCh38. 

# Recorded in the sample file provenance.
REFERENCE_NAME=GRCh38

ANCESTRAL_STATES_PREFIX=homo_sapiens_ancestor_GRCh38
ANCESTRAL_STATES_TARBALL=${ANCESTRAL_STATES_PREFIX}.tar.gz
ANCESTRAL_STATES_URL=ftp://ftp.ensembl.org/pub/release-99/fasta/ancestral_alleles/${ANCESTRAL_STATES_TARBALL}

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
	ln -sf homo_sapiens_ancestor_$*.fa ${ANCESTRAL_STATES_PREFIX}/$@

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
# HGDP Data 
#############################################

hgdp_%.samples: hgdp/hgdp_genotypes.%.phased.GRCh38.bcf %_ancestral_states.fa hgdp/hgdp_samples.txt 
	python convert.py hgdp -p \
		hgdp/hgdp_genotypes.$*.phased.GRCh38.bcf \
		${ANCESTRAL_STATES_PREFIX}/$*_ancestral_states.fa \
		-m hgdp/hgdp_samples.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL}\
		--reference-name=${REFERENCE_NAME}\
		$@ > $@.report

hgdp_%.snipped.trees.gnn.csv: hgdp_%.snipped.trees
	python3 tsutil.py compute-hgdp-gnn $^ $@ --num-threads=16
	
#############################################
# Max Planck Data 
#############################################

ARCHAIC_PATH=max_planck_data/

vindija_%.samples: max_planck_data/vindija/vindija.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/vindija/vindija_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/vindija/vindija.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/vindija/vindija_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

altai_%.samples: max_planck_data/altai/altai.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/altai/altai_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/altai/altai.$*_mq25_mapab100.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/altai/altai_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
		$@ > $@.report

chagyrskaya_%.samples: max_planck_data/chagyrskaya/chagyrskaya.%.noRB.vcf.gz %_ancestral_states_37.fa max_planck_data/chagyrskaya/chagyrskaya_metadata.txt
	python convert.py max-planck -p \
		max_planck_data/chagyrskaya/chagyrskaya.$*.noRB.vcf.gz \
		$*_ancestral_states_37.fa \
		-m max_planck_data/chagyrskaya/chagyrskaya_metadata.txt \
		--ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
		--reference-name=${REFERENCE_NAME_37}\
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
# 1240k Data
#############################################

afanasievo_%.samples: afanasievo/Afanasievo.%.0.999.phased.vcf.gz %_ancestral_states_37.fa
	python convert.py afanasievo -p \
		afanasievo/Afanasievo.$*.0.999.phased.vcf.gz \
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


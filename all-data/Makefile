NUM_THREADS ?= 0

# Requirements: bcftools, tabix, python3.
# See requirements.txt for Python package requirements.
#
help:
	@echo Makefile to create tree sequences used in tsdate paper

all: 1kg_chr20.samples sgdp_chr20.samples

simplebgen: setup.py simplebgenmodule.c
    python3 setup.py build_ext --inplace

%.bcf.csi: %.bcf
    bcftools index $(patsubst %.bcf.csi,%.bcf,$@)

%.vcf.gz.csi: %.vcf.gz
    bcftools index $(patsubst %.vcf.gz.csi,%.vcf.gz,$@)

# Save all intermediate files
.SECONDARY:

#############################################
# Standard pipeline for samples file to .dated.trees
#############################################

%.missing_binned.samples: %.samples
    python3 bin_missing.py $^ $@

1kg_chr20.trees: 1kg_chr20.samples
    python3 ../src/run_inference.py $^ -t ${NUM_THREADS} -A 0.1 -S 0.1

%.trees: %.samples
    python3 ../src/run_inference.py $^ -t ${NUM_THREADS} -A 0.1 -S 0.1 -m recomb-hg38/genetic_map_GRCh38_

%.dated.trees: %.trees
    python3 -m tsdate preprocess $^ $*.preprocessed.trees
    python3 -m tsdate date $*.preprocessed.trees $@ 10000 -m 1e-8 -p --ignore-oldest -t ${NUM_THREADS}

%.dated.samples: %.samples %.dated.trees
    python3 tsutil.py dated_samples $^

%.binned.samples: %.dated.samples
    python3 bin_dates.py $^ $@

%.dated.samples: %.samples %.modern.dated.trees %.modern.dates.p
    python3 get_dated_sampledata.py $^

%.trees.gz: %.trees
    gzip -c $^ > $@

%.trees.tsz: %.trees
    tszip -k $^ 

%.trees.bcf: %.trees
    msp vcf -P 2 $^ | bcftools view - -O b -o $@


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

###########################
# GRCh38 Recombination Maps
###########################

recomb-hg38/:
	wget http://csg.sph.umich.edu/locuszoom/download/recomb-hg38.tar.gz
	tar -xvzf recomb-hg38.tar.gz
	./modify_genetic_map.sh

genetic_map_GRCh38_%.txt: recomb-hg38/

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

1kg_%_genotypes.bcf:
    curl ${GENOTYPES_BCF_BASE}/ALL.$*.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.bcf -o $@

1kg_%.samples: 1kg_%_genotypes.vcf.gz %_ancestral_states_37.fa 1kg_samples.ped
    python3 convert.py 1kg -p \
        1kg_$*_genotypes.vcf.gz \
        $*_ancestral_states_37.fa \
        -m 1kg_samples.ped \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
        --reference-name=${REFERENCE_NAME_37} \
        --num-threads=${NUM_THREADS} \
        $@ > $@.report


#############################################
# 1000 Genomes GRCh38 data.
#############################################

GENOTYPES_BASE_GRCH38=ftp://ftp.sra.ebi.ac.uk/vol1/ERZ822/ERZ822766/
GRCH38_RECOMB_MAP=recomb-hg38/genetic_map_GRCh38_

1kg_GRCh38_%_genotypes.vcf.gz:
    curl ${GENOTYPES_BASE_GRCH38}/ALL.$*.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz -o $@
    tabix -p vcf $@ 

1kg_GRCh38_%.samples: 1kg_GRCh38_%_genotypes.vcf.gz %_ancestral_states.fa 1kg_samples.ped
    python3 convert.py 1kg -p \
        1kg_GRCh38_$*_genotypes.vcf.gz \
        $*_ancestral_states.fa \
        -m 1kg_samples.ped \
        --ancestral-states-url=${ANCESTRAL_STATES_URL}\
        --reference-name=${REFERENCE_NAME}\
        --num-threads ${NUM_THREADS}\
        $@  > $@.report

1kg_GRCh38_allsites_%.trees: 1kg_GRCh38_allsites_%.samples recomb-hg38/genetic_map_GRCh38_chr20.txt
    python3 ../src/run_inference.py $< -t ${NUM_THREADS} -m ${GRCH38_RECOMB_MAP}$*.txt

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

sgdp_GRCh38_%_genotypes.vcf.gz: sgdp_%_genotypes.vcf.gz
    gunzip -c sgdp_$*_genotypes.vcf.gz > sgdp_$*.vcf
    awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' sgdp_$*.vcf > sgdp_$*.withchr.vcf
    java -jar ../tools/picard.jar LiftoverVcf I=sgdp_$*.withchr.vcf O=sgdp_GRCh38_$*.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=sgdp_GRCh38_$*.rejected_variants.vcf R=hg38.fa
    rm sgdp_$*.withchr.vcf
    bgzip -c sgdp_GRCh38_$*.vcf > sgdp_GRCh38_$*_genotypes.vcf.gz 
    rm sgdp_GRCh38_$*.vcf
    tabix -p vcf sgdp_GRCh38_$*_genotypes.vcf.gz

sgdp_GRCh38_%_genotypes.bcf: sgdp_GRCh38_%_genotypes.vcf.gz
    # Remove the S_Naxi-2 individual because (a) it doesn't have any metadata in the 
    # file we're using and (b) it has a massively elevated sample edge count if we 
    # leave it in.
    bcftools view -s '^S_Naxi-2' $^ -O b -o $@

sgdp_%.samples: sgdp_%_genotypes.bcf.csi %_ancestral_states_37.fa sgdp_samples.txt
    python3 convert.py sgdp -p \
        sgdp_$*_genotypes.bcf \
        $*_ancestral_states_37.fa \
        -m sgdp/sgdp_samples.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
        --reference-name=${REFERENCE_NAME_37} \
        --num-threads=1 \
        $@  > $@.report

sgdp_GRCh38_%.samples: sgdp_GRCh38_%_genotypes.bcf.csi %_ancestral_states_37.fa sgdp_samples.txt
    python3 convert.py sgdp -p \
        sgdp_GRCh38_$*_genotypes.bcf \
        $*_ancestral_states.fa \
        -m sgdp/sgdp_samples.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL} \
        --reference-name=${REFERENCE_NAME} \
        --num-threads=1 \
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

hgdp_%.samples: hgdp/hgdp_genotypes.%.phased.GRCh38.vcf.gz %_ancestral_states.fa hgdp/hgdp_samples.txt 
    python convert.py hgdp -p \
        hgdp/hgdp_genotypes.$*.phased.GRCh38.vcf.gz \
        $*_ancestral_states.fa \
        -m hgdp/hgdp_samples.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL} \
        --reference-name=${REFERENCE_NAME} \
        --num-threads=${NUM_THREADS} \
        $@ > $@.report

hgdp_missing_data_%_binned.samples: hgdp_missing_data_%_

hgdp_%.snipped.trees.gnn.csv: hgdp_%.snipped.trees
    python3 tsutil.py compute-hgdp-gnn $^ $@ --num-threads=16
    
#############################################
# Max Planck Data 
#############################################

ARCHAIC_PATH=max_planck_data

altai.%_mq25_mapab100.GRCh38.vcf.gz: max_planck_data/altai/altai.%_mq25_mapab100.vcf.gz
    gunzip -c ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.vcf.gz > ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.vcf    
    awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.vcf > ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.withchr.vcf
    rm ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.vcf
    java -jar ../tools/picard.jar LiftoverVcf I=${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.withchr.vcf O=${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.rejected_variants.vcf R=hg38.fa
    vcftools --vcf ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.vcf --chr $* --recode --out ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38
    rm ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.withchr.vcf
    bgzip -c ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.recode.vcf > altai.$*_mq25_mapab100.GRCh38.vcf.gz 
    rm ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.vcf
    rm ${ARCHAIC_PATH}/altai/altai.$*_mq25_mapab100.GRCh38.recode.vcf
    tabix -p vcf altai.$*_mq25_mapab100.GRCh38.vcf.gz

altai_GRCh38_%.samples: altai.%_mq25_mapab100.GRCh38.vcf.gz %_ancestral_states.fa max_planck_data/altai/altai_metadata.txt
    python convert.py max-planck -p \
        altai.$*_mq25_mapab100.GRCh38.vcf.gz \
        $*_ancestral_states.fa \
        -m max_planck_data/altai/altai_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL}\
        --reference-name=${REFERENCE_NAME}\
        --num-threads ${NUM_THREADS}\
        $@ > $@.report

altai_%.samples: max_planck_data/altai/altai.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/altai/altai_metadata.txt
    python convert.py max-planck -p \
        max_planck_data/altai/altai.$*_mq25_mapab100.vcf.gz \
        $*_ancestral_states_37.fa \
        -m max_planck_data/altai/altai_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
        --reference-name=${REFERENCE_NAME_37}\
        --num-threads ${NUM_THREADS} \
        $@ > $@.report

chagyrskaya.%.noRB.GRCh38.vcf.gz: max_planck_data/chagyrskaya/chagyrskaya.%.noRB.vcf.gz
    gunzip -c ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.vcf.gz > ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.vcf  
    awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.vcf > ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.withchr.vcf
    rm ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.vcf
    java -jar ../tools/picard.jar LiftoverVcf I=${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.withchr.vcf O=${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.rejected_variants.vcf R=hg38.fa
    vcftools --vcf ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.vcf --chr $* --recode --out ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38
    rm ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.withchr.vcf
    bgzip -c ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.recode.vcf > chagyrskaya.$*.noRB.GRCh38.vcf.gz 
    rm ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.vcf
    rm ${ARCHAIC_PATH}/chagyrskaya/chagyrskaya.$*.noRB.GRCh38.recode.vcf
    tabix -p vcf chagyrskaya.$*.noRB.GRCh38.vcf.gz

chagyrskaya_%.samples: max_planck_data/chagyrskaya/chagyrskaya.%.noRB.vcf.gz %_ancestral_states_37.fa max_planck_data/chagyrskaya/chagyrskaya_metadata.txt
    python convert.py max-planck -p \
        max_planck_data/chagyrskaya/chagyrskaya.$*.noRB.vcf.gz \
        $*_ancestral_states_37.fa \
        -m max_planck_data/chagyrskaya/chagyrskaya_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
        --reference-name=${REFERENCE_NAME_37}\
        --num-threads ${NUM_THREADS} \
        $@ > $@.report

chagyrskaya_GRCh38_%.samples: chagyrskaya.%.noRB.GRCh38.vcf.gz %_ancestral_states.fa max_planck_data/chagyrskaya/chagyrskaya_metadata.txt
    python convert.py max-planck -p \
        chagyrskaya.$*.noRB.GRCh38.vcf.gz \
        $*_ancestral_states.fa \
        -m max_planck_data/chagyrskaya/chagyrskaya_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL}\
        --reference-name=${REFERENCE_NAME}\
        --num-threads ${NUM_THREADS}\
        $@ > $@.report

denisovan.%_mq25_mapab100.GRCh38.vcf.gz: max_planck_data/denisovan/denisovan.%_mq25_mapab100.vcf.gz
    gunzip -c ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.vcf.gz > ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.vcf
    awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.vcf > ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.withchr.vcf
    rm ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.vcf
    java -jar ../tools/picard.jar LiftoverVcf I=${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.withchr.vcf O=${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.rejected_variants.vcf R=hg38.fa
    rm ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.withchr.vcf
    vcftools --vcf ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.vcf --chr $* --recode --out ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38
    bgzip -c ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.recode.vcf > denisovan.$*_mq25_mapab100.GRCh38.vcf.gz 
    rm ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.vcf
    rm ${ARCHAIC_PATH}/denisovan/denisovan.$*_mq25_mapab100.GRCh38.recode.vcf
    tabix -p vcf denisovan.$*_mq25_mapab100.GRCh38.vcf.gz

denisovan_GRCh38_%.samples: denisovan.%_mq25_mapab100.GRCh38.vcf.gz %_ancestral_states.fa max_planck_data/denisovan/denisovan_metadata.txt
    python convert.py max-planck -p \
        denisovan.$*_mq25_mapab100.GRCh38.vcf.gz \
        $*_ancestral_states.fa \
        -m ${ARCHAIC_PATH}/denisovan/denisovan_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL}\
        --reference-name=${REFERENCE_NAME}\
        --num-threads=${NUM_THREADS}\
        $@ > $@.report

denisovan_%.samples: max_planck_data/denisovan/denisovan.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/denisovan/denisovan_metadata.txt
    python convert.py max-planck -p \
        max_planck_data/denisovan/denisovan.$*_mq25_mapab100.vcf.gz \
        $*_ancestral_states_37.fa \
        -m max_planck_data/denisovan/denisovan_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
        --reference-name=${REFERENCE_NAME_37}\
        --num-threads ${NUM_THREADS} \
        $@ > $@.report

vindija.%_mq25_mapab100.GRCh38.vcf.gz: max_planck_data/vindija/vindija.%_mq25_mapab100.vcf.gz
    gunzip -c ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.vcf.gz > ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.vcf    
    awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.vcf > ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.withchr.vcf
    rm ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.vcf 
    java -jar ../tools/picard.jar LiftoverVcf I=${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.withchr.vcf O=${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.rejected_variants.vcf R=hg38.fa
    vcftools --vcf ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.vcf --chr $* --recode --out ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38
    rm ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.withchr.vcf
    bgzip -c ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.recode.vcf > vindija.$*_mq25_mapab100.GRCh38.vcf.gz 
    rm ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.vcf
    rm ${ARCHAIC_PATH}/vindija/vindija.$*_mq25_mapab100.GRCh38.recode.vcf
    tabix -p vcf vindija.$*_mq25_mapab100.GRCh38.vcf.gz

vindija_%.samples: max_planck_data/vindija/vindija.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/vindija/vindija_metadata.txt
    python convert.py max-planck -p \
        max_planck_data/vindija/vindija.$*_mq25_mapab100.vcf.gz \
        $*_ancestral_states_37.fa \
        -m max_planck_data/vindija/vindija_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37} \
        --reference-name=${REFERENCE_NAME_37} \
        --num-threads ${NUM_THREADS} \
        $@ > $@.report

vindija_GRCh38_%.samples: vindija.%_mq25_mapab100.GRCh38.vcf.gz %_ancestral_states.fa max_planck_data/vindija/vindija_metadata.txt
    python convert.py max-planck -p \
        vindija.$*_mq25_mapab100.GRCh38.vcf.gz \
        $*_ancestral_states.fa \
        -m ${ARCHAIC_PATH}/vindija/vindija_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL}\
        --reference-name=${REFERENCE_NAME}\
        --num-threads ${NUM_THREADS}\
        $@ > $@.report

ust_ishim_%.samples: max_planck_data/ust_ishim/ust_ishim.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/ust_ishim/ust_ishim_metadata.txt
    python convert.py max-planck -p \
        max_planck_data/ust_ishim/ust_ishim.$*_mq25_mapab100.vcf.gz \
        $*_ancestral_states_37.fa \
        -m max_planck_data/ust_ishim/ust_ishim_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
        --reference-name=${REFERENCE_NAME_37}\
        --num-threads ${NUM_THREADS} \
        $@ > $@.report

loshbour_%.samples: max_planck_data/loshbour/loshbour.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/loshbour/loshbour_metadata.txt
    python convert.py max-planck -p \
        max_planck_data/loshbour/loshbour.$*_mq25_mapab100.vcf.gz \
        $*_ancestral_states_37.fa \
        -m max_planck_data/loshbour/loshbour_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
        --reference-name=${REFERENCE_NAME_37}\
        --num-threads ${NUM_THREADS} \
        $@ > $@.report

lbk_%.samples: max_planck_data/lbk/lbk.%_mq25_mapab100.vcf.gz %_ancestral_states_37.fa max_planck_data/lbk/lbk_metadata.txt
    python convert.py max-planck -p \
        max_planck_data/lbk/lbk.$*_mq25_mapab100.vcf.gz \
        $*_ancestral_states_37.fa \
        -m max_planck_data/lbk/lbk_metadata.txt \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
        --reference-name=${REFERENCE_NAME_37}\
        --num-threads ${NUM_THREADS} \
        $@ > $@.report

#############################################
# Afanasievo Data
#############################################
AFANASIEVO_PREFIX=afanasievo
VCF_SUFFIX=.phased.detailed.filtered

AfanasievoFamily_%.phased.detailed.filtered.GRCh38.vcf.gz: ${AFANASIEVO_PREFIX}/AfanasievoFamily_%${VCF_SUFFIX}.vcf.gz
    gunzip -c $^ > ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.vcf
    awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.vcf > ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.withchr.vcf
    java -jar ../tools/picard.jar LiftoverVcf I=${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.withchr.vcf O=${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.phased.GRCh38.rejected_variants.vcf R=hg38.fa
    rm ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.vcf
    rm ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.withchr.vcf
    bgzip -c ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf > ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf.gz
    tabix -p vcf ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf.gz
    bcftools view ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf.gz --regions $* -O z > AfanasievoFamily_$*${VCF_SUFFIX}.GRCh38.vcf.gz
    rm ${AFANASIEVO_PREFIX}/AfanasievoFamily_$*${VCF_SUFFIX}.liftedover.GRCh38.vcf*
    tabix -p vcf AfanasievoFamily_$*${VCF_SUFFIX}.GRCh38.vcf.gz

afanasievo_GRCh38_%.samples: AfanasievoFamily_%.phased.detailed.filtered.GRCh38.vcf.gz %_ancestral_states.fa
    python convert.py afanasievo -p \
        AfanasievoFamily_$*.phased.detailed.filtered.GRCh38.vcf.gz \
        $*_ancestral_states.fa \
        --ancestral-states-url=${ANCESTRAL_STATES_URL}\
        --reference-name=${REFERENCE_NAME}\
        --num-threads ${NUM_THREADS}\
        $@ > $@.report


#############################################
# 1240k Data
#############################################

REICH_DIRECTORY=1240k/
REICH_PREFIX=v42.4.1240K
REICH_TARBALL=${ANCESTRAL_STATES_PREFIX_37}.tar
REICH_URL=https://reichdata.hms.harvard.edu/pub/datasets/amh_repo/curated_releases/V42/V42.4/SHARE/public.dir/${REICH_TARBALL}

${REICH_DIRECTORY}${REICH_TARBALL}:
    curl ${REICH_URL} -o $@

${REICH_DIRECTORY}${REICH_PREFIX}.geno: ${REICH_DIRECTORY}${REICH_TARBALL}
    tar -xvzf ${ANCESTRAL_STATES_TARBALL}

${REICH_DIRECTORY}${REICH_PREFIX}.vcf.gz: ${REICH_DIRECTORY}${REICH_PREFIX}.geno
    ../tools/EIG/src/convertf -p ${REICH_DIRECTORY}par.PACKEDANCESTRYMAP.PACKEDPED
    mv ${REICH_DIRECTORY}${REICH_PREFIX}.pedsnp ${REICH_DIRECTORY}${REICH_PREFIX}.bim
    mv ${REICH_DIRECTORY}${REICH_PREFIX}.pedind ${REICH_DIRECTORY}${REICH_PREFIX}.fam
    plink --bfile ${REICH_DIRECTORY}${REICH_PREFIX} --recode vcf --out ${REICH_DIRECTORY}${REICH_PREFIX}
    bgzip -c ${REICH_DIRECTORY}${REICH_PREFIX}.vcf > $@
    rm ${REICH_DIRECTORY}${REICH_PREFIX}.vcf
    tabix -p vcf $@

${REICH_DIRECTORY}${REICH_PREFIX}_chr%.vcf.gz: ${REICH_DIRECTORY}${REICH_PREFIX}.vcf.gz
    bcftools view ${REICH_DIRECTORY}${REICH_PREFIX}.vcf.gz --regions $* -O z -o $@


reich_%.samples: 1240k/v42.4.1240K_%.vcf.gz %_ancestral_states_37.fa 1240k/v42.4.1240K.anno
    python convert.py 1240k -p \
        1240k/v42.4.1240K_%.vcf.gz \
        $*_ancestral_states_37.fa \
        -m 1240k/v42.4.1240K.anno \
        --ancestral-states-url=${ANCESTRAL_STATES_URL_37}\
        --reference-name=${REFERENCE_NAME_37}\
        --num-threads=${NUM_THREADS}\
        $@ > $@.report

reich_ancients_%.samples: reich_%.samples
    python tsutil.py remove-moderns-reich $^ $@

1240k/v42.4.1240K_GRCh38_%.vcf.gz: 1240k/v42.4.1240K_%.vcf.gz 
    gunzip -c 1240k/v42.4.1240K_$*.vcf.gz > 1240k/v42.4.1240K_$*.vcf
    awk '{if($$0 !~ /^#/) print "chr"$$0; else print $$0}' 1240k/v42.4.1240K_$*.vcf > 1240k/v42.4.1240K_$*.withchr.vcf
    java -jar ../tools/picard.jar LiftoverVcf \
        INPUT=1240k/v42.4.1240K_$*.withchr.vcf \
        OUTPUT=1240k/v42.4.1240K_GRCh38_$*.vcf \
        CHAIN=hg19ToHg38.over.chain.gz \
        REJECT=1240k/v42.4.1240K_GRCh38_$*.rejected_variants.vcf \
        R=hg38.fa \
        RECOVER_SWAPPED_REF_ALT=true
    bgzip -c 1240k/v42.4.1240K_GRCh38_$*.vcf > 1240k/v42.4.1240K_GRCh38_$*.vcf.gz 
    rm 1240k/v42.4.1240K_GRCh38_$*.vcf
    tabix -p vcf 1240k/v42.4.1240K_GRCh38_$*.vcf.gz

reich_GRCh38_%.samples: 1240k/v42.4.1240K_GRCh38_%.vcf.gz %_ancestral_states.fa 1240k/v42.4.1240K.anno
    python convert.py 1240k -p \
        1240k/v42.4.1240K_GRCh38_$*.vcf.gz \
        $*_ancestral_states.fa \
        -m 1240k/v42.4.1240K.anno \
        --ancestral-states-url=${ANCESTRAL_STATES_URL}\
        --reference-name=${REFERENCE_NAME}\
        --num-threads=${NUM_THREADS}\
        $@ > $@.report

reich_ancients_GRCh38_%.samples: reich_GRCh38_%.samples
    python tsutil.py remove-moderns-reich $^ $@


#############################################
# Merged Sampledata Files 
#############################################

hgdp_1kg_sgdp_%.samples: hgdp_%.samples 1kg_GRCh38_%.samples sgdp_GRCh38_%.samples
    python tsutil.py merge-sampledata-files --input-sampledata $^ --output $@

hgdp_1kg_sgdp_high_cov_ancients_%.samples: hgdp_1kg_sgdp_%.samples afanasievo_GRCh38_%.samples denisovan_GRCh38_%.samples vindija_GRCh38_%.samples chagyrskaya_GRCh38_%.samples altai_GRCh38_%.samples
    python tsutil.py make-sampledata-compatible --input-sampledata $^
    python tsutil.py merge-sampledata-files --input-sampledata $< afanasievo_GRCh38_$*.subset.samples denisovan_GRCh38_$*.subset.samples vindija_GRCh38_$*.subset.samples chagyrskaya_GRCh38_$*.subset.samples altai_GRCh38_$*.subset.samples --output $@

hgdp_1kg_sgdp_all_ancients_%.samples: hgdp_1kg_sgdp_high_cov_ancients_%.samples reich_ancients_GRCh38_%.samples
    python tsutil.py make-sampledata-compatible --input-sampledata $^
    python tsutil.py merge-sampledata-files --input-sampledata $< reich_ancients_GRCh38_$*.subset.samples --output $@

hgdp_1kg_sgdp_high_cov_ancients_dated_%.samples: hgdp_1kg_sgdp_high_cov_ancients_%.samples hgdp_1kg_sgdp_all_ancients_%.samples hgdp_1kg_sgdp_%.missing_binned.dated.trees
    python tsutil.py combined-ts-dated-samples --high-cov hgdp_1kg_sgdp_high_cov_ancients_%.samples --all-samples hgdp_1kg_sgdp_all_ancients_%.samples --dated-ts 1kg_sgdp_hgdp_%.binned.dated.trees --output hgdp_1kg_sgdp_high_cov_ancients_%.dated.samples


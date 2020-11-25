all: | relate geva htslib

geva: 
	git clone https://github.com/pkalbers/geva.git 
	cd geva && make

relate:
	wget https://myersgroup.github.io/relate/download/relate_v1.0.17_x86_64_dynamic.tgz
	tar -xzf relate_v1.0.17_x86_64_dynamic.tgz
	mv relate_v1.0.17_x86_64_dynamic relate

htslib:
	wget https://github.com/samtools/htslib/releases/download/1.11/htslib-1.11.tar.bz2
	tar -jxf htslib-1.11.tar.bz2
	mv htslib-1.11 htslib
	cd htslib && ./configure prefix=$(pwd) && make && make install

bcftools:
	wget https://github.com/samtools/bcftools/releases/download/1.11/bcftools-1.11.tar.bz2
	tar -jxf bcftools-1.11.tar.bz2
	mv bcftools-1.11 bcftools
	cd bcftools && ./configure --enable-libcurl=no && make -j 8


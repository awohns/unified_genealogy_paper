#FILE_SUFFIX=T_hg19_1000g.
FILE_SUFFIX=_mq25_mapab100.vcf.gz
denisovan.chr%:
	curl "http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Denisova/chr"${*}${FILE_SUFFIX} -o "denisovan/denisovan.chr"${*}${FILE_SUFFIX} 
	zgrep -v "^##" denisovan/denisovan.chr${*}${FILE_SUFFIX} | awk '$$5 != "." && substr($$10, 1, 3) != "0/0" {print $$1,$$2,$$4,$$5}' > denisovan/denisovan.chr${*}.GRCh37.snps.txt

vindija.chr%:
	curl "http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Vindija33.19/chr"${*}${FILE_SUFFIX} -o "vindija/vindija.chr"${*}${FILE_SUFFIX}
	zgrep -v "^##" vindija/vindija.chr${*}${FILE_SUFFIX} | awk '$$5 != "." && substr($$10, 1, 3) != "0/0" {print $$1,$$2,$$4,$$5}' > vindija/vindija.chr${*}.GRCh37.snps.txt

altai.chr%:
	curl "http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Altai/chr"${*}${FILE_SUFFIX} -o "altai/altai.chr"${*}${FILE_SUFFIX}
	zgrep -v "^##" altai/altai.chr${*}${FILE_SUFFIX} | awk '$$5 != "." && substr($$10, 1, 3) != "0/0" {print $$1,$$2,$$4,$$5}' > altai/altai.chr${*}.GRCh37.snps.txt

ust_ishim.chr%:
	curl "http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Ust_Ishim/chr"${*}${FILE_SUFFIX} -o "ust_ishim/ust_ishim.chr"${*}${FILE_SUFFIX}
	zgrep -v "^##" ust_ishim/ust_ishim.chr${*}${FILE_SUFFIX} | awk '$$5 != "." && substr($$10, 1, 3) != "0/0" {print $$1,$$2,$$4,$$5}' > ust_ishim/ust_ishim.chr${*}.GRCh37.snps.txt 

CHAGYRSKAYA_SUFFIX=.noRB.vcf.gz
chagyrskaya.chr%:
	curl "http://ftp.eva.mpg.de/neandertal/Chagyrskaya/VCF/chr"${*}${CHAGYRSKAYA_SUFFIX} -o "chagyrskaya/chagyrskaya.chr"${*}${CHAGYRSKAYA_SUFFIX}
	zgrep -v "^##" chagyrskaya/chagyrskaya.chr${*}${CHAGYRSKAYA_SUFFIX} | awk '$$5 != "." && substr($$10, 1, 3) != "0/0" {print $$1,$$2,$$4,$$5}' > chagyrskaya/chagyrskaya.chr${*}.GRCh37.snps.txt 

lbk.chr%:
	curl "http://cdna.eva.mpg.de/neandertal/Vindija/VCF/LBK/chr"${*}${FILE_SUFFIX} -o lbk/"lbk.chr"${*}${FILE_SUFFIX}
	zgrep -v "^##" lbk/lbk.chr${*}${FILE_SUFFIX} | awk '$$5 != "." && substr($$10, 1, 3) != "0/0" {print $$1,$$2,$$4,$$5}' > lbk/lbk.chr${*}.GRCh37.snps.txt

loshbour.chr%:
	curl "http://cdna.eva.mpg.de/neandertal/Vindija/VCF/Loschbour/chr"${*}${FILE_SUFFIX} -o "loshbour/loshbour.chr"${*}${FILE_SUFFIX}
	zgrep -v "^##" loshbour/loshbour.chr${*}${FILE_SUFFIX} | awk '$$5 != "." && substr($$10, 1, 3) != "0/0" {print $$1,$$2,$$4,$$5}' > loshbour/loshbour.chr${*}.GRCh37.snps.txt

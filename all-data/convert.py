"""
Convert input data from various sources to samples format.
"""
import argparse
import subprocess
import os
import sys
import math
import re

import allel
import numpy as np
import tsinfer
import attr
import cyvcf2
import pysam
import tqdm
import pandas as pd
import multiprocessing

import tskit

GENERATION_TIME = 25

try:
    import bgen_reader

    # Local module used to work around slow genotype access in bgen_reader
    import simplebgen
except ImportError:
    # bgen-reader isn't available for Python 3.4.
    print("WARNING: Cannot import bgen reader")


@attr.s()
class Site(object):
    position = attr.ib(None)
    alleles = attr.ib(None)
    genotypes = attr.ib(None)
    metadata = attr.ib({})
    inference = attr.ib(None)


def run_multiprocessing(args, function):
    """
    Run multiprocessing of sampledata files.
    We use multiple threads by splitting the VCF file into chunks and using the vcf_subset
    function of cyvcf2.
    """
    vcf_fn = args.data_file
    num_processes = args.num_threads
    if num_processes > 1:
        # Split the VCF into chunks
        callset = allel.read_vcf(vcf_fn, fields=["variants/CHROM", "variants/POS"])
        pos_list = callset["variants/POS"]
        chroms = callset["variants/CHROM"]
        assert np.all(chroms == chroms[0])
        chrom = str(chroms[0])

        def get_chromosome_chunks(lst, num_processes):
            length = len(lst)
            n = math.ceil(length / num_processes)
            chunks = list()
            for index, i in enumerate(range(0, length, n)):
                if index != num_processes - 1:
                    chunks.append(
                        (
                            args,
                            args.output_file + str(index),
                            (chrom + ":" + str(lst[i]) + "-" + str(lst[i + n])),
                        )
                    )
                else:
                    chunks.append(
                        (
                            args,
                            args.output_file + str(index),
                            (chrom + ":" + str(lst[i]) + "-" + str(lst[-1])),
                        )
                    )
            return chunks

        chunks = get_chromosome_chunks(pos_list, num_processes)
        chunks_iter = iter(chunks)
        reports = list()
        completed_files = list()
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=10) as pool:
            for index, row in enumerate(pool.map(function, chunks_iter)):
                reports.append(row)
                print("Processed Chunk {}: {} with {} sites added.".format(index, chunks[index][2], row["num_sites"]))
                if row["num_sites"] > 0:
                    completed_files.append(index)

        # Combine reports and print
        master_report = reports[0]
        for report in reports[1:]:
            for var_type, val in report.items():
                master_report[var_type] += val
        print(master_report)

        # Combine sampledata files
        filenames = completed_files
        all_samples = []
        for name in filenames:
            all_samples.append(tsinfer.load(args.output_file + str(name))) 
            os.remove(args.output_file + str(name))

        samples = all_samples[0].copy(args.output_file)
        samples.append_sites(*all_samples[1:])
        samples.finalise()
#        filenames = completed_files
#        samples = tsinfer.load(args.output_file + "0")
#        copy = samples.copy(args.output_file)
#        for sampledata_file in filenames[1:]:
#            cur_sampledata = tsinfer.load(args.output_file + str(sampledata_file))
#            copy.data["sites/position"].append(cur_sampledata.sites_position[:])
#            copy.data["sites/alleles"].append(cur_sampledata.sites_alleles[:])
#            copy.data["sites/genotypes"].append(cur_sampledata.sites_genotypes[:])
#            copy.data["sites/inference"].append(cur_sampledata.sites_inference[:])
#            copy.data["sites/metadata"].append(cur_sampledata.sites_metadata[:])
#            copy.data["sites/time"].append(cur_sampledata.sites_time[:])
#            os.remove(args.output_file + str(sampledata_file))
#        copy.finalise()
        assert np.all(np.diff(samples.sites_position[:]) > 0) 

    else:
        raise ValueError


def make_sampledata(args):
    if isinstance(args, tuple):
        vcf_subset = args[2]
        args[0].output_file = str(args[1])
        args = args[0]
    else:
        vcf_subset = None
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
    git_provenance = {
        "repo": "git@github.com:mcveanlab/treeseq-inference.git",
        "hash": git_hash.decode().strip(),
        "dir": "human-data",
        "notes:": ("Use the Makefile to download and process the upstream data files"),
    }
    data_provenance = {
        "ancestral_states_url": args.ancestral_states_url,
        "reference_name": args.reference_name,
    }

    # Get the ancestral states.
    fasta = pysam.FastaFile(args.ancestral_states_file)
    # NB! We put in an extra character at the start to convert to 1 based coords.
    ancestral_states = "X" + fasta.fetch(reference=fasta.references[0])
    # The largest possible site position is len(ancestral_states). Positions must
    # be strictly less than sequence_length, so we add 1.
    sequence_length = len(ancestral_states) + 1

    converter_class = {
        "1kg": ThousandGenomesConverter,
        "sgdp": SgdpConverter,
        "ukbb": UkbbConverter,
        "hgdp": HgdpConverter,
        "max-planck": MaxPlanckConverter,
        "afanasievo": AfanasievoConverter,
        "1240k": ReichConverter,
    }
    try:
        with tsinfer.SampleData(
            path=args.output_file, num_flush_threads=1, sequence_length=sequence_length
        ) as samples:
            converter = converter_class[args.source](
                args.data_file, ancestral_states, samples, args.target_samples
            )
            if args.metadata_file:
                converter.process_metadata(args.metadata_file, args.progress)
            else:
                converter.process_metadata(args.progress)
            if vcf_subset is not None:
                report = converter.process_sites(
                    vcf_subset=vcf_subset,
                    show_progress=args.progress,
                    max_sites=args.max_variants,
                )
            else:
                report = converter.process_sites(
                    show_progress=args.progress, max_sites=args.max_variants
                )
            samples.record_provenance(
                command=sys.argv[0],
                args=sys.argv[1:],
                git=git_provenance,
                data=data_provenance,
            )
            assert np.all(np.diff(samples.sites_position[:]) > 0) 
    except Exception as e:
        os.unlink(args.output_file)
        if report["num_sites"] == 0:
            return report
        raise e
    if report["num_sites"] == 0:
        os.unlink(args.output_file)
    return report


def filter_duplicates_target(vcf, target_sites_pos=None):
    """
    Returns the variants from this VCF with duplicate sites filtered
    out. If any site position appears more than once, throw all variants away.
    If target_sites_pos is not None, only returns variants from this VCF which are present in the target sampledata file.
    """
    if target_sites_pos is not None:
        def site_in_target(site):
            return site in target_sites_pos
    else:
        def site_in_target(site):
            return True
    row = next(vcf, None)
    bad_pos = -1
    for next_row in vcf:
        if bad_pos == -1 and next_row.POS != row.POS: 
            if site_in_target(row.POS):
                yield row
        else:
            if bad_pos == -1:
                bad_pos = row.POS
            elif bad_pos != next_row.POS:
                bad_pos = -1
        row = next_row
    if row is not None and bad_pos != -1 and site_in_target(row.POS):
        yield row


class Converter(object):
    """
    Superclass of converters.
    """

    def __init__(self, data_file, ancestral_states, samples, target_samples=None):
        self.data_file = data_file
        self.ancestral_states = ancestral_states
        self.samples = samples
        if target_samples is not None:
            self.target_sites_pos = set(tsinfer.load(target_samples).sites_position[:])
        else:
            self.target_sites_pos = None
        self.num_samples = -1
        self.num_sites = 0
        # ancestral states counters.
        self.num_no_ancestral_state = 0
        self.num_low_confidence_ancestral_state = 0
        # Counters for genotypes and sites.
        self.num_unphased = 0
        self.num_missing_data = 0
        self.num_invariant = 0
        self.num_indels = 0
        self.num_non_biallelic = 0
        self.num_singletons = 0
        # (n - 1)-tons
        self.num_nmo_tons = 0

    def report(self):
        report_dict = {}
        report_dict["num_sites"] = self.num_sites
        report_dict["unphased"] = self.num_unphased
        report_dict["missing_data"] = self.num_missing_data
        report_dict["invariant"] = self.num_invariant
        report_dict["num_indels"] = self.num_indels
        report_dict["non_biallelic"] = self.num_non_biallelic
        report_dict["no_ancestral_state"] = self.num_no_ancestral_state
        report_dict[
            "low_confidence_ancestral_state"
        ] = self.num_low_confidence_ancestral_state
        report_dict["num_singletons"] = self.num_singletons
        report_dict["num_(n - 1)_tons"] = self.num_nmo_tons
        return report_dict

    def process_metadata(self, metadata_file):
        pass

    def get_ancestral_state(self, position):
        # From the ancestral states README:
        # The convention for the sequence is:
        #    ACTG : high-confidence call, ancestral state supproted by the other two sequences
        #    actg : low-confindence call, ancestral state supported by one sequence only
        #    N    : failure, the ancestral state is not supported by any other sequence
        #    -    : the extant species contains an insertion at this postion
        #    .    : no coverage in the alignment

        ret = None
        # NB: we assume that this array is modified so that the 1-indexed coordinates
        # work correctly!
        ancestral_state = self.ancestral_states[position]
        if ancestral_state in [".", "N", "-"]:
            self.num_no_ancestral_state += 1
        elif ancestral_state in ["a", "c", "t", "g"]:
            self.num_low_confidence_ancestral_state += 1
        else:
            assert ancestral_state in ["A", "C", "T", "G"]
            ret = ancestral_state
        return ret


class VcfConverter(Converter):
    def convert_genotypes(self, row, ancestral_state):
        def return_genotype(allele, ancestral_state):
            if allele == ".":
                return tskit.MISSING_DATA
            else:
                return allele != ancestral_state

        ret = None
        num_diploids = self.num_samples // 2
        a = np.zeros(self.num_samples, dtype=np.int8)
        # Use upper case version of ancestral state (keep original
        # for checking low-confidence ancestral state)
        all_alleles = set([ancestral_state])
        # Fill in a with genotypes.
        bases = np.array(row.gt_bases)
        for j in range(num_diploids):
            missing = False
            if "|" in bases[j]:
                alleles = bases[j].split("|")
            else:
                self.num_unphased += 1
                alleles = bases[j].split("/")
            if len(alleles) != 2:
                break
            for allele in alleles:
                if allele == ".":
                    self.num_missing_data += 1
                    missing = True
                else:
                    all_alleles.add(allele)
            a[2 * j] = alleles[0] != ancestral_state 
            a[2 * j + 1] = alleles[1] != ancestral_state

            if missing:
                if alleles[0] == ".":
                    a[2 * j] = tskit.MISSING_DATA
                if alleles[1] == ".":
                    a[2 * j + 1] = tskit.MISSING_DATA
        else:
            freq = np.sum(a == 1)
            if len(all_alleles) > 2:
                self.num_non_biallelic += 1
            elif freq == self.num_samples or freq == 0:
                self.num_invariant += 1
            elif any(len(allele) != 1 for allele in all_alleles):
                self.num_indels += 1
            elif freq == self.num_samples - 1:
                self.num_nmo_tons += 1
            else:
                metadata = {"ID": row.ID, "REF": row.REF}
                if freq == 1:
                    self.num_singletons += 1
                all_alleles.remove(ancestral_state)
                alleles = [ancestral_state, all_alleles.pop()]
                ret = Site(
                    position=row.POS, alleles=alleles, genotypes=a, metadata=metadata
                )
        return ret

    def process_sites(self, vcf_subset=None, show_progress=False, max_sites=None):
        num_data_sites = int(
            subprocess.check_output(["bcftools", "index", "--nrecords", self.data_file])
        )

        progress = tqdm.tqdm(total=num_data_sites, disable=not show_progress)
        self.num_sites = 0
        if vcf_subset is None:
            vcf = cyvcf2.VCF(self.data_file)
        else:
            vcf = cyvcf2.VCF(self.data_file)(vcf_subset)
            start_pos = vcf_subset.replace(":", " ").replace("-", " ").split()[1]
        for row in filter_duplicates_target(vcf, self.target_sites_pos):
            ancestral_state = self.get_ancestral_state(row.POS)
            if ancestral_state is not None:
                site = self.convert_genotypes(row, ancestral_state)
                if site is not None:
                    if site.inference is not None:
                        self.samples.add_site(
                            position=site.position,
                            genotypes=site.genotypes,
                            alleles=site.alleles,
                            metadata=site.metadata,
                            inference=site.inference,
                        )
                    else:
                        self.samples.add_site(
                            position=site.position,
                            genotypes=site.genotypes,
                            alleles=site.alleles,
                            metadata=site.metadata,
                        )

                    progress.set_postfix(used=str(self.num_sites))
                    self.num_sites += 1
                    if self.num_sites == max_sites:
                        break
            progress.update()
        progress.close()
        report_dict = self.report()
        return report_dict


class ThousandGenomesConverter(VcfConverter):
    """
    Converts data for the 1000 Genomes.
    """

    def process_metadata(self, metadata_file, show_progress=False):
        """
        Adds the 1000 genomes populations metadata.
        """
        # Based on
        # http://www.internationalgenome.org/faq/which-populations-are-part-your-study/
        populations = [
            ["CHB", "Han Chinese in Beijing, China", "EAS"],
            ["JPT", "Japanese in Tokyo, Japan", "EAS"],
            ["CHS", "Southern Han Chinese", "EAS"],
            ["CDX", "Chinese Dai in Xishuangbanna, China", "EAS"],
            ["KHV", "Kinh in Ho Chi Minh City, Vietnam", "EAS"],
            [
                "CEU",
                "Utah Residents (CEPH) with Northern and Western European Ancestry",
                "EUR",
            ],
            ["TSI", "Toscani in Italia", "EUR"],
            ["FIN", "Finnish in Finland", "EUR"],
            ["GBR", "British in England and Scotland", "EUR"],
            ["IBS", "Iberian Population in Spain", "EUR"],
            ["YRI", "Yoruba in Ibadan, Nigeria", "AFR"],
            ["LWK", "Luhya in Webuye, Kenya", "AFR"],
            ["GWD", "Gambian in Western Divisions in the Gambia", "AFR"],
            ["MSL", "Mende in Sierra Leone", "AFR"],
            ["ESN", "Esan in Nigeria", "AFR"],
            ["ASW", "Americans of African Ancestry in SW USA", "AFR"],
            ["ACB", "African Caribbeans in Barbados", "AFR"],
            ["MXL", "Mexican Ancestry from Los Angeles USA", "AMR"],
            ["PUR", "Puerto Ricans from Puerto Rico", "AMR"],
            ["CLM", "Colombians from Medellin, Colombia", "AMR"],
            ["PEL", "Peruvians from Lima, Peru", "AMR"],
            ["GIH", "Gujarati Indian from Houston, Texas", "SAS"],
            ["PJL", "Punjabi from Lahore, Pakistan", "SAS"],
            ["BEB", "Bengali from Bangladesh", "SAS"],
            ["STU", "Sri Lankan Tamil from the UK", "SAS"],
            ["ITU", "Indian Telugu from the UK", "SAS"],
        ]

        population_id_map = {}
        for pop in populations:
            pop_id = self.samples.add_population(
                dict(zip(["name", "description", "super_population"], pop))
            )
            population_id_map[pop[0]] = pop_id

        with open(metadata_file, "r") as ped_file:
            # Parse the individual metadata out of the ped file.
            columns = next(ped_file).split("\t")
            sane_names = [col.replace(" ", "_").lower().strip() for col in columns]
            metadata = {}
            populations = {}
            for line in ped_file:
                row = dict(zip(sane_names, line.strip().split("\t")))
                name = row["individual_id"]
                population_name = row.pop("population")
                populations[name] = population_id_map[population_name]
                # The value '0' seems to be used to encode missing, so insert None
                # instead to be more useful.
                nulled = {}
                for key, value in row.items():
                    if value == "0":
                        value = None
                    nulled[key] = value
                metadata[name] = nulled

        vcf = cyvcf2.VCF(self.data_file)
        individual_names = list(vcf.samples)
        vcf.close()
        self.num_samples = len(individual_names) * 2
        # Add in the metadata rows in the order of the VCF.
        for index, name in enumerate(individual_names):
            self.samples.add_individual(
                metadata=metadata[name], population=populations[name], ploidy=2
            )


class SgdpConverter(VcfConverter):
    """
    Converts data for the Simons Genome Diversity project data.
    """

    def process_metadata(self, metadata_file, show_progress=False):
        """
        Adds the SGDP populations metadata.
        """
        # All populations in SGDP mapped to their regions.
        region_map = {
            "Abkhasian": "WestEurasia",
            "Adygei": "WestEurasia",
            "Albanian": "WestEurasia",
            "Aleut": "CentralAsiaSiberia",
            "Altaian": "CentralAsiaSiberia",
            "Ami": "EastAsia",
            "Armenian": "WestEurasia",
            "Atayal": "EastAsia",
            "Australian": "Oceania",
            "Balochi": "SouthAsia",
            "BantuHerero": "Africa",
            "BantuKenya": "Africa",
            "BantuTswana": "Africa",
            "Basque": "WestEurasia",
            "BedouinB": "WestEurasia",
            "Bengali": "SouthAsia",
            "Bergamo": "WestEurasia",
            "Biaka": "Africa",
            "Bougainville": "Oceania",
            "Brahmin": "SouthAsia",
            "Brahui": "SouthAsia",
            "Bulgarian": "WestEurasia",
            "Burmese": "EastAsia",
            "Burusho": "SouthAsia",
            "Cambodian": "EastAsia",
            "Chane": "America",
            "Chechen": "WestEurasia",
            "Chipewyan": "America",
            "Chukchi": "CentralAsiaSiberia",
            "Cree": "America",
            "Crete": "WestEurasia",
            "Czech": "WestEurasia",
            "Dai": "EastAsia",
            "Daur": "EastAsia",
            "Dinka": "Africa",
            "Druze": "WestEurasia",
            "Dusun": "Oceania",
            "English": "WestEurasia",
            "Esan": "Africa",
            "Eskimo_Chaplin": "CentralAsiaSiberia",
            "Eskimo_Naukan": "CentralAsiaSiberia",
            "Eskimo_Sireniki": "CentralAsiaSiberia",
            "Estonian": "WestEurasia",
            "Even": "CentralAsiaSiberia",
            "Finnish": "WestEurasia",
            "French": "WestEurasia",
            "Gambian": "Africa",
            "Georgian": "WestEurasia",
            "Greek": "WestEurasia",
            "Han": "EastAsia",
            "Hawaiian": "Oceania",
            "Hazara": "SouthAsia",
            "Hezhen": "EastAsia",
            "Hungarian": "WestEurasia",
            "Icelandic": "WestEurasia",
            "Igbo": "Africa",
            "Igorot": "Oceania",
            "Iranian": "WestEurasia",
            "Iraqi_Jew": "WestEurasia",
            "Irula": "SouthAsia",
            "Itelman": "CentralAsiaSiberia",
            "Japanese": "EastAsia",
            "Jordanian": "WestEurasia",
            "Ju_hoan_North": "Africa",
            "Kalash": "SouthAsia",
            "Kapu": "SouthAsia",
            "Karitiana": "America",
            "Kashmiri_Pandit": "SouthAsia",
            "Kharia": "SouthAsia",
            "Khomani_San": "Africa",
            "Khonda_Dora": "SouthAsia",
            "Kinh": "EastAsia",
            "Kongo": "Africa",
            "Korean": "EastAsia",
            "Kurumba": "SouthAsia",
            "Kusunda": "SouthAsia",
            "Kyrgyz": "CentralAsiaSiberia",
            "Lahu": "EastAsia",
            "Lemande": "Africa",
            "Lezgin": "WestEurasia",
            "Luhya": "Africa",
            "Luo": "Africa",
            "Madiga": "SouthAsia",
            "Makrani": "SouthAsia",
            "Mala": "SouthAsia",
            "Mandenka": "Africa",
            "Mansi": "CentralAsiaSiberia",
            "Maori": "Oceania",
            "Masai": "Africa",
            "Mayan": "America",
            "Mbuti": "Africa",
            "Mende": "Africa",
            "Miao": "EastAsia",
            "Mixe": "America",
            "Mixtec": "America",
            "Mongola": "CentralAsiaSiberia",
            "Mozabite": "Africa",
            "Nahua": "America",
            "Naxi": "EastAsia",
            "North_Ossetian": "WestEurasia",
            "Norwegian": "WestEurasia",
            "Onge": "SouthAsia",
            "Orcadian": "WestEurasia",
            "Oroqen": "EastAsia",
            "Palestinian": "WestEurasia",
            "Papuan": "Oceania",
            "Pathan": "SouthAsia",
            "Piapoco": "America",
            "Pima": "America",
            "Polish": "WestEurasia",
            "Punjabi": "SouthAsia",
            "Quechua": "America",
            "Relli": "SouthAsia",
            "Russian": "WestEurasia",
            "Saami": "WestEurasia",
            "Saharawi": "Africa",
            "Samaritan": "WestEurasia",
            "Sardinian": "WestEurasia",
            "She": "EastAsia",
            "Sherpa": "SouthAsia",
            "Sindhi": "SouthAsia",
            "Somali": "Africa",
            "Spanish": "WestEurasia",
            "Surui": "America",
            "Tajik": "WestEurasia",
            "Thai": "EastAsia",
            "Tibetan": "SouthAsia",
            "Tlingit": "CentralAsiaSiberia",
            "Tubalar": "CentralAsiaSiberia",
            "Tu": "EastAsia",
            "Tujia": "EastAsia",
            "Turkish": "WestEurasia",
            "Tuscan": "WestEurasia",
            "Ulchi": "CentralAsiaSiberia",
            "Uygur": "EastAsia",
            "Xibo": "EastAsia",
            "Yadava": "SouthAsia",
            "Yakut": "CentralAsiaSiberia",
            "Yemenite_Jew": "WestEurasia",
            "Yi": "EastAsia",
            "Yoruba": "Africa",
            "Zapotec": "America",
        }
        population_id_map = {}
        for name in sorted(region_map.keys()):
            pop_id = self.samples.add_population(
                {"name": name, "region": region_map[name]}
            )
            population_id_map[name] = pop_id

        # The file contains some non UTF-8 codepoints for a contributors name.
        with open(metadata_file, "r", encoding="ISO-8859-1") as md_file:
            columns = next(md_file).lstrip("#").split("\t")
            sane_names = [col.lower().strip() for col in columns]
            j = sane_names.index("sample_id(aliases)")
            sane_names[j] = "aliases"
            for j, name in enumerate(sane_names):
                if name.startswith('"sgdp-lite category'):
                    # There's a very long key that doesn't impart any information here.
                    # Remove it.
                    sane_names[j] = "DELETE"
            rows = {}
            populations = {}
            locations = {}
            for line in md_file:
                metadata = dict(zip(sane_names, line.strip().split("\t")))
                del metadata["DELETE"]
                name = metadata["sgdp_id"]
                population_name = metadata.pop("population_id")
                populations[name] = population_id_map[population_name]
                rows[name] = metadata
                location = [
                    float(metadata.pop("latitude")),
                    float(metadata.pop("longitude")),
                ]
                locations[name] = location
                if metadata["town"] == "?":
                    metadata["town"] = None

        vcf = cyvcf2.VCF(self.data_file)
        individual_names = list(vcf.samples)
        vcf.close()
        self.num_samples = 2 * len(individual_names)

        # Add in the metadata rows in the order of the VCF.
        for name in individual_names:
            metadata = rows[name]
            self.samples.add_individual(
                metadata=metadata,
                location=locations[name],
                ploidy=2,
                population=populations[name],
            )


class HgdpConverter(VcfConverter):
    """
    Converts data for the Human Genome Diversity project data.
    """

    def process_metadata(self, metadata_file, show_progress=False):
        """
        Adds the HGDP populations metadata.
        """
        # All populations in HGDP mapped to their regions.
        region_map = {
            "Brahui": "CENTRAL_SOUTH_ASIA",
            "Balochi": "CENTRAL_SOUTH_ASIA",
            "Hazara": "CENTRAL_SOUTH_ASIA",
            "Makrani": "CENTRAL_SOUTH_ASIA",
            "Sindhi": "CENTRAL_SOUTH_ASIA",
            "Pathan": "CENTRAL_SOUTH_ASIA",
            "Kalash": "CENTRAL_SOUTH_ASIA",
            "Burusho": "CENTRAL_SOUTH_ASIA",
            "Mbuti": "AFRICA",
            "Biaka": "AFRICA",
            "Bougainville": "OCEANIA",
            "French": "EUROPE",
            "PapuanSepik": "OCEANIA",
            "PapuanHighlands": "OCEANIA",
            "Druze": "MIDDLE_EAST",
            "Bedouin": "MIDDLE_EAST",
            "Sardinian": "EUROPE",
            "Palestinian": "MIDDLE_EAST",
            "Colombian": "AMERICA",
            "Cambodian": "EAST_ASIA",
            "Japanese": "EAST_ASIA",
            "Han": "EAST_ASIA",
            "Orcadian": "EUROPE",
            "Surui": "AMERICA",
            "Maya": "AMERICA",
            "Russian": "EUROPE",
            "Mandenka": "AFRICA",
            "Yoruba": "AFRICA",
            "Yakut": "EAST_ASIA",
            "San": "AFRICA",
            "BantuSouthAfrica": "AFRICA",
            "Karitiana": "AMERICA",
            "Pima": "AMERICA",
            "Tujia": "EAST_ASIA",
            "BergamoItalian": "EUROPE",
            "Tuscan": "EUROPE",
            "Yi": "EAST_ASIA",
            "Miao": "EAST_ASIA",
            "Oroqen": "EAST_ASIA",
            "Daur": "EAST_ASIA",
            "Mongolian": "EAST_ASIA",
            "Hezhen": "EAST_ASIA",
            "Xibo": "EAST_ASIA",
            "Mozabite": "MIDDLE_EAST",
            "NorthernHan": "EAST_ASIA",
            "Uygur": "CENTRAL_SOUTH_ASIA",
            "Dai": "EAST_ASIA",
            "Lahu": "EAST_ASIA",
            "She": "EAST_ASIA",
            "Naxi": "EAST_ASIA",
            "Tu": "EAST_ASIA",
            "Basque": "EUROPE",
            "Adygei": "EUROPE",
            "BantuKenya": "AFRICA",
        }
        population_id_map = {}
        for name in sorted(region_map.keys()):
            pop_id = self.samples.add_population(
                {"name": name, "region": region_map[name]}
            )
            population_id_map[name] = pop_id

        # The file contains some non UTF-8 codepoints for a contributors name.
        with open(metadata_file, "r", encoding="ISO-8859-1") as md_file:
            columns = next(md_file).lstrip("#").split("\t")
            sane_names = [col.lower().strip() for col in columns]
            rows = {}
            populations = {}
            locations = {}
            for line in md_file:
                metadata = dict(zip(sane_names, line.strip().split("\t")))
                name = metadata["sample"]
                population_name = metadata.pop("population")
                populations[name] = population_id_map[population_name]
                rows[name] = metadata
                location = [
                    float(metadata.pop("latitude")),
                    float(metadata.pop("longitude")),
                ]
                locations[name] = location

        vcf = cyvcf2.VCF(self.data_file)
        individual_names = list(vcf.samples)
        vcf.close()
        self.num_samples = 2 * len(individual_names)

        # Add in the metadata rows in the order of the VCF.
        for name in individual_names:
            metadata = rows[name]
            self.samples.add_individual(
                metadata=metadata,
                location=locations[name],
                ploidy=2,
                population=populations[name],
            )



class MaxPlanckConverter(VcfConverter):
    """
    Converts data for Max Planck Data. 
    """

    def process_metadata(self, metadata_file, show_progress=False):
        """
        Adds the Max Planck metadata.
        """
        with open(metadata_file, "r") as max_planck_metadata:
            # Parse the individual metadata out of the file.
            lines = max_planck_metadata.read().splitlines()
            metadata = {}
            row = lines[1].split(" ")
            name = row[0]
            metadata["name"] = name
            metadata["age"] = int(row[2]) / GENERATION_TIME
            population = row[1]
        vcf = cyvcf2.VCF(self.data_file)
        individual_names = list(vcf.samples)
        vcf.close()
        self.num_samples = len(individual_names) * 2
        pop_id = self.samples.add_population(
            {"name": population, "super_population": "Max Planck"}
        )
        self.samples.add_individual(time=metadata["age"], metadata=metadata, population=pop_id, ploidy=2)

    def convert_genotypes(self, row, ancestral_state):
        def return_genotype(allele, ancestral_state):
            if allele == ".":
                return tskit.MISSING_DATA
            else:
                return allele != ancestral_state

        ret = None
        num_diploids = self.num_samples // 2
        a = np.zeros(self.num_samples, dtype=np.int8)
        # Use upper case version of ancestral state (keep original
        # for checking low-confidence ancestral state)
        all_alleles = set([ancestral_state])
        # Fill in a with genotypes.
        bases = np.array(row.gt_bases)
        for j in range(num_diploids):
            missing = False
            if "|" in bases[j]:
                alleles = bases[j].split("|")
            else:
                self.num_unphased += 1
                alleles = bases[j].split("/")
            if len(alleles) != 2:
                break
            for allele in alleles:
                if allele == ".":
                    self.num_missing_data += 1
                    missing = True
                else:
                    all_alleles.add(allele)
            a[2 * j] = alleles[0] != ancestral_state 
            a[2 * j + 1] = alleles[1] != ancestral_state

            if missing:
                if alleles[0] == ".":
                    a[2 * j] = tskit.MISSING_DATA
                if alleles[1] == ".":
                    a[2 * j + 1] = tskit.MISSING_DATA
        else:
            freq = np.sum(a == 1)
            if len(all_alleles) > 2:
                self.num_non_biallelic += 1
            elif any(len(allele) != 1 for allele in all_alleles):
                self.num_indels += 1
            else:
                metadata = {"ID": row.ID, "REF": row.REF}
                if freq == 0:
                    self.num_invariant += 1
                    all_alleles.remove(ancestral_state)
                    alleles = [ancestral_state]
                else:
                    if freq == 1:
                        self.num_singletons += 1
                    elif freq == self.num_samples - 1:
                        self.num_nmo_tons += 1
                    elif freq == self.num_samples:
                        self.num_invariant += 1

                    all_alleles.remove(ancestral_state)
                    alleles = [ancestral_state, all_alleles.pop()]
                ret = Site(
                    position=row.POS, alleles=alleles, genotypes=a, metadata=metadata
                )
        return ret


class AfanasievoConverter(MaxPlanckConverter):
    """
    Converts data for Afanasievo Family. 
    """

    def process_metadata(self, show_progress=False):
        """
        Adds the Afanasievo metadata.
        """
        pop_id = self.samples.add_population(
            {"name": "Afanasievo", "super_population": "Afanasievo"}
        )
        vcf = cyvcf2.VCF(self.data_file)
        individual_names = list(vcf.samples)
        vcf.close()
        for name in individual_names:
            metadata = {}
            metadata["name"] = name
            if "Son" in name:
                metadata["age"] = 4100 / GENERATION_TIME
            else:
                metadata["age"] = 4125 / GENERATION_TIME
            self.samples.add_individual(metadata=metadata, time=metadata["age"], population=pop_id, ploidy=2)
        self.num_samples = len(individual_names) * 2


class ReichConverter(VcfConverter):
    """
    Convert data from 1240K array to SampleData file.
    """

    def process_metadata(self, metadata_file, show_progress=False):
        metadata_df = pd.read_csv(metadata_file, delimiter="\t")
        # Add populations
        population_id_map = {}
        seen_populations = list()
        for index, row in metadata_df.iterrows():
            country = row["Country"]
            if country not in seen_populations:
                pop_id = self.samples.add_population(
                    {"name": country, "locality": row["Locality"]}
                )
                population_id_map[country] = pop_id
                seen_populations.append(country)

        # The file contains some non UTF-8 codepoints for a contributors name.
        with open(metadata_file, "r") as md_file:
            columns = next(md_file).split("\t")
            sane_names = [
                col.lower().split(" (")[0].split(" [")[0].replace(" ", "_")
                for col in columns
            ]
            rows = {}
            populations = {}
            locations = {}
            for line in md_file:
                metadata = dict(zip(sane_names, line.strip().split("\t")))
                name = metadata["index"] + "_" + metadata["instance_id"]
                population_name = metadata.pop("country")
                populations[name] = population_id_map[population_name]
                age = metadata.pop("average_of_95.4%_date_range_in_calbp")
                metadata["age"] = int(age) / GENERATION_TIME
                rows[name] = metadata
                try:
                    location = [
                        float(metadata.pop("lat.")),
                        float(metadata.pop("long.")),
                    ]
                except:
                    pass
                locations[name] = location
        vcf = cyvcf2.VCF(self.data_file)
        individual_names = list(vcf.samples)
        vcf.close()
        self.num_samples = len(individual_names) * 2

        # Add in the metadata rows in the order of the VCF.
        for name in individual_names:
            metadata = rows[name]
            self.samples.add_individual(
                time=metadata["age"],
                metadata=metadata,
                location=locations[name],
                ploidy=2,
                population=populations[name],
            )

#    def convert_genotypes(self, row, ancestral_state):
#        ret = None
#        num_diploids = self.num_samples // 2
#        a = np.zeros(self.num_samples, dtype=np.int8)
#        all_alleles = set([ancestral_state])
#        # Fill in a with genotypes.
#        bases = np.array(row.gt_bases)
#        for j in range(num_diploids):
#            alleles = bases[j].split("/")
#            if len(alleles) != 2:
#                self.num_unphased += 1
#                break
#            missing = False
#            for allele in alleles:
#                if allele == ".":
#                    missing = True
#                else:
#                    all_alleles.add(allele)
#            a[2 * j] = alleles[0] != ancestral_state
#            a[2 * j + 1] = alleles[1] != ancestral_state
#            if missing:
#                if alleles[0] == ".":
#                    a[2 * j] = tskit.MISSING_DATA
#                if alleles[1] == ".":
#                    a[2 * j + 1] = tskit.MISSING_DATA
#        else:
#            freq = np.sum(a)
#            # The loop above exited without breaking, so we have valid data.
#            if freq == 0:
#                self.num_invariant += 1
#            elif any(len(allele) != 1 for allele in all_alleles):
#                self.num_indels += 1
#            elif len(all_alleles) != 2:
#                self.num_non_biallelic += 1
#            else:
#                all_alleles.remove(ancestral_state)
#                alleles = [ancestral_state, all_alleles.pop()]
#                metadata = {"ID": row.ID, "REF": row.REF}
#                ret = Site(
#                    position=row.POS, alleles=alleles, genotypes=a, metadata=metadata
#                )
#        return ret
#
#    def process_sites(self, show_progress=False, max_sites=None):
#        num_data_sites = int(
#            subprocess.check_output(["bcftools", "index", "--nrecords", self.data_file])
#        )
#        progress = tqdm.tqdm(total=num_data_sites, disable=not show_progress)
#        seen_positions = list()
#        num_sites = 0
#        for index, row in enumerate(filter_duplicates(cyvcf2.VCF(self.data_file))):
#            ancestral_state = self.get_ancestral_state(row.POS)
#            if ancestral_state is not None:
#                site = self.convert_genotypes(row, ancestral_state)
#                if site is not None:
#                    self.samples.add_site(
#                        position=site.position,
#                        genotypes=site.genotypes,
#                        alleles=site.alleles,
#                        metadata=site.metadata,
#                    )
#                    seen_positions.append(site.position)
#                    progress.set_postfix(used=str(num_sites))
#                    num_sites += 1
#                    if num_sites == max_sites:
#                        break
#            progress.update()
#
#        progress.close()


class UkbbConverter(Converter):
    def process_metadata(self, metadata_file, show_progress=False):
        # TODO Should make this an explicit requirement rather than hardcoding.
        withdrawn_ids = set()
        with open("ukbb_withdrawn.csv") as f:
            for line in f:
                withdrawn_ids.add(int(line))

        # The sample IDs aren't in the BGEN file so we have to match by the Order
        # field, which gives the order that each sample is at in the BGEN (0 based).
        metadata_df = pd.read_csv(metadata_file)
        metadata_df.sort_values(by="Order", inplace=True)
        metadata_df = metadata_df.set_index("Order")

        bgen = bgen_reader.read_bgen(self.data_file, verbose=False)
        sample_df = bgen["samples"]
        keep_samples = []
        row_iter = tqdm.tqdm(
            metadata_df.iterrows(), total=len(metadata_df), disable=not show_progress
        )
        for index, row in row_iter:
            if not pd.isnull(index):
                order = int(index)
                if int(row.SampleID) not in withdrawn_ids:
                    keep_samples.extend([2 * order, 2 * order + 1])
                    metadata = {}
                    for k, v in row.items():
                        metadata[k] = None if pd.isnull(v) else str(v)
                    self.samples.add_individual(ploidy=2, metadata=metadata)
        self.num_samples = len(keep_samples)
        self.keep_index = np.array(keep_samples, dtype=int)

    def process_sites(self, show_progress=False, max_sites=None):

        bgen = bgen_reader.read_bgen(self.data_file, verbose=False)
        num_alleles = np.array(bgen["variants"]["nalleles"])
        position = np.array(bgen["variants"]["pos"])
        rsid = np.array(bgen["variants"]["rsid"])
        allele_id = np.array(bgen["variants"]["allele_ids"])
        del bgen

        bg = simplebgen.BgenReader(self.data_file)
        N = 2 * bg.num_samples
        for j in tqdm.tqdm(range(bg.num_variants)):
            ancestral_state = self.get_ancestral_state(position[j])
            if ancestral_state is not None:
                alleles = allele_id[j].split(",")
                if num_alleles[j] != 2 or ancestral_state not in alleles:
                    self.num_non_biallelic += 1
                elif any(len(allele) != 1 for allele in alleles):
                    self.num_indels += 1
                else:
                    P = bg.get_probabilities(j).astype(np.int8).reshape((N, 2))
                    # The probabilities for each site is a (num_diploids, 4) array,
                    # in the form (n0_a0, n0_a1, n1_a0, n1_a1). These are always zero
                    # or one for the different alleles. We first flatten this array so
                    # that it's (N, 2) and then generate the genotypes based on that.
                    genotypes = np.zeros(N, dtype=np.int8)
                    if ancestral_state == alleles[0]:
                        genotypes[P[:, 1] == 1] = 1
                        ref = alleles[0]
                    else:
                        genotypes[P[:, 0] == 1] = 1
                        ref = alleles[0]
                        alleles = alleles[::-1]

                    freq = np.sum(genotypes)
                    if freq == self.num_samples or freq == 0:
                        self.num_invariant += 1
                    elif freq == 1:
                        self.num_singletons += 1
                    elif freq == self.num_samples - 1:
                        self.num_nmo_tons += 1
                    else:
                        metadata = {"ID": rsid[j], "REF": ref}
                        self.samples.add_site(
                            position=float(position[j]),
                            genotypes=genotypes[self.keep_index],
                            alleles=alleles,
                            metadata=metadata,
                        )
            if j == max_sites:
                break
        self.report()


def main():
    parser = argparse.ArgumentParser(
        description="Script to convert VCF files into tsinfer input."
    )
    parser.add_argument(
        "source",
        choices=["1kg", "sgdp", "ukbb", "hgdp", "max-planck", "afanasievo", "1240k"],
        help="The source of the input data.",
    )
    parser.add_argument("data_file", help="The input data file pattern.")
    parser.add_argument(
        "ancestral_states_file", help="A vcf file containing ancestral allele states. "
    )
    parser.add_argument("output_file", help="The tsinfer output file")
    parser.add_argument(
        "-m",
        "--metadata_file",
        default=None,
        help="The metadata file containing population and sample data",
    )
    parser.add_argument(
        "-n",
        "--max-variants",
        default=None,
        type=int,
        help="Keep only the first n variants",
    )
    parser.add_argument(
        "--target-samples",
        default=None,
        help="A target sampledata file, only variants present in this target file will \
            will be used in the resulting sampledata file"
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Show progress bars and output extra information when done",
    )
    parser.add_argument(
        "--ancestral-states-url",
        default=None,
        help="The source of ancestral state information for provenance.",
    )
    parser.add_argument(
        "--reference-name",
        default=None,
        help="The name of the reference for provenance.",
    )
    parser.add_argument(
        "--num-threads", type=int, default=1, help="Number of threads to use."
    )

    args = parser.parse_args()

    if args.num_threads > 1:
        run_multiprocessing(args, make_sampledata)
    else:
        report = make_sampledata(args)
        print(report)


if __name__ == "__main__":
    main()

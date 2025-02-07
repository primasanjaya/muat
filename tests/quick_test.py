import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from muat.download import download_icgc_object_storage as download
from muat.download import download_reference
from muat.preprocessing import preprocessing_vcf

import pdb

files_to_download = ['PCAWG/consensus_snv_indel/README.md',
        'PCAWG/consensus_snv_indel/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz',
        'PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public.tgz',
        'PCAWG/consensus_sv/README.md',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.icgc.public.tgz',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.tcga.public.tgz',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-05-27.blacklisted_donors.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-05-27.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-06-24.blacklisted_donors.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-06-24.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-06-30.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-08-12.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.v1.3.2016-09-11.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.v1.4.2016-09-14.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.1.jsonl',
        'PCAWG/data_releases/latest/release_may2016.v1.1.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.2.jsonl',
        'PCAWG/data_releases/latest/release_may2016.v1.2.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.3.jsonl',
        'PCAWG/data_releases/latest/release_may2016.v1.3.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.4.jsonl',
        'PCAWG/data_releases/latest/release_may2016.v1.4.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.4.with_consensus_calls.jsonl',
        'PCAWG/data_releases/latest/release_may2016.v1.4.with_consensus_calls.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.blacklisted_donors.jsonl',
        'PCAWG/data_releases/latest/release_may2016.v1.blacklisted_donors.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.jsonl',
        'PCAWG/data_releases/latest/release_may2016.v1.tsv']

#download data
#download(data_path="./data/", files_to_download=files_to_download)

#download reference genome
genome_reference_path = '/Users/primasan/Documents/work/muat/data/genome_reference/'
download_reference(genome_reference_path=genome_reference_path)

#preprocess vcf
'''
To process VCF files, you need to specify the following arguments:
- vcf_file: path to the VCF file
- genome_reference_path: path to the genome reference file that matches the VCF file
- tmp_dir: path to the temporary directory for storing preprocessed files
'''

preprocessing_vcf(vcf_file="/Users/primasan/Documents/work/muat/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public/snv_mnv/0a9c9db0-c623-11e3-bf01-24c6515278c0.consensus.20160830.somatic.snv_mnv.vcf.gz",
genome_reference_path=genome_reference_path+"hg19.fa",tmp_dir="/Users/primasan/Documents/work/muat/data/preprocessed/")










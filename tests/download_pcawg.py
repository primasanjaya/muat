import sys
import os
import tarfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from muat.download import download_icgc_object_storage as download
from muat.download import download_reference
import glob
import pandas as pd
import pdb

files_to_download = ['PCAWG/consensus_snv_indel/README.md',
        'PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public.tgz',
        'PCAWG/consensus_sv/README.md',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.icgc.public.tgz',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.tcga.public.tgz',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.v1.4.2016-09-14.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.4.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-08-12.tsv',
        'PCAWG/clinical_and_histology/pcawg_specimen_histology_August2016_v9.xlsx']

muat_dir = '/path/to/muat'

download_data_path = muat_dir + '/data/'
#download data
download(data_path=download_data_path, files_to_download=files_to_download)

tgz_file_path = muat_dir + '/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public.tgz'
# Extract the .tgz file
with tarfile.open(tgz_file_path, 'r:gz') as tar:
    tar.extractall(path=muat_dir + '/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public/')  # Specify the directory to extract to
    print("Extraction complete.")
import sys
import os
import tarfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from muat.download import download_icgc_object_storage as download
from muat.download import download_reference
from muat.preprocessing import *
import glob
import pandas as pd
import pdb
from muat.util import mutation_type_ratio,model_input
from muat.dataloader import MuAtDataloader,DataloaderConfig
from muat.trainer import *
from muat.model import *

#preprocess vcf
'''
To process VCF files, you need to specify the following arguments:
- vcf_file: str or list of path to the VCF file
- genome_reference_path: path to the genome reference file that matches the VCF file
- tmp_dir: path to the temporary directory for storing preprocessed files
'''

muat_dir = '/path/to/muat'
genome_reference_38_path = 'path/to/genome_reference/hg38.fa.gz'
genome_reference_19_path = 'path/to/genome_reference/hg19.fa.gz'
tmp_dir = 'path/to/preprocessed/'

#tokenizing motif position and ges using dict_motif, dict_pos, dict_ges
dict_motif = pd.read_csv(muat_dir + '/muat/extfile/dictMutation.tsv',sep='\t')
dict_pos = pd.read_csv(muat_dir + '/muat/extfile/dictChpos.tsv',sep='\t')
dict_ges = pd.read_csv(muat_dir + '/muat/extfile/dictGES.tsv',sep='\t')

#example for preprocessing multiple vcf files
vcf_files = 'path/to/input/sample.vcf.gz'
tmp_dir = muat_dir + '/data/preprocessed/'
vcf_files = multifiles_handler(vcf_files)

#run preprocessing and tokenizing
preprocessing_vcf38_tokenizing(vcf_file=vcf_files,
                            genome_reference_38_path=genome_reference_38_path,
                            genome_reference_19_path=genome_reference_19_path,
                            tmp_dir=tmp_dir,
                            dict_motif=dict_motif,
                            dict_pos=dict_pos,
                            dict_ges=dict_ges)
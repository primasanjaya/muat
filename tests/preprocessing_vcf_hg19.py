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
genome_reference_path = 'path/to/genome/reference/hg19/'
tmp_dir = 'path/to/preprocessed/'

#download reference genome
genome_reference_path = muat_dir + '/data/genome_reference/'
download_reference(genome_reference_path=genome_reference_path)

#example for preprocessing multiple vcf files
all_vcf = glob.glob(muat_dir + '/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public/snv_mnv/*.vcf.gz')
preprocessing_vcf(vcf_file=all_vcf,
genome_reference_path=genome_reference_path+"hg19.fa.gz",tmp_dir=tmp_dir)

#tokenizing motif position and ges using dict_motif, dict_pos, dict_ges

dict_motif = pd.read_csv(muat_dir + '/muat/extfile/dictMutation.tsv',sep='\t')
dict_pos = pd.read_csv(muat_dir + '/muat/extfile/dictChpos.tsv',sep='\t')
dict_ges = pd.read_csv(muat_dir + '/muat/extfile/dictGES.tsv',sep='\t')

#get all preprocessed vcf
all_preprocessed_vcf = glob.glob(muat_dir + '/data/preprocessed/*gc.genic.exonic.cs.tsv.gz')
#tokenizing
tokenizing(dict_motif, dict_pos, dict_ges,all_preprocessed_vcf,pos_bin_size=1000000)
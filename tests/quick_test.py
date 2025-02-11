import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from muat.download import download_icgc_object_storage as download
from muat.download import download_reference
from muat.preprocessing import *
import glob
import pandas as pd
import pdb
from muat.util import mutation_type_ratio,model_input
from muat.dataloader import MuAtDataloader
from muat.trainer import *
from muat.model import *

files_to_download = ['PCAWG/consensus_snv_indel/README.md',
        'PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public.tgz',
        'PCAWG/consensus_sv/README.md',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.icgc.public.tgz',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.tcga.public.tgz',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.v1.4.2016-09-14.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.4.tsv',
        'PCAWG/clinical_and_histology/pcawg_specimen_histology_August2016_v9.xlsx']

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

#example for preprocessing single vcf file
'''
preprocessing_vcf(vcf_file="/Users/primasan/Documents/work/muat/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public/snv_mnv/0a9c9db0-c623-11e3-bf01-24c6515278c0.consensus.20160830.somatic.snv_mnv.vcf.gz",
genome_reference_path=genome_reference_path+"hg19.fa",tmp_dir="/Users/primasan/Documents/work/muat/data/preprocessed/")

#example for preprocessing multiple vcf files

all_vcf = glob.glob('/Users/primasan/Documents/work/muat/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public/snv_mnv/*.vcf.gz')
preprocessing_vcf(vcf_file=all_vcf,
genome_reference_path=genome_reference_path+"hg19.fa",tmp_dir="/Users/primasan/Documents/work/muat/data/preprocessed/")
'''

#tokenizing moitf position and ges using dict_motif, dict_pos, dict_ges
'''
#get all preprocessed vcf
all_preprocessed_vcf = glob.glob('/Users/primasan/Documents/work/muat/data/preprocessed/*gc.genic.exonic.cs.tsv.gz')

#create dictionary
dict_motif, dict_pos, dict_ges = create_dictionary(all_preprocessed_vcf,pos_bin_size=1000000,save_dict_path='/Users/primasan/Documents/work/muat/data/dict/dict_motif_pos_ges.json')
'''

dict_motif = pd.read_csv('/Users/primasan/Documents/work/muat/extfile/dictMutation.tsv',sep='\t')
dict_pos = pd.read_csv('/Users/primasan/Documents/work/muat/extfile/dictChpos.tsv',sep='\t')
dict_ges = pd.read_csv('/Users/primasan/Documents/work/muat/extfile/dictGES.tsv',sep='\t')
'''
#tokenizing
tokenizing(dict_motif, dict_pos, dict_ges,all_preprocessed_vcf,pos_bin_size=1000000)
'''

#prepare training split
metadata_path = '/Users/primasan/Documents/work/muat/data/PCAWG/data_releases/latest/pcawg_sample_sheet.2016-08-12.tsv'
pd_meta = pd.read_csv(metadata_path, sep='\t')
histology_path = '/Users/primasan/Documents/work/muat/data/PCAWG/clinical_and_histology/pcawg_specimen_histology_August2016_v9.xlsx'
pd_histology = pd.read_excel(histology_path)
# merge pd_meta and pd_consensus
pd_merged = pd.merge(pd_meta, pd_histology, left_on='icgc_donor_id', right_on='icgc_donor_id', how='inner')
pd_merged = pd_merged.loc[pd_merged['specimen_library_strategy'] == 'WGS']
pd_merged = pd_merged[['aliquot_id','histology_abbreviation']].drop_duplicates()

#get all preprocessed vcf
all_preprocessed_vcf = glob.glob('/Users/primasan/Documents/work/muat/data/preprocessed/*gc.genic.exonic.cs.tsv.gz')
pd_preprocessed_vcf = pd.DataFrame(all_preprocessed_vcf, columns=['prep_path'])

# Split the directory and filename
pd_preprocessed_vcf['directory'], pd_preprocessed_vcf['filename'] = zip(*pd_preprocessed_vcf['prep_path'].apply(os.path.split))
pd_preprocessed_vcf['aliquot_id'] = pd_preprocessed_vcf['filename'].str.split('.gc.genic.exonic.cs.tsv.gz',expand=True)[0]

pd_preprocessed_vcf = pd_preprocessed_vcf.merge(pd_merged,on='aliquot_id',how='inner')
pd_preprocessed_vcf['nm_class'] = pd_preprocessed_vcf['histology_abbreviation']
pd_preprocessed_vcf = pd_preprocessed_vcf[['prep_path','nm_class']]


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
pd_preprocessed_vcf['idx_class'] = le.fit_transform(pd_preprocessed_vcf['nm_class'])

train_split, test_split = train_test_split(pd_preprocessed_vcf, test_size=0.2, random_state=42,stratify=pd_preprocessed_vcf['nm_class'])

#hyperparameter setup

mutation_sampling_size = 5000
mutation_type,motif_size = mutation_type_ratio(snv=0.5,mnv=0.5,indel=0,sv_mei=0,neg=0,pd_motif=dict_motif) #proportion of mutation type
model_input = model_input(motif=True,pos=True,ges=True) #model input

train_dataloader = MuAtDataloader( train_split,model_input,mutation_type,mutation_sampling_size)
test_dataloader = MuAtDataloader( test_split,model_input,mutation_type,mutation_sampling_size)

#pdb.set_trace()

n_layer = 2
n_emb = 128
n_head = 1
n_class = len(le.classes_)

# Ensure these values are set correctly
mutation_type, motif_size = mutation_type_ratio(snv=0.5, mnv=0.5, indel=0, sv_mei=0, neg=0, pd_motif=dict_motif)

# Check the values before creating the model
model_config = ModelConfig(
    motif_size=motif_size,
    num_class=n_class,
    mutation_sampling_size=mutation_sampling_size,
    position_size=len(dict_pos),
    ges_size=len(dict_ges),
    n_embd=n_emb,
    n_layer=n_layer,
    n_head=n_head
)

model = MuAtMotif(model_config)

n_epochs = 10
batch_size = 4
learning_rate = 0.001

trainer_config = TrainerConfig(max_epochs=n_epochs, 
                                batch_size=batch_size, 
                                learning_rate=learning_rate, 
                                num_workers=1,
                                save_ckpt_dir='/Users/primasan/Documents/work/muat/data/ckpt/')

trainer = Trainer(model, train_dataloader, test_dataloader, trainer_config)
trainer.batch_train()










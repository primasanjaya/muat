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
from muat.dataloader import MuAtDataloader
from muat.trainer import *
from muat.predict import *
from muat.model import *

muat_dir = '/csc/epitkane/projects/github/muat'
muat_dir = '/Users/primasan/Documents/work/muat'
device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()

genome_reference_path = muat_dir + '/data/genome_reference/'

#load ckpt
load_ckpt_path = muat_dir + '/data/ckpt/2class/best_ckpt.pthx'
checkpoint = torch.load(load_ckpt_path)
dict_motif,dict_pos,dict_ges = load_token_dict(checkpoint)

#preprocess vcf
'''
To process VCF files, you need to specify the following arguments:
- vcf_file: path to the VCF file
- genome_reference_path: path to the genome reference file that matches the VCF file
- tmp_dir: path to the temporary directory for storing preprocessed files
'''
'''
#example for preprocessing single vcf file
preprocessing_vcf_tokenizing(vcf_file=muat_dir + '/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public/snv_mnv/0a9c9db0-c623-11e3-bf01-24c6515278c0.consensus.20160830.somatic.snv_mnv.vcf.gz',
genome_reference_path=genome_reference_path+"hg19.fa.gz",tmp_dir=muat_dir + '/data/preprocessed/',dict_motif=dict_motif,dict_pos=dict_pos,dict_ges=dict_ges)
'''

#example for preprocessing multiple vcf files
vcf_files = glob.glob(muat_dir + '/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public/snv_mnv/*.vcf.gz')
vcf_files = vcf_files[0:5]

tmp_dir = muat_dir + '/data/preprocessed_test/'
preprocessing_vcf_tokenizing(vcf_file=vcf_files,
                            genome_reference_path=genome_reference_path+"hg19.fa.gz",
                            tmp_dir=tmp_dir,
                            dict_motif=dict_motif,
                            dict_pos=dict_pos,
                            dict_ges=dict_ges)

predict_ready_files = []
for x in vcf_files:
    if os.path.exists(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
        predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')

pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])

le = load_target_handler(checkpoint)

mutation_sampling_size = 1000
mutation_type,motif_size = mutation_type_ratio(snv=0.5,mnv=0.5,indel=0,sv_mei=0,neg=0,pd_motif=dict_motif) #proportion of mutation type
model_input = model_input(motif=True,pos=True,ges=True) #model input

#pdb.set_trace()
dataloader_config = checkpoint['dataloader_config']
test_dataloader = MuAtDataloader( pd_predict,dataloader_config)

model_config = checkpoint['model_config']
model = checkpoint['model']

#load weight to the model
model = model.to(device)
model.load_state_dict(checkpoint['weight'])

predict_config = PredictorConfig(max_epochs=1, batch_size=1,load_ckpt_dir=os.path.dirname(load_ckpt_path),target_handler=le)
predictor = Predictor(model, test_dataloader, predict_config)

predictor.batch_predict()




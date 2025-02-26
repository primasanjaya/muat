import sys
import os
import tarfile

from muat.download import *
from muat.preprocessing import *
import glob
import pandas as pd
import pdb
from muat.util import *
from muat.dataloader import *
from muat.trainer import *
from muat.predict import *
from muat.model import *
from muat.checkpoint import *

if __name__ == "__main__":

    args = get_main_args()

    if args.predict_vcf_hg19:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        genome_reference_path = args.hg19_filepath
        #load ckpt
        load_ckpt_path = args.load_ckpt_filepath
        checkpoint = load_and_check_checkpoint(load_ckpt_path)

        dict_motif,dict_pos,dict_ges = load_token_dict(checkpoint)

        #example for preprocessing multiple vcf files
        vcf_files = args.vcf_hg19_filepath
        tmp_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/preprocessed/'))
        preprocessing_vcf_tokenizing(vcf_file=vcf_files,
                                    genome_reference_path=genome_reference_path,
                                    tmp_dir=tmp_dir,
                                    dict_motif=dict_motif,
                                    dict_pos=dict_pos,
                                    dict_ges=dict_ges)
        predict_ready_files = []
        for x in vcf_files:
            if os.path.exists(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
                predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')

        pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])
        target_handler = load_target_handler(checkpoint)

        dataloader_config = checkpoint['dataloader_config']
        test_dataloader = MuAtDataloader(pd_predict,dataloader_config)

        #pdb.set_trace()
        model_name = checkpoint['model_name']
        model = get_model(model_name,checkpoint['model_config'])
        model = model.to(device)
        model.load_state_dict(checkpoint['weight'])

        result_dir = args.result_dir
        predict_config = PredictorConfig(max_epochs=1, batch_size=1,result_dir=result_dir,target_handler=target_handler)
        predictor = Predictor(model, test_dataloader, predict_config)

        predictor.batch_predict()

    
    if args.preprocessing_vcf_hg19:

        #preprocess vcf
        '''
        To process VCF files, you need to specify the following arguments:
        - vcf_file: str or list of path to the VCF file
        - genome_reference_path: path to the genome reference file that matches the VCF file
        - tmp_dir: path to the temporary directory for storing preprocessed files
        '''
        genome_reference_path = args.hg19_filepath
        tmp_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/preprocessed/'))

        #example for preprocessing multiple vcf files
        vcf_files = args.vcf_hg19_filepath
        preprocessing_vcf(vcf_file=all_vcf,genome_reference_path=genome_reference_path,tmp_dir=tmp_dir)

    if args.preprocessing_vcf_hg38:
        tmp_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/preprocessed/'))
        genome_reference_38_path = args.hg38_filepath
        genome_reference_19_path = args.hg38_filepath

        #example for preprocessing multiple vcf files
        vcf_files = args.vcf_hg38_filepath
        #run preprocessing 
        preprocessing_vcf38(vcf_file,genome_reference_38_path,genome_reference_19_path,tmp_dir)

    if args.tokenizing:

        dict_motif = args.motif_dictionary_filepath
        dict_pos = args.position_dictionary_filepath
        dict_ges = args.ges_dictionary_filepath
        all_preprocessed_vcf = args.tmp_dir
        tokenizing(dict_motif, dict_pos, dict_ges,all_preprocessed_vcf,pos_bin_size=1000000)

    

    
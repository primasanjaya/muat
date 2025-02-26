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

        if not args.hg19_filepath or not args.load_ckpt_filepath or not args.vcf_hg19_filepath or not args.result_dir:
            raise ValueError('--predict-vcf-hg19 requires --load-ckpt-filepath --vcf-hg19-filepath --result-dir --hg19-filepath')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        genome_reference_path = args.hg19_filepath
        #load ckpt
        load_ckpt_path = args.load_ckpt_filepath
        checkpoint = load_and_check_checkpoint(load_ckpt_path)

        dict_motif,dict_pos,dict_ges = load_token_dict(checkpoint)

        #example for preprocessing multiple vcf files
        vcf_files = args.vcf_hg19_filepath
        tmp_dir = check_tmp_dir(args)
        preprocessing_vcf_tokenizing(vcf_file=vcf_files,
                                    genome_reference_path=genome_reference_path,
                                    tmp_dir=tmp_dir,
                                    dict_motif=dict_motif,
                                    dict_pos=dict_pos,
                                    dict_ges=dict_ges)
        print('preprocessed data saved in ' + tmp_dir)
        predict_ready_files = []
        for x in vcf_files:
            if os.path.exists(tmp_dir + '/' + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz'):
                predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

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

    if args.predict_vcf_hg38:

        if not args.load_ckpt_filepath or not args.hg38_filepath or not args.hg19_filepath or not args.vcf_hg38_filepath or not args.result_dir:
            raise ValueError('--predict-vcf-hg38 requires --load-ckpt-filepath --vcf-hg38-filepath --hg38-filepath --hg19-filepath --result-dir')

        #load ckpt
        load_ckpt_path = args.load_ckpt_filepath
        checkpoint = load_and_check_checkpoint(load_ckpt_path)
        tmp_dir = check_tmp_dir(args)

        genome_reference_38_path = args.hg38_filepath
        genome_reference_19_path = args.hg19_filepath

        dict_motif,dict_pos,dict_ges = load_token_dict(checkpoint)
        #example for preprocessing multiple vcf files
        vcf_files = args.vcf_hg38_filepath
        
        preprocessing_vcf38_tokenizing(vcf_file=vcf_files,
                                    genome_reference_38_path=genome_reference_38_path,
                                    genome_reference_19_path=genome_reference_19_path,
                                    tmp_dir=tmp_dir,
                                    dict_motif=dict_motif,
                                    dict_pos=dict_pos,
                                    dict_ges=dict_ges)

        print('preprocessed data saved in ' + tmp_dir)
        predict_ready_files = []
        for x in vcf_files:
            if os.path.exists(tmp_dir + '/' + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz'):
                predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

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

        if not args.hg19_filepath or not args.vcf_hg19_filepath:
            raise ValueError('--preprocessing-vcf-hg19 requires --hg19-filepath --vcf-hg19-filepath')
        genome_reference_path = args.hg19_filepath
        tmp_dir = check_tmp_dir(args)

        #example for preprocessing multiple vcf files
        vcf_files = args.vcf_hg19_filepath
        preprocessing_vcf(vcf_file=vcf_files,genome_reference_path=genome_reference_path,tmp_dir=tmp_dir)

    if args.preprocessing_vcf_hg38:
        if not args.hg19_filepath or not args.vcf_hg38_filepath or not args.hg38_filepath:
            raise ValueError('--preprocessing-vcf-hg38 requires --hg38-filepath --hg19-filepath --vcf-hg38-filepath')
        tmp_dir = check_tmp_dir(args)
        genome_reference_38_path = args.hg38_filepath
        genome_reference_19_path = args.hg19_filepath

        #example for preprocessing multiple vcf files
        vcf_files = args.vcf_hg38_filepath
        #run preprocessing 
        preprocessing_vcf38(vcf_files,genome_reference_38_path,genome_reference_19_path,tmp_dir)

    if args.tokenizing:
        dict_motif = pd.read_csv(args.motif_dictionary_filepath,sep='\t')
        dict_pos = pd.read_csv(args.position_dictionary_filepath,sep='\t')
        dict_ges = pd.read_csv(args.ges_dictionary_filepath,sep='\t')
        all_preprocessed_vcf = args.preprocessed_filepath

        tmp_dir = check_tmp_dir(args)
        tokenizing(dict_motif, dict_pos, dict_ges,all_preprocessed_vcf,pos_bin_size=1000000,tmp_dir=tmp_dir)

    

    

    
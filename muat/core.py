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
import warnings
from pkg_resources import resource_filename

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

    if args.train:

        if not args.motif_dictionary_filepath or not args.position_dictionary_filepath or not args.ges_dictionary_filepath:
            
            extdir = resource_filename('muat', 'extfile')
            motif_path = extdir + '/' + 'dictMutation.tsv'
            pos_path = extdir + '/' + 'dictChpos.tsv'
            ges_path = extdir + '/' + 'dictGES.tsv'
            
            warnings.warn("Some dictionary file paths were not defined and have been set automatically.\n"
    "--motif-dictionary-filepath is set to {}, --position-dictionary-filepath is set to {}, --ges-dictionary-filepath is set to {}\n"
    "These motif position ges dictionary might be different from your preprocessed files!".format(
        motif_path, pos_path, ges_path))

        else:
            motif_path = args.motif_dictionary_filepath
            pos_path = args.position_dictionary_filepath
            ges_path = args.ges_dictionary_filepath

        dict_motif = pd.read_csv(motif_path,sep='\t')
        dict_pos = pd.read_csv(pos_path,sep='\t')
        dict_ges = pd.read_csv(ges_path,sep='\t')

        if not args.arch or not args.mutation_type or not args.n_layer or not args.n_emb or not args.n_head or not args.target_dict_filepath:
            raise ValueError('--train requires --arch --mutation-type --n-layer n-emb --n-head --target-dict-filepath')

        model_name = args.arch
        mutation_type = args.mutation_type
        n_layer = args.n_layer
        n_emb = args.n_emb
        n_head = args.n_head
        mutation_sampling_size = args.mutation_sampling_size

        target_handler = []

        le = LabelEncoderFromCSV(csv_file=args.target_dict_filepath)
        n_class = len(le.classes_)
        target_handler.append(le)

        model_config = ModelConfig(
                            model_name,
                            dict_motif,
                            dict_pos,
                            dict_ges,
                            mutation_sampling_size,
                            n_layer,
                            n_emb,
                            n_head,
                            n_class, 
                            mutation_type)

        model = get_model(model_name,model_config)

        if args.load_ckpt_filepath is not None:
            model = initialize_pretrained_weight(model_name,model_config,checkpoint)

        if args.subtarget_dict_filepath is not None:
            le2 = LabelEncoderFromCSV(csv_file=args.subtarget_dict_filepath)
            subclass = len(le2.classes_)
            target_handler.append(le2)
            model_config.num_subclass = subclass
        #pdb.set_trace()
        train_dataloader_config = DataloaderConfig(model_input=model_config.model_input,mutation_type_ratio=model_config.mutatation_type_ratio,mutation_sampling_size=mutation_sampling_size)
        test_dataloader_config = DataloaderConfig(model_input=model_config.model_input,mutation_type_ratio=model_config.mutatation_type_ratio,mutation_sampling_size=mutation_sampling_size)
        
        train_split = pd.read_csv(args.train_split_filepath,sep='\t',low_memory=False)
        test_split = pd.read_csv(args.val_split_filepath,sep='\t',low_memory=False)

        train_dataloader = MuAtDataloader(train_split,train_dataloader_config)
        test_dataloader = MuAtDataloader(test_split,test_dataloader_config)

        n_epochs = args.epoch
        batch_size = args.batch_size
        learning_rate = args.learning_rate

        save_ckpt_dir = args.save_ckpt_dir
        trainer_config = TrainerConfig(max_epochs=n_epochs, 
                                        batch_size=batch_size, 
                                        learning_rate=learning_rate, 
                                        num_workers=1,
                                        save_ckpt_dir=save_ckpt_dir,
                                        target_handler=target_handler)
        trainer = Trainer(model, train_dataloader, test_dataloader, trainer_config)
        trainer.batch_train()

    

    
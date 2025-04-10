import sys
import os
import tarfile
import zipfile
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

def validate_checkpoint(trainer_config):
    """Ensure required hyperparameters exist in the checkpoint."""
    missing_params = []
    if not hasattr(trainer_config, 'max_epochs') or trainer_config.max_epochs is None:
        missing_params.append('max_epochs')
    if not hasattr(trainer_config, 'batch_size') or trainer_config.batch_size is None:
        missing_params.append('batch_size')
    if not hasattr(trainer_config, 'learning_rate') or trainer_config.learning_rate is None:
        missing_params.append('learning_rate')

    if missing_params:
        raise ValueError(f"Checkpoint is missing required parameters: {', '.join(missing_params)}.\n"
                         "Please provide these values via command-line arguments (--epoch, --batch-size, --learning-rate).")

def load_data(train_path, val_path):
            train_split = pd.read_csv(train_path, sep='\t', low_memory=False)
            test_split = pd.read_csv(val_path, sep='\t', low_memory=False)
            return train_split, test_split

def initialize_label_encoders(target_path, subtarget_path=None):
    target_handler = [LabelEncoderFromCSV(csv_file=target_path, class_name_col='class_name', class_index_col='class_index')]
    n_class = len(target_handler[0].classes_)

    n_subclass = None
    if subtarget_path is not None:
        le2 = LabelEncoderFromCSV(csv_file=subtarget_path, class_name_col='subclass_name', class_index_col='subclass_index')
        target_handler.append(le2)
        n_subclass = len(le2.classes_)

    return target_handler, n_class, n_subclass

def setup_trainer(model, train_data, test_data, trainer_config):
    train_dataloader = MuAtDataloader(train_data, trainer_config.dataloader_config)
    test_dataloader = MuAtDataloader(test_data, trainer_config.dataloader_config)
    return Trainer(model, train_dataloader, test_dataloader, trainer_config)

def main():
    args = get_main_args()
    genomedir = resource_filename('muat', 'genome_reference')
    genomedir = ensure_dirpath(genomedir)

    pkg_ckpt = resource_filename('muat', 'pkg_ckpt')
    pkg_ckpt = ensure_dirpath(pkg_ckpt)
    unziping_from_package_installation()
    #pdb.set_trace()

    if args.command == 'download':

        files_to_download = ['PCAWG/consensus_snv_indel/README.md',
        'PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public.tgz',
        'PCAWG/consensus_sv/README.md',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.icgc.public.tgz',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.tcga.public.tgz',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.v1.4.2016-09-14.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.4.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-08-12.tsv',
        'PCAWG/clinical_and_histology/pcawg_specimen_histology_August2016_v9.xlsx']

        download_data_path = args.download_dir    
        #download data
        download_icgc_object_storage(data_path=download_data_path, files_to_download=files_to_download)
        # Specify the directory to extract to
        print("Download completed. Data saved in " + str(download_data_path))

    if args.command == 'wgs' or args.command=='wes':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        genome_reference_path_hg19 = args.hg19
        genome_reference_path_hg38 = args.hg38
        #load ckpt

        if args.mutation_type is not None:
            if args.command=='wgs':
                wgs_wes = 'wgs'
                benchmark_ckpt = resource_filename('muat', 'pkg_ckpt')
                benchmark_ckpt = ensure_dirpath(benchmark_ckpt) + 'pcawg_wgs/'
                url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/best_wgs_pcawg.zip"
            else:
                wgs_wes = 'wes'
                benchmark_ckpt = resource_filename('muat', 'pkg_ckpt')
                benchmark_ckpt = ensure_dirpath(benchmark_ckpt) + 'tcga_wes/'
                url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/best_wes_tcga.zip"

            check_pth = glob.glob(benchmark_ckpt + args.mutation_type + '/*.pthx')
            if len(check_pth)==0:
                print('cant find model in ' + benchmark_ckpt + args.mutation_type + '. Downloading model from ' + url )
                download_checkpoint(url,'my_checkpoint.zip')
                check_pth = glob.glob(benchmark_ckpt + args.mutation_type + '/*.pthx')
            if len(check_pth) == 0:
                raise ValueError('cant find benchmark model in ' + benchmark_ckpt + args.mutation_type + '. Download benchmark model from ' + url + ' and extract to this path.')

            load_ckpt_path = mut_type_checkpoint_handler(args.mutation_type,wgs_wes)
        else:
            load_ckpt_path = args.ckpt_filepath
        
        checkpoint = load_and_check_checkpoint(load_ckpt_path)
        model_name = checkpoint['model_name']

        dict_motif,dict_pos,dict_ges = load_token_dict(checkpoint)
        vcf_files = multifiles_handler(args.input_filepath)
        tmp_dir = check_tmp_dir(args)

        if args.hg19 is not None:
            genome_reference_path_hg19 = args.hg19
            preprocessing_vcf_tokenizing(vcf_file=vcf_files,
                                    genome_reference_path=genome_reference_path_hg19,
                                    tmp_dir=tmp_dir,
                                    dict_motif=dict_motif,
                                    dict_pos=dict_pos,
                                    dict_ges=dict_ges)
            print('preprocessed data saved in ' + tmp_dir)
            predict_ready_files = []
            for x in vcf_files:
                if os.path.exists(tmp_dir + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz'):
                    predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

            pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])
        if args.hg38 is not None:
            genome_reference_path_hg38 = args.hg38
            preprocessing_vcf38_tokenizing(vcf_file=vcf_files,
                                    genome_reference_38_path=genome_reference_path_hg38,
                                    tmp_dir=tmp_dir,
                                    dict_motif=dict_motif,
                                    dict_pos=dict_pos,
                                    dict_ges=dict_ges)
            print('preprocessed data saved in ' + tmp_dir)

            predict_ready_files = []
            for x in vcf_files:
                if os.path.exists(tmp_dir + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz'):
                    predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

            pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])

        if args.no_preprocessing:
            predict_ready_files = vcf_files
            pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])
        
        target_handler = load_target_handler(checkpoint)

        dataloader_config = checkpoint['dataloader_config']
        #pdb.set_trace()
        test_dataloader = MuAtDataloader(pd_predict,dataloader_config)

        #pdb.set_trace()
        model_name = checkpoint['model_name']
        model = get_model(model_name,checkpoint['model_config'])
        model = model.to(device)
        model.load_state_dict(checkpoint['weight'])

        result_dir = ensure_dirpath(args.result_dir)
        predict_config = PredictorConfig(max_epochs=1, batch_size=1,result_dir=result_dir,target_handler=target_handler)
        predictor = Predictor(model, test_dataloader, predict_config)
        predictor.batch_predict()
        
    if args.command == 'preprocessing':

        genome_reference_path_hg19 = args.hg19
        genome_reference_path_hg38 = args.hg38

        tmp_dir = check_tmp_dir(args)
        #example for preprocessing multiple vcf files
        vcf_files = multifiles_handler(args.input_filepath)

        #pdb.set_trace()
        if args.motif_dictionary_filepath is None or args.position_dictionary_filepath is None or args.ges_dictionary_filepath is None:
            extdir = resource_filename('muat', 'extfile')
            extdir = ensure_dirpath(extdir)
            
            dict_motif = pd.read_csv(extdir + 'dictMutation.tsv',sep='\t')
            dict_pos = pd.read_csv(extdir + 'dictChpos.tsv',sep='\t')
            dict_ges = pd.read_csv(extdir + 'dictGES.tsv',sep='\t')
            print('using default dictionary in ' + extdir + 'dict{Mutation,Chpos,GES}.tsv')
        else:
            dict_motif = pd.read_csv(args.motif_dictionary_filepath,sep='\t')
            dict_pos = pd.read_csv(args.position_dictionary_filepath,sep='\t')
            dict_ges = pd.read_csv(args.ges_dictionary_filepath,sep='\t')

        if args.vcf:
            if args.hg19 is not None:
                preprocessing_vcf_tokenizing(vcf_file=vcf_files,
                                        genome_reference_path=genome_reference_path_hg19,
                                        tmp_dir=tmp_dir,
                                        dict_motif=dict_motif,
                                        dict_pos=dict_pos,
                                        dict_ges=dict_ges)
            if args.hg38 is not None:
                preprocessing_vcf38_tokenizing(vcf_file=vcf_files,
                                        genome_reference_38_path=genome_reference_path_hg38,
                                        tmp_dir=tmp_dir,
                                        dict_motif=dict_motif,
                                        dict_pos=dict_pos,
                                        dict_ges=dict_ges)

            print('preprocessed data saved in ' + tmp_dir)

        if args.tsv:
            print('todo')            
        if args.somagg:
            if args.hg19 is not None:
                print('todo')
            if args.hg38 is not None:
                tmp_dir = check_tmp_dir(args)
                
                filtering_somagg_vcf(vcf_files,tmp_dir)
                
                #pdb.set_trace()
                tsv_files = []
                for x in vcf_files:
                    sample_name = get_sample_name(x)
                    all_tsv = glob.glob(tmp_dir + '*' + sample_name +'*.tsv')
                    only_tsv = [x for x in all_tsv if x[-4:] == '.tsv'][0]
                    tsv_files.append(only_tsv)
                #pdb.set_trace()
                preprocessing_tsv38_tokenizing(tsv_files,genome_reference_path_hg38,tmp_dir,dict_motif,dict_pos,dict_ges)

    #pdb.set_trace()      
    if args.command == 'from-scratch': #training from scratch
        #pdb.set_trace()
        extdir = ensure_dirpath(resource_filename('muat', 'extfile'))
        motif_path = args.motif_dictionary_filepath or f"{extdir}/dictMutation.tsv"
        pos_path = args.position_dictionary_filepath or f"{extdir}/dictChpos.tsv"
        ges_path = args.ges_dictionary_filepath or f"{extdir}/dictGES.tsv"

        save_dir = ensure_dirpath(args.save_dir)
        os.makedirs(save_dir,exist_ok=True)

        if args.motif_dictionary_filepath is None or args.position_dictionary_filepath is None or args.ges_dictionary_filepath is None:
            warnings.warn(f"Dictionary file paths were not defined and have been set automatically:\n"
                        f"--motif-dictionary-filepath: {motif_path}\n"
                        f"--position-dictionary-filepath: {pos_path}\n"
                        f"--ges-dictionary-filepath: {ges_path}\n"
                        "These dictionaries might be different from your preprocessed files!")
        # Load mutation-related dictionaries
        dict_motif = pd.read_csv(motif_path, sep='\t')
        dict_pos = pd.read_csv(pos_path, sep='\t')
        dict_ges = pd.read_csv(ges_path, sep='\t')

        train_split, test_split = load_data(args.train_split_filepath, args.val_split_filepath)
        all_split = pd.concat([train_split, test_split], ignore_index=True)
        label_1 = all_split[['class_name','class_index']].drop_duplicates()
        label_1 = label_1.sort_values(by=['class_index']).reset_index(drop=True)   
        save_label_1 = save_dir+'/label_1.tsv'
        label_1.to_csv(save_label_1,sep='\t',index=False)
        
        label_2 = None
        save_label_2 = None
        if 'subclass_name' in all_split.columns:
            label_2 = all_split[['subclass_name','subclass_index']].drop_duplicates()
            label_2 = label_2.sort_values(by=['subclass_index']).reset_index(drop=True) 
            save_label_2 = save_dir+'/label_2.tsv'
            label_2.to_csv(save_label_2,sep='\t',index=False)

        if label_2 is None:
            if args.use_motif and args.use_position is False and args.use_ges is False:
                arch = 'MuAtMotifF'
            if args.use_motif and args.use_position and args.use_ges is False:
                arch = 'MuAtMotifPositionF'
            if args.use_motif and args.use_position and args.use_ges:
                arch = 'MuAtMotifPositionGESF'
        else:
            if args.use_motif and args.use_position is False and args.use_ges is False:
                arch = 'MuAtMotifF_2Labels'
            if args.use_motif and args.use_position and args.use_ges is False:
                arch = 'MuAtMotifPositionF_2Labels'
            if args.use_motif and args.use_position and args.use_ges:
                arch = 'MuAtMotifPositionGESF_2Labels'

        # Define model configuration
        model_config = ModelConfig(
            model_name=arch,
            dict_motif=dict_motif,
            dict_pos=dict_pos,
            dict_ges=dict_ges,
            mutation_sampling_size=args.mutation_sampling_size,
            n_layer=args.n_layer,
            n_emb=args.n_emb,
            n_head=args.n_head,
            n_class=None,  # Will be set after label encoding
            mutation_type=args.mutation_type
        )    

        # Define trainer configuration
        trainer_config = TrainerConfig(
            max_epochs=args.epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=1,
            save_ckpt_dir=save_dir,
            target_handler=[]
        )
        # Initialize label encoders BEFORE creating the model
        target_handler, n_class, n_subclass = initialize_label_encoders(save_label_1, save_label_2)
        model_config.num_class = n_class
        if n_subclass is not None:
            model_config.num_subclass = n_subclass

        trainer_config.target_handler = target_handler

        model = get_model(arch, model_config)

        # Define dataloader configurations
        train_dataloader_config = DataloaderConfig(
            model_input=model_config.model_input,
            mutation_type_ratio=model_config.mutation_type_ratio,
            mutation_sampling_size=args.mutation_sampling_size
        )
        test_dataloader_config = DataloaderConfig(
            model_input=model_config.model_input,
            mutation_type_ratio=model_config.mutation_type_ratio,
            mutation_sampling_size=args.mutation_sampling_size
        )

        # Initialize dataloaders
        train_dataloader = MuAtDataloader(train_split, train_dataloader_config)
        test_dataloader = MuAtDataloader(test_split, test_dataloader_config)

        # Initialize trainer and start training
        trainer = Trainer(model, train_dataloader, test_dataloader, trainer_config)
        trainer.batch_train()

    if args.command == 'from-checkpoint':

        save_dir = ensure_dirpath(args.save_dir)
        load_ckpt_filepath = os.path.normpath(args.ckpt_filepath)
        # Load checkpoint and extract stored configurations
        checkpoint = load_and_check_checkpoint(load_ckpt_filepath)

        #pdb.set_trace()
        model_config_muttype = checkpoint['model_config'].mutation_type
        if model_config_muttype != args.mutation_type:
            raise ValueError('You selected a different mutation type from the checkpoint. The checkpoint mutation type is ' + model_config_muttype + ', which is different from --mutation-type ' + args.mutation_type + '. Please select the correct checkpoint.')

        dataloader_config = checkpoint['dataloader_config']
        model_config = checkpoint['model_config']
        trainer_config = checkpoint['trainer_config']
        trainer_config.save_ckpt_dir = save_dir
        os.makedirs(save_dir,exist_ok=True)

        train_split, test_split = load_data(args.train_split_filepath, args.val_split_filepath)
        all_split = pd.concat([train_split, test_split], ignore_index=True)
        label_1 = all_split[['class_name','class_index']].drop_duplicates()
        label_1 = label_1.sort_values(by=['class_index']).reset_index(drop=True)   
        save_label_1 = save_dir+'/label_1.tsv'
        label_1.to_csv(save_label_1,sep='\t',index=False)
        
        label_2 = None
        save_label_2 = None
        if 'subclass_name' in all_split.columns:
            label_2 = all_split[['subclass_name','subclass_index']].drop_duplicates()
            label_2 = label_2.sort_values(by=['subclass_index']).reset_index(drop=True) 
            save_label_2 = save_dir+'/label_2.tsv'
            label_2.to_csv(save_label_2,sep='\t',index=False)

        #pdb.set_trace()
        if label_2 is None:
            arch = checkpoint['model_name']
        else:
            if checkpoint['model_name'] == 'MuAtMotifF' or checkpoint['model_name'] == 'MuAtMotif':
                arch = 'MuAtMotifF_2Labels'
            if checkpoint['model_name'] == 'MuAtMotifPositionF' or checkpoint['model_name'] == 'MuAtMotifPosition':
                arch = 'MuAtMotifPositionF_2Labels'
            if checkpoint['model_name'] == 'MuAtMotifPositionGESF' or checkpoint['model_name'] == 'MuAtMotifPositionGES':
                arch = 'MuAtMotifPositionGESF_2Labels'

        # Initialize label encoders BEFORE creating the model
        target_handler, n_class, n_subclass = initialize_label_encoders(save_label_1, save_label_2)
        model_config.num_class = n_class
        if n_subclass is not None:
            model_config.num_subclass = n_subclass

        trainer_config.target_handler = target_handler

        # Initialize model (whether from scratch or checkpoint)
        model = get_model(arch, model_config)

        print('initializing checkpoint parameters')
        #pdb.set_trace()
        '''
        'mutation_sampling_size': 5000, 
        'n_layer': 1, 'n_embd': 512, 
        'n_head': 2, 'num_class': 14, 
        'mutation_type': 'snv+mnv', 
        'model_input': {'motif': True, 'pos': True, 'ges': False}, 
        'position_size': 2916,
        'ges_size': 16, 
        'mutation_type_ratio': {'snv': 0.5, 'mnv': 0.5, 'indel': 0, 'sv_mei': 0, 'neg': 0}, 
        'motif_size': 2267, 
        'num_subclass': 12}
        '''
        model = initialize_pretrained_weight(arch,model_config,checkpoint)

        # Prepare dataloaders
        train_split = pd.read_csv(args.train_split_filepath, sep='\t', low_memory=False)
        test_split = pd.read_csv(args.val_split_filepath, sep='\t', low_memory=False)

        # Define dataloader configurations
        train_dataloader_config = DataloaderConfig(
            model_input=model_config.model_input,
            mutation_type_ratio=model_config.mutation_type_ratio,
            mutation_sampling_size=args.mutation_sampling_size
        )
        test_dataloader_config = DataloaderConfig(
            model_input=model_config.model_input,
            mutation_type_ratio=model_config.mutation_type_ratio,
            mutation_sampling_size=args.mutation_sampling_size
        )

        # Initialize dataloaders
        train_dataloader = MuAtDataloader(train_split, train_dataloader_config)
        test_dataloader = MuAtDataloader(test_split, test_dataloader_config)

        # Initialize trainer and start training
        trainer = Trainer(model, train_dataloader, test_dataloader, trainer_config)
        trainer.batch_train()


    if args.command == 'muat-wgs' or args.command=='muat-wes':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        genome_reference_path_hg19 = args.hg19
        genome_reference_path_hg38 = args.hg38

        #benchmark_ckpt = resource_filename('muat', 'pkg_ckpt')
        if args.command == 'muat-wgs':
            benchmark_ckpt = resource_filename('muat', 'pkg_ckpt')
            benchmark_ckpt = ensure_dirpath(benchmark_ckpt) + 'benchmark_wgs/'
            url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/benchmark_wgs.zip"

        if args.command == 'muat-wes':
            benchmark_ckpt = resource_filename('muat', 'pkg_ckpt')
            benchmark_ckpt = ensure_dirpath(benchmark_ckpt) + 'benchmark_wes/'
            url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/benchmark_wes.zip"

        #pdb.set_trace()
        check_pth = glob.glob(benchmark_ckpt + args.mutation_type + '/*.pthx')
        if len(check_pth)==0:
            download_checkpoint(url,'my_checkpoint.zip')
            check_pth = glob.glob(benchmark_ckpt + args.mutation_type + '/*.pthx')

        if len(check_pth) == 0:
            raise ValueError('cant find benchmark model in ' + benchmark_ckpt + args.mutation_type + '. Download benchmark model from ' + url + ' and extract to this path.')

        print('running prediction of ensemble models')
        #pdb.set_trace()
        for i_fold in range(len(check_pth)):
            pth_file = check_pth[i_fold]

            fold = pth_file.split('fold')[-1].split('.pthx')[0]
            load_ckpt_path = pth_file
            print('prediction from {}'.format(pth_file))
            #pdb.set_trace()
            checkpoint = load_and_check_checkpoint(load_ckpt_path)
            model_name = checkpoint['model_name']

            dict_motif,dict_pos,dict_ges = load_token_dict(checkpoint)
            vcf_files = multifiles_handler(args.input_filepath)
            tmp_dir = check_tmp_dir(args)

            if args.hg19 is not None:
                if i_fold == 0:
                    genome_reference_path_hg19 = args.hg19
                    preprocessing_vcf_tokenizing(vcf_file=vcf_files,
                                            genome_reference_path=genome_reference_path_hg19,
                                            tmp_dir=tmp_dir,
                                            dict_motif=dict_motif,
                                            dict_pos=dict_pos,
                                            dict_ges=dict_ges)
                    print('preprocessed data saved in ' + tmp_dir)
                predict_ready_files = []
                for x in vcf_files:
                    if os.path.exists(tmp_dir + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz'):
                        predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

                pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])
            if args.hg38 is not None:
                genome_reference_path_hg38 = args.hg38
                preprocessing_vcf38_tokenizing(vcf_file=vcf_files,
                                        genome_reference_38_path=genome_reference_path_hg38,
                                        tmp_dir=tmp_dir,
                                        dict_motif=dict_motif,
                                        dict_pos=dict_pos,
                                        dict_ges=dict_ges)
                print('preprocessed data saved in ' + tmp_dir)

                predict_ready_files = []
                for x in vcf_files:
                    if os.path.exists(tmp_dir + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz'):
                        predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

                pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])

            if args.no_preprocessing:
                if i_fold == 0:
                    predict_ready_files = vcf_files
                    pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])
            
            target_handler = load_target_handler(checkpoint)

            dataloader_config = checkpoint['dataloader_config']
            #pdb.set_trace()
            test_dataloader = MuAtDataloader(pd_predict,dataloader_config)

            #pdb.set_trace()
            model_name = checkpoint['model_name']
            model = get_model(model_name,checkpoint['model_config'])
            model = model.to(device)
            model.load_state_dict(checkpoint['weight'])

            result_dir = ensure_dirpath(args.result_dir)
            predict_config = PredictorConfig(max_epochs=1, batch_size=1,result_dir=result_dir,target_handler=target_handler)
            predict_config.prefix = 'fold' + str(fold) + '_'
            predictor = Predictor(model, test_dataloader, predict_config)
            predictor.batch_predict()

        all_fold = glob.glob(result_dir + 'fold*')
        pd_allfold = pd.DataFrame()
        for i_f in all_fold:
            pd_perfold = pd.read_csv(i_f,sep='\t',low_memory=False)
            fold = i_f.split('fold')[1].split('_')[0]
            pd_perfold['fold'] = fold
            pd_allfold = pd.concat([pd_allfold,pd_perfold])
            os.remove(i_f)
        #pdb.set_trace()
        pd_logits = pd_allfold.drop(columns=['prediction'])

        all_samples = pd_logits['sample'].unique()
        pd_mean = pd.DataFrame()
        for x in all_samples:
            pd_persamp = pd_logits.loc[pd_logits['sample']==x]
            pd_logit = pd_persamp.drop(columns=['fold'])
            samp_mean = pd_logit.groupby(['sample']).mean()
            samp_mean = samp_mean.round(4)
            samp_mean['prediction'] = samp_mean.idxmax(axis='columns').values[0]
            samp_mean = samp_mean.reset_index()
            pd_mean = pd.concat([pd_mean,samp_mean],ignore_index=True)
        pd_mean.to_csv(result_dir + 'ensemble_prediction.tsv',sep='\t',float_format='%.4f',index=False)
            
'''
def main_old():
    args = get_main_args()

    if args.predict_vcf_hg19:

        if not args.hg19_filepath or not args.load_ckpt_filepath or not args.vcf_hg19_filepath or not args.result_dir:
            raise ValueError('--predict-vcf-hg19 requires --load-ckpt-filepath --vcf-hg19-filepath --result-dir --hg19-filepath')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        genome_reference_path = args.hg19_filepath
        #load ckpt
        load_ckpt_path = args.load_ckpt_filepath
        checkpoint = load_and_check_checkpoint(load_ckpt_path)

        model_name = checkpoint['model_name']

        #pdb.set_trace()

        dict_motif,dict_pos,dict_ges = load_token_dict(checkpoint)

        #example for preprocessing multiple vcf files
        #pdb.set_trace()
        vcf_files = pd.read_csv(args.vcf_hg19_filepath,sep='\t')['vcf_hg19_path'].to_list()
        vcf_files = multifiles_handler(vcf_files)

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
            if os.path.exists(tmp_dir + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz'):
                predict_ready_files.append(tmp_dir + '/' + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

        pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])
        target_handler = load_target_handler(checkpoint)

        dataloader_config = checkpoint['dataloader_config']
        #pdb.set_trace()
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #load ckpt
        load_ckpt_path = args.load_ckpt_filepath
        checkpoint = load_and_check_checkpoint(load_ckpt_path)
        tmp_dir = check_tmp_dir(args)
        tmp_dir = ensure_dirpath(tmp_dir)

        genome_reference_38_path = args.hg38_filepath
        genome_reference_19_path = args.hg19_filepath

        dict_motif,dict_pos,dict_ges = load_token_dict(checkpoint)
        #example for preprocessing multiple vcf files
        vcf_files = pd.read_csv(args.vcf_hg38_filepath,sep='\t')['vcf_hg38_path'].to_list()
        vcf_files = multifiles_handler(vcf_files)
        
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
            if os.path.exists(tmp_dir  + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz'):
                predict_ready_files.append(tmp_dir + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

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
        vcf_files = pd.read_csv(args.vcf_hg19_filepath,sep='\t')['vcf_hg19_path'].to_list()
        vcf_files = multifiles_handler(vcf_files)

        preprocessing_vcf(vcf_file=vcf_files,genome_reference_path=genome_reference_path,tmp_dir=tmp_dir)

    if args.preprocessing_vcf_hg38:
        if not args.hg19_filepath or not args.vcf_hg38_filepath or not args.hg38_filepath:
            raise ValueError('--preprocessing-vcf-hg38 requires --hg38-filepath --hg19-filepath --vcf-hg38-filepath')
        tmp_dir = check_tmp_dir(args)
        genome_reference_38_path = args.hg38_filepath
        genome_reference_19_path = args.hg19_filepath

        #example for preprocessing multiple vcf files
        vcf_files = pd.read_csv(args.vcf_hg38_filepath,sep='\t')['vcf_hg38_path'].to_list()
        vcf_files = multifiles_handler(vcf_files)
        #run preprocessing 
        preprocessing_vcf38(vcf_files,genome_reference_38_path,genome_reference_19_path,tmp_dir)

    if args.tokenizing:

        if not args.motif_dictionary_filepath or not args.position_dictionary_filepath or not args.ges_dictionary_filepath:
            extdir = resource_filename('muat', 'extfile')
            extdir = ensure_dirpath(extdir)
            
            dict_motif = pd.read_csv(extdir + 'dictMutation.tsv',sep='\t')
            dict_pos = pd.read_csv(extdir + 'dictChpos.tsv',sep='\t')
            dict_ges = pd.read_csv(extdir + 'dictGES.tsv',sep='\t')

            print('using default dictionary in ' + extdir + 'dict{Mutation,Chpos,GES}.tsv')

        else:
            dict_motif = pd.read_csv(args.motif_dictionary_filepath,sep='\t')
            dict_pos = pd.read_csv(args.position_dictionary_filepath,sep='\t')
            dict_ges = pd.read_csv(args.ges_dictionary_filepath,sep='\t')

        tmp_dir = check_tmp_dir(args)
        all_preprocessed_vcf = glob.glob(ensure_dirpath(tmp_dir) + '/*.gc.genic.exonic.cs.tsv.gz')
        all_preprocessed_vcf = multifiles_handler(all_preprocessed_vcf)

        tokenizing(dict_motif, dict_pos, dict_ges,all_preprocessed_vcf,pos_bin_size=1000000,tmp_dir=tmp_dir)

    if args.train:
       # Load train and test datasets
        train_split, test_split = load_data(args.train_split_filepath, args.val_split_filepath)

        if args.from_checkpoint:
            # Load checkpoint and extract stored configurations
            checkpoint = load_and_check_checkpoint(args.load_ckpt_filepath)
            dataloader_config = checkpoint['dataloader_config']
            model_config = checkpoint['model_config']
            trainer_config = checkpoint['trainer_config']
            trainer_config.save_ckpt_dir = ensure_dirpath(args.save_ckpt_dir)

        elif args.from_scratch:
            # Define dictionary file paths (use defaults if not provided)
            extdir = ensure_dirpath(resource_filename('muat', 'extfile'))
            motif_path = args.motif_dictionary_filepath or f"{extdir}/dictMutation.tsv"
            pos_path = args.position_dictionary_filepath or f"{extdir}/dictChpos.tsv"
            ges_path = args.ges_dictionary_filepath or f"{extdir}/dictGES.tsv"

            if not args.motif_dictionary_filepath or not args.position_dictionary_filepath or not args.ges_dictionary_filepath:
                warnings.warn(f"Dictionary file paths were not defined and have been set automatically:\n"
                            f"--motif-dictionary-filepath: {motif_path}\n"
                            f"--position-dictionary-filepath: {pos_path}\n"
                            f"--ges-dictionary-filepath: {ges_path}\n"
                            "These dictionaries might be different from your preprocessed files!")

            # Load mutation-related dictionaries
            dict_motif = pd.read_csv(motif_path, sep='\t')
            dict_pos = pd.read_csv(pos_path, sep='\t')
            dict_ges = pd.read_csv(ges_path, sep='\t')

            # Ensure all necessary training arguments are provided
            if not all([args.arch, args.mutation_type, args.n_layer, args.n_emb, args.n_head, args.target_dict_filepath]):
                raise ValueError("--train requires --arch, --mutation-type, --n-layer, --n-emb, --n-head, and --target-dict-filepath.")

            # Define model configuration
            model_config = ModelConfig(
                model_name=args.arch,
                dict_motif=dict_motif,
                dict_pos=dict_pos,
                dict_ges=dict_ges,
                mutation_sampling_size=args.mutation_sampling_size,
                n_layer=args.n_layer,
                n_emb=args.n_emb,
                n_head=args.n_head,
                n_class=None,  # Will be set after label encoding
                mutation_type=args.mutation_type
            )

            # Define trainer configuration
            trainer_config = TrainerConfig(
                max_epochs=args.epoch,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_workers=1,
                save_ckpt_dir=args.save_ckpt_dir,
                target_handler=[]
            )

        # Initialize label encoders BEFORE creating the model
        target_handler, n_class, n_subclass = initialize_label_encoders(args.target_dict_filepath, args.subtarget_dict_filepath)
        model_config.num_class = n_class
        if n_subclass is not None:
            model_config.num_subclass = n_subclass

        trainer_config.target_handler = target_handler

        # Initialize model (whether from scratch or checkpoint)
        model = get_model(args.arch, model_config)
        #pdb.set_trace()

        if args.from_checkpoint:
            print('initializing checkpoint parameters to ' + args.arch)
            model = initialize_pretrained_weight(args.arch,model_config,checkpoint)

        # Prepare dataloaders
        train_split = pd.read_csv(args.train_split_filepath, sep='\t', low_memory=False)
        test_split = pd.read_csv(args.val_split_filepath, sep='\t', low_memory=False)

        # Define dataloader configurations
        train_dataloader_config = DataloaderConfig(
            model_input=model_config.model_input,
            mutation_type_ratio=model_config.mutation_type_ratio,
            mutation_sampling_size=args.mutation_sampling_size
        )
        test_dataloader_config = DataloaderConfig(
            model_input=model_config.model_input,
            mutation_type_ratio=model_config.mutation_type_ratio,
            mutation_sampling_size=args.mutation_sampling_size
        )

        # Initialize dataloaders
        train_dataloader = MuAtDataloader(train_split, train_dataloader_config)
        test_dataloader = MuAtDataloader(test_split, test_dataloader_config)

        # Initialize trainer and start training
        trainer = Trainer(model, train_dataloader, test_dataloader, trainer_config)
        trainer.batch_train()
'''

if __name__ == "__main__":

    main()
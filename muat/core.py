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

            '''
            # Validate trainer configuration to ensure all necessary hyperparameters are present
            required_params = ['batch_size', 'learning_rate']
            for param in required_params:
                if not hasattr(trainer_config, param) or trainer_config.__dict__[param] is None:
                    raise ValueError(f"Checkpoint trainer configuration is missing '{param}'.")
            

            # Override checkpoint hyperparameters with command-line args if provided
            trainer_config.max_epochs = args.epoch if args.epoch is not None else trainer_config.max_epochs
            trainer_config.batch_size = args.batch_size if args.batch_size is not None else trainer_config.batch_size
            trainer_config.learning_rate = args.learning_rate if args.learning_rate is not None else trainer_config.learning_rate

            trainer_config.save_ckpt_dir = ensure_dirpath(args.save_ckpt_dir)

            # Initialize model from checkpoint

            pdb.set_trace()
            model = get_model(args.arch, model_config)
            model = initialize_pretrained_weight(args.arch, model_config, checkpoint)
            '''

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

if __name__ == "__main__":

    main()
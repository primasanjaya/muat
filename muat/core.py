import sys
import os
import tarfile
import zipfile
import glob
import warnings

import pandas as pd
import numpy as np
import torch

from pkg_resources import resource_filename

from muat.download import *
from muat.preprocessing import *
from muat.util import *
from muat.dataloader import *
from muat.trainer import *
from muat.predict import *
from muat.model import *
from muat.checkpoint import *

def attach_label_indices(train_split, test_split, label_1, label_2=None):
    class_map = label_1.set_index('class_name')['class_index']
    train_split['class_index'] = train_split['class_name'].map(class_map)
    test_split['class_index'] = test_split['class_name'].map(class_map)

    if train_split['class_index'].isna().any():
        missing = train_split.loc[train_split['class_index'].isna(), 'class_name'].unique()
        raise ValueError(f"Train split contains class_name not found in label_1: {missing}")
    if test_split['class_index'].isna().any():
        missing = test_split.loc[test_split['class_index'].isna(), 'class_name'].unique()
        raise ValueError(f"Validation split contains class_name not found in label_1: {missing}")

    train_split['class_index'] = train_split['class_index'].astype(int)
    test_split['class_index'] = test_split['class_index'].astype(int)

    if label_2 is not None:
        subclass_map = label_2.set_index('subclass_name')['subclass_index']
        train_split['subclass_index'] = train_split['subclass_name'].map(subclass_map)
        test_split['subclass_index'] = test_split['subclass_name'].map(subclass_map)

        if train_split['subclass_index'].isna().any():
            missing = train_split.loc[train_split['subclass_index'].isna(), 'subclass_name'].unique()
            raise ValueError(f"Train split contains subclass_name not found in label_2: {missing}")
        if test_split['subclass_index'].isna().any():
            missing = test_split.loc[test_split['subclass_index'].isna(), 'subclass_name'].unique()
            raise ValueError(f"Validation split contains subclass_name not found in label_2: {missing}")

        train_split['subclass_index'] = train_split['subclass_index'].astype(int)
        test_split['subclass_index'] = test_split['subclass_index'].astype(int)

    return train_split, test_split


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
        raise ValueError(
            f"Checkpoint is missing required parameters: {', '.join(missing_params)}.\n"
            "Please provide these values via command-line arguments (--epoch, --batch-size, --learning-rate)."
        )


def load_data(train_path, val_path):
    train_split = pd.read_csv(train_path, sep='\t', low_memory=False)
    test_split = pd.read_csv(val_path, sep='\t', low_memory=False)
    return train_split, test_split


def initialize_label_encoders(target_path, subtarget_path=None):
    target_handler = [
        LabelEncoderFromCSV(
            csv_file=target_path,
            class_name_col='class_name',
            class_index_col='class_index'
        )
    ]
    n_class = len(target_handler[0].classes_)

    n_subclass = None
    if subtarget_path is not None:
        le2 = LabelEncoderFromCSV(
            csv_file=subtarget_path,
            class_name_col='subclass_name',
            class_index_col='subclass_index'
        )
        target_handler.append(le2)
        n_subclass = len(le2.classes_)

    return target_handler, n_class, n_subclass


def unziping_from_package_installation():
    pkg_ckpt = resource_filename('muat', 'pkg_ckpt')
    pkg_ckpt = ensure_dirpath(pkg_ckpt)

    all_zip = glob.glob(os.path.join(pkg_ckpt, '*.zip'))
    if len(all_zip) > 0:
        for checkpoint_file in all_zip:
            with zipfile.ZipFile(checkpoint_file, 'r') as zip_ref:
                zip_ref.extractall(path=pkg_ckpt)
            os.remove(checkpoint_file)


def _collect_preprocessed_files(vcf_files, tmp_dir):
    predict_ready_files = []
    for x in vcf_files:
        candidate = os.path.join(tmp_dir, get_sample_name(x) + '.muat.tsv')
        if os.path.exists(candidate):
            predict_ready_files.append(candidate)
    return predict_ready_files


def _run_predict(args, device):
    """Shared logic for muat predict {pretrained|from-checkpoint}."""

    if args.source == 'pretrained':
        if args.assay == 'wgs':
            benchmark_ckpt = os.path.join(ensure_dirpath(resource_filename('muat', 'pkg_ckpt')), 'pcawg_wgs')
            url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/best_wgs_pcawg.zip"
        else:
            benchmark_ckpt = os.path.join(ensure_dirpath(resource_filename('muat', 'pkg_ckpt')), 'tcga_wes')
            url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/best_wes_tcga.zip"

        check_pth = glob.glob(os.path.join(benchmark_ckpt, args.mutation_type, '*.pthx'))
        if len(check_pth) == 0:
            print('cant find model in ' + os.path.join(benchmark_ckpt, args.mutation_type) + '. Downloading model from ' + url)
            download_checkpoint(url, 'my_checkpoint.zip')
            check_pth = glob.glob(os.path.join(benchmark_ckpt, args.mutation_type, '*.pthx'))

        if len(check_pth) == 0:
            raise ValueError(
                'cant find benchmark model in ' +
                os.path.join(benchmark_ckpt, args.mutation_type) +
                '. Download benchmark model from ' + url + ' and extract to this path.'
            )

        load_ckpt_path = mut_type_checkpoint_handler(args.mutation_type, args.assay)
    else:
        load_ckpt_path = resolve_path(args.ckpt_filepath)

    checkpoint = load_and_check_checkpoint(load_ckpt_path)
    dict_motif, dict_pos, dict_ges = load_token_dict(checkpoint)

    if getattr(args, 'input_list', None) is not None:
        with open(resolve_path(args.input_list)) as f:
            vcf_files = [line.strip() for line in f if line.strip()]
    else:
        vcf_files = multifiles_handler(args.input_filepath)

    if not args.needs_preprocessing:
        predict_ready_files = [resolve_path(x) for x in vcf_files]
        pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])
    else:
        tmp_dir = check_tmp_dir(args)
        if args.hg19 is not None:
            preprocessing_vcf_tokenizing(
                vcf_file=vcf_files,
                genome_reference_path=resolve_path(args.hg19),
                tmp_dir=tmp_dir,
                dict_motif=dict_motif,
                dict_pos=dict_pos,
                dict_ges=dict_ges
            )
        else:
            preprocessing_vcf38_tokenizing(
                vcf_file=vcf_files,
                genome_reference_38_path=resolve_path(args.hg38),
                tmp_dir=tmp_dir,
                dict_motif=dict_motif,
                dict_pos=dict_pos,
                dict_ges=dict_ges
            )
        print('preprocessed data saved in ' + tmp_dir)
        predict_ready_files = _collect_preprocessed_files(vcf_files, tmp_dir)
        pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])

    target_handler = load_target_handler(checkpoint)
    dataloader_config = checkpoint['dataloader_config']
    test_dataloader = MuAtDataloader(pd_predict, dataloader_config)

    model_name = checkpoint['model_name']
    model = get_model(model_name, checkpoint['model_config'])
    model = model.to(device)
    model.load_state_dict(checkpoint['weight'])

    result_dir = ensure_dirpath(resolve_path(args.result_dir))
    predict_config = PredictorConfig(
        max_epochs=1,
        batch_size=1,
        result_dir=result_dir,
        target_handler=target_handler
    )
    predictor = Predictor(model, test_dataloader, predict_config)
    predictor.batch_predict()


def _run_predict_ensemble(args, device):
    """Shared logic for muat predict-ensemble {pretrained|from-checkpoint}."""

    if args.source == 'from-checkpoint':
        check_pth = [resolve_path(p) for p in args.ckpt_filepath]
        missing = [p for p in check_pth if not os.path.exists(p)]
        if missing:
            raise ValueError('checkpoint(s) not found: ' + ', '.join(missing))
    else:
        if args.assay == 'wgs':
            benchmark_ckpt = os.path.join(ensure_dirpath(resource_filename('muat', 'pkg_ckpt')), 'benchmark_wgs')
            url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/benchmark_wgs.zip"
        else:
            benchmark_ckpt = os.path.join(ensure_dirpath(resource_filename('muat', 'pkg_ckpt')), 'benchmark_wes')
            url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/benchmark_wes.zip"

        check_pth = glob.glob(os.path.join(benchmark_ckpt, args.mutation_type, '*.pthx'))
        if len(check_pth) == 0:
            download_checkpoint(url, 'my_checkpoint.zip')
            check_pth = glob.glob(os.path.join(benchmark_ckpt, args.mutation_type, '*.pthx'))

        if len(check_pth) == 0:
            raise ValueError(
                'cant find benchmark model in ' +
                os.path.join(benchmark_ckpt, args.mutation_type) +
                '. Download benchmark model from ' + url + ' and extract to this path.'
            )

    print('running prediction of ensemble models')

    result_dir = ensure_dirpath(resolve_path(args.result_dir))

    if getattr(args, 'input_list', None) is not None:
        with open(resolve_path(args.input_list)) as f:
            vcf_files = [line.strip() for line in f if line.strip()]
    else:
        vcf_files = multifiles_handler(args.input_filepath)

    tmp_dir = check_tmp_dir(args) if args.needs_preprocessing else None

    for i_fold, pth_file in enumerate(check_pth):
        if args.source == 'from-checkpoint':
            fold = str(i_fold)
        else:
            fold = pth_file.split('fold')[-1].split('.pthx')[0]
        print('prediction from {}'.format(pth_file))

        checkpoint = load_and_check_checkpoint(pth_file)
        dict_motif, dict_pos, dict_ges = load_token_dict(checkpoint)

        if not args.needs_preprocessing:
            if i_fold == 0:
                predict_ready_files = [resolve_path(x) for x in vcf_files]
                pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])
        else:
            if i_fold == 0:
                if args.hg19 is not None:
                    preprocessing_vcf_tokenizing(
                        vcf_file=vcf_files,
                        genome_reference_path=resolve_path(args.hg19),
                        tmp_dir=tmp_dir,
                        dict_motif=dict_motif,
                        dict_pos=dict_pos,
                        dict_ges=dict_ges
                    )
                else:
                    preprocessing_vcf38_tokenizing(
                        vcf_file=vcf_files,
                        genome_reference_38_path=resolve_path(args.hg38),
                        tmp_dir=tmp_dir,
                        dict_motif=dict_motif,
                        dict_pos=dict_pos,
                        dict_ges=dict_ges
                    )
                print('preprocessed data saved in ' + tmp_dir)
            predict_ready_files = _collect_preprocessed_files(vcf_files, tmp_dir)
            pd_predict = pd.DataFrame(predict_ready_files, columns=['prep_path'])

        target_handler = load_target_handler(checkpoint)
        dataloader_config = checkpoint['dataloader_config']
        test_dataloader = MuAtDataloader(pd_predict, dataloader_config)

        model_name = checkpoint['model_name']
        model = get_model(model_name, checkpoint['model_config'])
        model = model.to(device)
        model.load_state_dict(checkpoint['weight'])

        predict_config = PredictorConfig(
            max_epochs=1,
            batch_size=1,
            result_dir=result_dir,
            target_handler=target_handler
        )
        predict_config.prefix = 'fold' + str(fold) + '_'
        predictor = Predictor(model, test_dataloader, predict_config)
        predictor.batch_predict()

    all_fold = glob.glob(os.path.join(result_dir, 'fold*'))
    pd_allfold = pd.DataFrame()

    for i_f in all_fold:
        pd_perfold = pd.read_csv(i_f, sep='\t', low_memory=False)
        fold = i_f.split('fold')[1].split('_')[0]
        pd_perfold['fold'] = fold
        pd_allfold = pd.concat([pd_allfold, pd_perfold], ignore_index=True)
        os.remove(i_f)

    pd_logits = pd_allfold.drop(columns=['prediction'])
    all_samples = pd_logits['sample'].unique()
    pd_mean = pd.DataFrame()

    for x in all_samples:
        pd_persamp = pd_logits.loc[pd_logits['sample'] == x]
        pd_logit = pd_persamp.drop(columns=['fold'])
        samp_mean = pd_logit.groupby(['sample']).mean()
        samp_mean = samp_mean.round(4)
        samp_mean['prediction'] = samp_mean.idxmax(axis='columns').values[0]
        samp_mean = samp_mean.reset_index()
        pd_mean = pd.concat([pd_mean, samp_mean], ignore_index=True)

    pd_mean.to_csv(
        os.path.join(result_dir, 'ensemble_prediction.tsv'),
        sep='\t',
        float_format='%.4f',
        index=False
    )
    print('ensemble prediction saved to ' + os.path.join(result_dir, 'ensemble_prediction.tsv'))


def main():
    args = get_main_args()

    genomedir = resource_filename('muat', 'genome_reference')
    genomedir = ensure_dirpath(genomedir)

    pkg_ckpt = resource_filename('muat', 'pkg_ckpt')
    pkg_ckpt = ensure_dirpath(pkg_ckpt)

    if args.command == 'predict-ensemble' and args.source == 'pretrained':
        unziping_from_package_installation()

    if args.command == 'download':
        files_to_download = [
            'PCAWG/consensus_snv_indel/README.md',
            'PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public.tgz',
            'PCAWG/consensus_sv/README.md',
            'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.icgc.public.tgz',
            'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.tcga.public.tgz',
            'PCAWG/data_releases/latest/pcawg_sample_sheet.v1.4.2016-09-14.tsv',
            'PCAWG/data_releases/latest/release_may2016.v1.4.tsv',
            'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-08-12.tsv',
            'PCAWG/clinical_and_histology/pcawg_specimen_histology_August2016_v9.xlsx'
        ]
        download_data_path = resolve_path(args.download_dir)
        download_icgc_object_storage(data_path=download_data_path, files_to_download=files_to_download)
        print("Download completed. Data saved in " + str(download_data_path))
        
    if args.command == 'predict':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _run_predict(args, device)

    elif args.command == 'predict-ensemble':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _run_predict_ensemble(args, device)

    elif args.command == 'preprocess':
        tmp_dir = check_tmp_dir(args)
        if getattr(args, 'input_list', None) is not None:
            with open(resolve_path(args.input_list)) as f:
                vcf_files = [line.strip() for line in f if line.strip()]
        else:
            vcf_files = multifiles_handler(args.input_filepath)
        vcf_files = [resolve_path(x) for x in vcf_files]

        if not args.annotated and args.hg19 is None and args.hg38 is None:
            raise ValueError("--hg19 or --hg38 is required unless --annotated is set.")

        if (
            args.motif_dictionary_filepath is None or
            args.position_dictionary_filepath is None or
            args.ges_dictionary_filepath is None
        ):
            extdir = ensure_dirpath(resource_filename('muat', 'extfile'))
            dict_motif = pd.read_csv(os.path.join(extdir, 'dictMutation.tsv'), sep='\t')
            dict_pos = pd.read_csv(os.path.join(extdir, 'dictChpos.tsv'), sep='\t')
            dict_ges = pd.read_csv(os.path.join(extdir, 'dictGES.tsv'), sep='\t')
            print('using default dictionary in ' + extdir + 'dict{Mutation,Chpos,GES}.tsv')
        else:
            dict_motif = pd.read_csv(resolve_path(args.motif_dictionary_filepath), sep='\t')
            dict_pos = pd.read_csv(resolve_path(args.position_dictionary_filepath), sep='\t')
            dict_ges = pd.read_csv(resolve_path(args.ges_dictionary_filepath), sep='\t')

        if args.annotated:
            preprocessing_annotated_tokenizing(
                input_files=vcf_files,
                tmp_dir=tmp_dir,
                dict_motif=dict_motif,
                dict_pos=dict_pos,
                dict_ges=dict_ges,
            )
            print('preprocessed data saved in ' + tmp_dir)

        elif args.vcf:
            if args.hg19 is not None:
                genome_reference_path_hg19 = resolve_path(args.hg19)
                preprocessing_vcf_tokenizing(
                    vcf_file=vcf_files,
                    genome_reference_path=genome_reference_path_hg19,
                    tmp_dir=tmp_dir,
                    dict_motif=dict_motif,
                    dict_pos=dict_pos,
                    dict_ges=dict_ges
                )
            elif args.hg38 is not None:
                genome_reference_path_hg38 = resolve_path(args.hg38)
                if args.liftover:
                    preprocessing_vcf38_tokenizing(
                        vcf_file=vcf_files,
                        genome_reference_38_path=genome_reference_path_hg38,
                        tmp_dir=tmp_dir,
                        dict_motif=dict_motif,
                        dict_pos=dict_pos,
                        dict_ges=dict_ges
                    )
                else:
                    preprocessing_vcf38_native_tokenizing(
                        vcf_file=vcf_files,
                        genome_reference_38_path=genome_reference_path_hg38,
                        tmp_dir=tmp_dir,
                        dict_motif=dict_motif,
                        dict_pos=dict_pos,
                        dict_ges=dict_ges
                    )
            else:
                raise ValueError("For VCF preprocessing, please provide either --hg19 or --hg38.")
            print('preprocessed data saved in ' + tmp_dir)

        elif args.tsv:
            if args.hg19 is not None:
                genome_reference_path_hg19 = resolve_path(args.hg19)
                preprocessing_tsv_tokenizing(
                    vcf_files,
                    genome_reference_path_hg19,
                    tmp_dir,
                    dict_motif,
                    dict_pos,
                    dict_ges
                )
            elif args.hg38 is not None:
                genome_reference_path_hg38 = resolve_path(args.hg38)
                if args.liftover:
                    preprocessing_tsv38_tokenizing(
                        vcf_files,
                        genome_reference_path_hg38,
                        tmp_dir,
                        dict_motif,
                        dict_pos,
                        dict_ges
                    )
                else:
                    preprocessing_tsv38_native_tokenizing(
                        vcf_files,
                        genome_reference_path_hg38,
                        tmp_dir,
                        dict_motif,
                        dict_pos,
                        dict_ges
                    )
            else:
                raise ValueError("For TSV preprocessing, please provide either --hg19 or --hg38.")

        elif args.somagg:
            if args.hg19 is not None:
                print('todo')
            elif args.hg38 is not None:
                genome_reference_path_hg38 = resolve_path(args.hg38)
                filtering_somagg_vcf(vcf_files, tmp_dir)

                tsv_files = []
                for x in vcf_files:
                    sample_name = get_sample_name(x)
                    all_tsv = glob.glob(os.path.join(tmp_dir, '*' + sample_name + '*.tsv'))
                    only_tsv = [y for y in all_tsv if y.endswith('.tsv')][0]
                    tsv_files.append(only_tsv)

                if args.liftover:
                    preprocessing_tsv38_tokenizing(
                        tsv_files,
                        genome_reference_path_hg38,
                        tmp_dir,
                        dict_motif,
                        dict_pos,
                        dict_ges
                    )
                else:
                    preprocessing_tsv38_native_tokenizing(
                        tsv_files,
                        genome_reference_path_hg38,
                        tmp_dir,
                        dict_motif,
                        dict_pos,
                        dict_ges
                    )
            else:
                raise ValueError("For somagg preprocessing, please provide --hg38 or implement --hg19 branch.")

    elif args.command == 'train':
        if args.subcommand == 'from-scratch':
            extdir = ensure_dirpath(resource_filename('muat', 'extfile'))

            motif_path = resolve_path(args.motif_dictionary_filepath) or f"{extdir}/dictMutation.tsv"
            pos_path = resolve_path(args.position_dictionary_filepath) or f"{extdir}/dictChpos.tsv"
            ges_path = resolve_path(args.ges_dictionary_filepath) or f"{extdir}/dictGES.tsv"

            save_dir = ensure_dirpath(resolve_path(args.save_dir))
            os.makedirs(save_dir, exist_ok=True)

            if (
                args.motif_dictionary_filepath is None or
                args.position_dictionary_filepath is None or
                args.ges_dictionary_filepath is None
            ):
                warnings.warn(
                    f"Dictionary file paths were not defined and have been set automatically:\n"
                    f"--motif-dictionary-filepath: {motif_path}\n"
                    f"--position-dictionary-filepath: {pos_path}\n"
                    f"--ges-dictionary-filepath: {ges_path}\n"
                    "These dictionaries might be different from your preprocessed files!"
                )

            dict_motif = pd.read_csv(motif_path, sep='\t')
            dict_pos = pd.read_csv(pos_path, sep='\t')
            dict_ges = pd.read_csv(ges_path, sep='\t')

            train_split, test_split = load_data(
                resolve_path(args.train_split_filepath),
                resolve_path(args.val_split_filepath)
            )

            all_split = pd.concat([train_split, test_split], ignore_index=True)
            columns = all_split.columns

            if 'class_index' in columns:
                label_1 = all_split[['class_name', 'class_index']].drop_duplicates()
                label_1 = label_1.sort_values(by=['class_index']).reset_index(drop=True)
            else:
                label_1 = all_split[['class_name']].drop_duplicates()
                label_1 = label_1.sort_values(by=['class_name']).reset_index(drop=True)
                label_1['class_index'] = np.arange(len(label_1))

            save_label_1 = os.path.join(save_dir, 'label_1.tsv')
            label_1.to_csv(save_label_1, sep='\t', index=False)

            label_2 = None
            save_label_2 = None
            if 'subclass_name' in columns:
                if 'subclass_index' in columns:
                    label_2 = all_split[['subclass_name', 'subclass_index']].drop_duplicates()
                    label_2 = label_2.sort_values(by=['subclass_index']).reset_index(drop=True)
                else:
                    label_2 = all_split[['subclass_name']].drop_duplicates()
                    label_2 = label_2.sort_values(by=['subclass_name']).reset_index(drop=True)
                    label_2['subclass_index'] = np.arange(len(label_2))

                save_label_2 = os.path.join(save_dir, 'label_2.tsv')
                label_2.to_csv(save_label_2, sep='\t', index=False)

            target_handler, n_class, n_subclass = initialize_label_encoders(save_label_1, save_label_2)
            train_split, test_split = attach_label_indices(train_split, test_split, label_1, label_2)

            if label_2 is None:
                if args.use_motif and not args.use_position and not args.use_ges:
                    arch = 'MuAtMotifF'
                elif args.use_motif and args.use_position and not args.use_ges:
                    arch = 'MuAtMotifPositionF'
                elif args.use_motif and args.use_position and args.use_ges:
                    arch = 'MuAtMotifPositionGESF'
                else:
                    raise ValueError("Invalid combination of motif/position/ges flags.")
            else:
                if args.use_motif and not args.use_position and not args.use_ges:
                    arch = 'MuAtMotifF_2Labels'
                elif args.use_motif and args.use_position and not args.use_ges:
                    arch = 'MuAtMotifPositionF_2Labels'
                elif args.use_motif and args.use_position and args.use_ges:
                    arch = 'MuAtMotifPositionGESF_2Labels'
                else:
                    raise ValueError("Invalid combination of motif/position/ges flags.")

            model_config = ModelConfig(
                model_name=arch,
                dict_motif=dict_motif,
                dict_pos=dict_pos,
                dict_ges=dict_ges,
                mutation_sampling_size=args.mutation_sampling_size,
                n_layer=args.n_layer,
                n_emb=args.n_emb,
                n_head=args.n_head,
                n_class=n_class,
                mutation_type=args.mutation_type,
                num_subclass=n_subclass
            )

            trainer_config = TrainerConfig(
                max_epochs=args.epoch,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_workers=1,
                save_ckpt_dir=save_dir,
                target_handler=target_handler
            )

            model = get_model(arch, model_config)

            train_dataloader_config = DataloaderConfig(
                model_input=model_config.model_input,
                mutation_type_ratio=model_config.mutation_type_ratio,
                mutation_sampling_size=args.mutation_sampling_size,
                sampling_replacement=args.sampling_replacement
            )
            test_dataloader_config = DataloaderConfig(
                model_input=model_config.model_input,
                mutation_type_ratio=model_config.mutation_type_ratio,
                mutation_sampling_size=args.mutation_sampling_size,
                sampling_replacement=args.sampling_replacement
            )

            train_dataloader = MuAtDataloader(train_split, train_dataloader_config)
            test_dataloader = MuAtDataloader(test_split, test_dataloader_config)

            trainer = Trainer(model, train_dataloader, test_dataloader, trainer_config)
            trainer.batch_train()

        elif args.subcommand == 'from-checkpoint':
            save_dir = ensure_dirpath(resolve_path(args.save_dir))
            os.makedirs(save_dir, exist_ok=True)

            load_ckpt_filepath = resolve_path(args.ckpt_filepath)
            checkpoint = load_and_check_checkpoint(load_ckpt_filepath)

            model_config = checkpoint['model_config']
            trainer_config = checkpoint['trainer_config']
            dataloader_config = checkpoint['dataloader_config']

            trainer_config.save_ckpt_dir = save_dir

            validate_checkpoint(trainer_config)

            train_split, test_split = load_data(
                resolve_path(args.train_split_filepath),
                resolve_path(args.val_split_filepath)
            )

            all_split = pd.concat([train_split, test_split], ignore_index=True)
            columns = all_split.columns

            if 'class_index' in columns:
                label_1 = all_split[['class_name', 'class_index']].drop_duplicates()
                label_1 = label_1.sort_values(by=['class_index']).reset_index(drop=True)
            else:
                label_1 = all_split[['class_name']].drop_duplicates()
                label_1 = label_1.sort_values(by=['class_name']).reset_index(drop=True)
                label_1['class_index'] = np.arange(len(label_1))

            save_label_1 = os.path.join(save_dir, 'label_1.tsv')
            label_1.to_csv(save_label_1, sep='\t', index=False)

            label_2 = None
            save_label_2 = None
            if 'subclass_name' in columns:
                if 'subclass_index' in columns:
                    label_2 = all_split[['subclass_name', 'subclass_index']].drop_duplicates()
                    label_2 = label_2.sort_values(by=['subclass_index']).reset_index(drop=True)
                else:
                    label_2 = all_split[['subclass_name']].drop_duplicates()
                    label_2 = label_2.sort_values(by=['subclass_name']).reset_index(drop=True)
                    label_2['subclass_index'] = np.arange(len(label_2))

                save_label_2 = os.path.join(save_dir, 'label_2.tsv')
                label_2.to_csv(save_label_2, sep='\t', index=False)

            if label_2 is None:
                arch = checkpoint['model_name']
            else:
                if checkpoint['model_name'] in ['MuAtMotifF', 'MuAtMotif']:
                    arch = 'MuAtMotifF_2Labels'
                elif checkpoint['model_name'] in ['MuAtMotifPositionF', 'MuAtMotifPosition']:
                    arch = 'MuAtMotifPositionF_2Labels'
                elif checkpoint['model_name'] in ['MuAtMotifPositionGESF', 'MuAtMotifPositionGES']:
                    arch = 'MuAtMotifPositionGESF_2Labels'
                else:
                    raise ValueError(f"Unsupported checkpoint model for 2-label conversion: {checkpoint['model_name']}")

            target_handler, n_class, n_subclass = initialize_label_encoders(save_label_1, save_label_2)
            train_split, test_split = attach_label_indices(train_split, test_split, label_1, label_2)

            model_config.num_class = n_class
            model_config.n_class = n_class
            if hasattr(model_config, 'n_embd'):
                model_config.n_emb = model_config.n_embd

            if n_subclass is not None:
                model_config.num_subclass = n_subclass

            trainer_config.target_handler = target_handler

            model = get_model(arch, model_config)
            model = initialize_pretrained_weight(arch, model_config, checkpoint)

            mutation_sampling_size = getattr(args, 'mutation_sampling_size', None)
            if mutation_sampling_size is None:
                mutation_sampling_size = model_config.mutation_sampling_size

            sampling_replacement = getattr(args, 'sampling_replacement', False)

            train_dataloader_config = DataloaderConfig(
                model_input=model_config.model_input,
                mutation_type_ratio=model_config.mutation_type_ratio,
                mutation_sampling_size=mutation_sampling_size,
                sampling_replacement=sampling_replacement
            )
            test_dataloader_config = DataloaderConfig(
                model_input=model_config.model_input,
                mutation_type_ratio=model_config.mutation_type_ratio,
                mutation_sampling_size=mutation_sampling_size,
                sampling_replacement=sampling_replacement
            )

            train_dataloader = MuAtDataloader(train_split, train_dataloader_config)
            test_dataloader = MuAtDataloader(test_split, test_dataloader_config)

            trainer = Trainer(model, train_dataloader, test_dataloader, trainer_config)
            trainer.batch_train()

        else:
            raise ValueError(f"Unknown train subcommand: {args.subcommand}")

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
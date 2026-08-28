import sys
import os
import re
import json
import tarfile
import zipfile
import glob
import warnings

import pandas as pd
import numpy as np
import torch

from muat._resources import pkg_path

from muat.download import *
from muat.preprocessing import *
from muat.util import *
from muat.dataloader import *
from muat.trainer import *
from muat.predict import *
from muat.model import *
from muat.checkpoint import *
from muat.reproduce import (
    load_experiments, get_recipe, list_tags, resolve_cache_dir,
    fetch_tag, ensure_assets_present, build_predict_namespace,
    build_train_namespace, _sha256,
)
from muat.seed import set_seed

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
    pkg_ckpt = pkg_path('pkg_ckpt')
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
            benchmark_ckpt = os.path.join(ensure_dirpath(pkg_path('pkg_ckpt')), 'pcawg_wgs')
            url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/best_wgs_pcawg.zip"
        else:
            benchmark_ckpt = os.path.join(ensure_dirpath(pkg_path('pkg_ckpt')), 'tcga_wes')
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
        elif getattr(checkpoint.get('dataloader_config'), 'genome_build_mode', 'hg19') == 'hg38_native':
            # Checkpoint's own position/GES dictionaries were built from native
            # hg38 coordinates (no liftover) -- tokenize raw hg38 input the same
            # way, or every position/GES lookup is against the wrong coordinate
            # system.
            preprocessing_vcf38_native_tokenizing(
                vcf_file=vcf_files,
                genome_reference_38_path=resolve_path(args.hg38),
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
        target_handler=target_handler,
        prerelu=not getattr(args, 'relu', False)
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
            benchmark_ckpt = os.path.join(ensure_dirpath(pkg_path('pkg_ckpt')), 'benchmark_wgs')
            url = "https://huggingface.co/primasanjaya/muat-checkpoint/resolve/main/benchmark_wgs.zip"
        else:
            benchmark_ckpt = os.path.join(ensure_dirpath(pkg_path('pkg_ckpt')), 'benchmark_wes')
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
            m = re.search(r'fold(\d+)', os.path.basename(pth_file))
            fold = m.group(1) if m else str(i_fold)
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
                elif getattr(checkpoint.get('dataloader_config'), 'genome_build_mode', 'hg19') == 'hg38_native':
                    # See _run_predict: checkpoints trained on native hg38
                    # dictionaries must be tokenized without a liftover step.
                    preprocessing_vcf38_native_tokenizing(
                        vcf_file=vcf_files,
                        genome_reference_38_path=resolve_path(args.hg38),
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
            target_handler=target_handler,
            prerelu=not getattr(args, 'relu', False)
        )
        predict_config.prefix = 'fold' + str(fold) + '_'
        predictor = Predictor(model, test_dataloader, predict_config)
        predictor.batch_predict()

    all_fold = glob.glob(os.path.join(result_dir, 'fold*_prediction_*.tsv'))
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

    # Concatenate per-fold feature files horizontally; one ensemble file per feature head.
    # Per-fold files are named fold<label>_features_<head>.tsv with columns M1, M2, ..., sample.
    feat_files = glob.glob(os.path.join(result_dir, 'fold*_features_*.tsv'))
    heads = {}
    for fpath in feat_files:
        m = re.match(r'fold(.+?)_features_(.+)\.tsv$', os.path.basename(fpath))
        if m:
            heads.setdefault(m.group(2), []).append((m.group(1), fpath))

    def _fold_sort_key(label):
        try:
            return (0, int(label))
        except ValueError:
            return (1, label)

    for head, items in heads.items():
        items.sort(key=lambda x: _fold_sort_key(x[0]))
        merged = None
        for fold_label, fpath in items:
            df = pd.read_csv(fpath, sep='\t', low_memory=False)
            df = df.rename(columns={c: '{}_{}'.format(c, fold_label) for c in df.columns if c != 'sample'})
            merged = df if merged is None else merged.merge(df, on='sample', how='inner')
            os.remove(fpath)
        cols = [c for c in merged.columns if c != 'sample'] + ['sample']
        merged = merged[cols]
        out_path = os.path.join(result_dir, 'ensemble_features_{}.tsv'.format(head))
        merged.to_csv(out_path, sep='\t', index=False, float_format='%.8f')
        print('ensemble features saved to ' + out_path)


def _hash_dir_outputs(result_dir):
    """sha256 of each prediction tsv in result_dir, keyed by filename — used to
    check that repeated same-seed runs (e.g. tag d6) are bit-identical."""
    return {os.path.basename(fpath): _sha256(fpath)
            for fpath in sorted(glob.glob(os.path.join(result_dir, '*prediction*.tsv')))}


def _emit_reproduce_metrics(save_dir):
    """After a reproduce-train run, compute per-class precision/recall/F1 plus a
    confusion matrix from the best-epoch validation logits and write them next to
    the checkpoint (per_class_metrics.tsv / metrics_summary.tsv /
    confusion_matrix.tsv). Best-effort: a metrics failure must never fail a
    training run that already succeeded, so everything here is guarded."""
    from muat.metrics import metrics_from_logits  # local import (sklearn/pandas)

    logits_files = sorted(glob.glob(os.path.join(save_dir, 'best_val_*.tsv')))
    if not logits_files:
        print('metrics: no best_val_*.tsv in {} — skipping.'.format(save_dir))
        return
    for lf in logits_files:
        # 'best_val_first_logits.tsv' -> head 'first_logits'
        head = os.path.basename(lf)[len('best_val_'):-len('.tsv')]
        prefix = '' if len(logits_files) == 1 else head + '_'
        try:
            m = metrics_from_logits(lf, out_dir=save_dir, prefix=prefix)
        except Exception as exc:  # noqa: BLE001 — never crash a finished run
            print('metrics: skipping {} ({}).'.format(os.path.basename(lf), exc))
            continue
        print('metrics [{}]: accuracy={:.4f}  macro-F1={:.4f}  weighted-F1={:.4f}  '
              '({} samples, {} classes) -> {}per_class_metrics.tsv'.format(
                  head, m['accuracy'], m['macro']['f1'], m['weighted']['f1'],
                  m['n_samples'], m['n_classes'], prefix))


def _run_reproduce(args):
    """Dispatch `muat reproduce <tag>`: resolve assets from cache, then run via
    the existing predict/train code paths with a pinned seed."""
    experiments = load_experiments()

    if getattr(args, 'list', False):
        list_tags(experiments)
        return
    if not args.tag:
        raise ValueError('a tag is required (e.g. `muat reproduce d2`), or use --list.')

    recipe = get_recipe(args.tag, experiments)
    cache_dir = resolve_cache_dir(args.cache_dir)
    result_dir = ensure_dirpath(
        resolve_path(args.result_dir) if args.result_dir
        else os.path.join(os.getcwd(), 'reproduce_results', args.tag))

    # Seed resolution: --unseeded skips seeding entirely (non-deterministic),
    # --seed overrides the recipe, otherwise use the recipe's pinned seed.
    if getattr(args, 'unseeded', False):
        seed = None
    elif getattr(args, 'seed', None) is not None:
        seed = args.seed
    else:
        seed = recipe.get('seed')

    if seed is not None:
        set_seed(seed)
    else:
        print('running UNSEEDED (no set_seed; cuDNN left non-deterministic)')

    mode = recipe.get('mode')
    if mode not in ('predict', 'train'):
        raise NotImplementedError(
            "reproduce for mode {!r} (tag {}) is not wired yet.".format(mode, args.tag))

    # Offline by default: verify the cache, or fetch if explicitly allowed.
    try:
        ensure_assets_present(recipe, cache_dir, experiments, args.from_raw)
    except FileNotFoundError:
        if not args.allow_download:
            raise
        print('assets missing; --allow-download set, fetching now...')
        fetch_tag(args.tag, cache_dir_arg=args.cache_dir, from_raw=args.from_raw)
        ensure_assets_present(recipe, cache_dir, experiments, args.from_raw)

    if mode == 'train':
        if args.dry_run:
            print('DRY RUN — reproduce {} (train)'.format(args.tag))
            print('  seed        : {}'.format(seed))
            print('  cache_dir   : {}'.format(cache_dir))
            print('  save_dir    : {}'.format(result_dir))
            print('  hyperparams : {}'.format(recipe.get('hyperparams', {})))
            return
        save_dir = _run_train_from_scratch(
            build_train_namespace(recipe, cache_dir, result_dir, experiments, seed=seed))
        print('reproduce {} (train) done -> checkpoint saved in {}'.format(args.tag, save_dir))
        _emit_reproduce_metrics(save_dir)
        return

    repeat = int(recipe.get('repeat', 1))

    def make_ns(out_dir):
        return build_predict_namespace(recipe, cache_dir, out_dir, experiments,
                                       from_raw=args.from_raw, relu=args.relu)

    if args.dry_run:
        ns = make_ns(result_dir)
        print('DRY RUN — reproduce {}'.format(args.tag))
        print('  seed        : {}'.format(seed))
        print('  cache_dir   : {}'.format(cache_dir))
        print('  result_dir  : {}'.format(result_dir))
        print('  repeat      : {}'.format(repeat))
        print('  checkpoint  : {}'.format(ns.ckpt_filepath))
        print('  input_list  : {}'.format(ns.input_list))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if repeat <= 1:
        _run_predict(make_ns(result_dir), device)
        print('reproduce {} done -> {}'.format(args.tag, result_dir))
        return

    # Run-level reproducibility (tag d6): repeat with the same seed and compare.
    hashes = []
    for i in range(repeat):
        run_dir = ensure_dirpath(os.path.join(result_dir, 'run{}'.format(i + 1)))
        set_seed(recipe.get('seed'))
        _run_predict(make_ns(run_dir), device)
        hashes.append(_hash_dir_outputs(run_dir))

    identical = all(h == hashes[0] for h in hashes)
    print('reproduce {}: {} runs, outputs {}identical'.format(
        args.tag, repeat, '' if identical else 'NOT '))
    if recipe.get('assert_identical') and not identical:
        raise RuntimeError(
            'tag {} requires identical outputs across {} same-seed runs, but they differ.'
            .format(args.tag, repeat))


def _run_train_from_scratch(args):
    """Train a MuAt model from scratch. Shared by `muat train from-scratch` and
    `muat reproduce` train tags (e.g. d1). Returns the save_dir holding the
    trained checkpoint."""
    extdir = ensure_dirpath(pkg_path('extfile'))

    motif_path = resolve_path(args.motif_dictionary_filepath) or f"{extdir}/dictMutation.tsv"
    pos_path = resolve_path(args.position_dictionary_filepath) or f"{extdir}/dictChpos.tsv"
    ges_path = resolve_path(args.ges_dictionary_filepath) or f"{extdir}/dictGES.tsv"

    # `preprocess --build-dictionary` records how the position dictionary's coordinates
    # were built (native hg38, or hg19/lifted-to-hg19) in a sidecar next to it -- read it
    # here so the checkpoint carries it automatically, with no extra flag for the user to
    # remember. Absent for the shipped default dictionary or any dictionary built before
    # this sidecar existed, both of which are 'hg19' (today's only prior behaviour).
    genome_build_mode = 'hg19'
    pos_sidecar = pos_path[:-len('.tsv')] + '.genome_build_mode.json' if pos_path.endswith('.tsv') else None
    if pos_sidecar and os.path.exists(pos_sidecar):
        with open(pos_sidecar) as f:
            genome_build_mode = json.load(f).get('genome_build_mode', 'hg19')

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
        target_handler=target_handler,
        patience=getattr(args, 'patience', 0),
        lr_patience=getattr(args, 'lr_patience', None),
        lr_factor=getattr(args, 'lr_factor', 0.5),
        min_lr=getattr(args, 'min_lr', 1e-7),
        seed=getattr(args, 'seed', None),
    )

    model = get_model(arch, model_config)

    # Reuses the same --seed already passed to TrainerConfig above (no new flag).
    # None means unseeded -> the dataloader samples genuinely randomly; an int means
    # every over-cap sample's subsample is deterministic for that seed, and since it's
    # saved into the checkpoint's dataloader_config, a later `predict` against this
    # checkpoint reproduces the identical draw automatically.
    dataloader_seed = getattr(args, 'seed', None)
    train_dataloader_config = DataloaderConfig(
        model_input=model_config.model_input,
        mutation_type_ratio=model_config.mutation_type_ratio,
        mutation_sampling_size=args.mutation_sampling_size,
        sampling_replacement=args.sampling_replacement,
        genome_build_mode=genome_build_mode,
        seed=dataloader_seed
    )
    test_dataloader_config = DataloaderConfig(
        model_input=model_config.model_input,
        mutation_type_ratio=model_config.mutation_type_ratio,
        mutation_sampling_size=args.mutation_sampling_size,
        sampling_replacement=args.sampling_replacement,
        genome_build_mode=genome_build_mode,
        seed=dataloader_seed
    )

    train_dataloader = MuAtDataloader(train_split, train_dataloader_config)
    test_dataloader = MuAtDataloader(test_split, test_dataloader_config)

    trainer = Trainer(model, train_dataloader, test_dataloader, trainer_config)
    trainer.batch_train()
    return save_dir


def main():
    args = get_main_args()

    genomedir = pkg_path('genome_reference')
    genomedir = ensure_dirpath(genomedir)

    pkg_ckpt = pkg_path('pkg_ckpt')
    pkg_ckpt = ensure_dirpath(pkg_ckpt)

    if args.command in ('fetch', 'reproduce'):
        # These commands surface expected, user-actionable states (missing
        # cached assets, unpopulated splits, not-yet-wired tags). Report them
        # as a clean message + non-zero exit instead of a Python traceback.
        try:
            if args.command == 'fetch':
                fetch_tag(args.tag, cache_dir_arg=args.cache_dir,
                          from_raw=args.from_raw)
            else:
                _run_reproduce(args)
        except (FileNotFoundError, NotImplementedError, ValueError) as err:
            print('muat {}: {}'.format(args.command, err), file=sys.stderr)
            sys.exit(1)
        return

    if args.command == 'predict-ensemble' and args.source == 'pretrained':
        unziping_from_package_installation()

    if args.command == 'download':
        if not (args.pcawg or getattr(args, 'reference', False)):
            raise SystemExit(
                'muat download: choose at least one of --pcawg / --reference')
        download_data_path = resolve_path(args.download_dir)

        if args.pcawg:
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
            download_icgc_object_storage(data_path=download_data_path, files_to_download=files_to_download)
            print("Download completed. Data saved in " + str(download_data_path))

        if getattr(args, 'reference', False):
            # Neither build selected means both, matching download_reference()'s
            # own defaults. Files are left as downloaded (.fa.gz). read_reference()
            # does accept a .gz path, but it shells out to gunzip_file() into a temp
            # file on EVERY call, so anyone making repeated runs should gunzip once
            # and pass the plain .fa instead.
            pick_one = args.hg19 or args.hg38
            download_reference(download_data_path,
                               hg19=args.hg19 or not pick_one,
                               hg38=args.hg38 or not pick_one)
            print("Reference download completed. Files saved in " + str(download_data_path))
        # Without this return the branch falls through to the `else: raise
        # ValueError(f"Unknown command: ...")` that closes main(), because
        # 'download' is not part of the predict/preprocess/train elif chain below.
        # Every released version so far finished the download and *then* exited
        # non-zero with "Unknown command: download".
        return

    if args.command == 'predict':
        if getattr(args, 'seed', None) is not None:
            set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _run_predict(args, device)

    elif args.command == 'predict-ensemble':
        if getattr(args, 'seed', None) is not None:
            set_seed(args.seed)
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

        # NOTE: the reference requirement is checked per stage in
        # util._validate_preprocess_args() -- `tokenize` and `build-dictionary` consume
        # already-annotated input and legitimately have no --hg19/--hg38.

        # Stage -> what to run. The four stages make the previously-guarded contradictions
        # unrepresentable: `annotate` has no dictionary arguments at all, `tokenize` has no
        # reference and cannot build, `build-dictionary` cannot tokenize.
        stage = getattr(args, 'stage', None) or 'full'
        if stage == 'annotate':
            annotate_only, build_dict = True, False
        elif stage == 'build-dictionary':
            # Inputs are the annotated corpus; derive the vocabulary and stop.
            annotate_only, build_dict = True, True
            args.annotated = True
            tokenize_after_build = False
        elif stage == 'tokenize':
            annotate_only, build_dict = False, False
            args.annotated = True
        else:                                       # full
            annotate_only = getattr(args, 'annotate_only', False)
            build_dict = getattr(args, 'build_dictionary', False)
            if build_dict and (args.motif_dictionary_filepath
                               or args.position_dictionary_filepath
                               or args.ges_dictionary_filepath):
                raise ValueError(
                    "--build-dictionary derives the dictionaries from this data, so it cannot "
                    "be combined with an explicit --*-dictionary-filepath. Drop one, or split "
                    "the run into `preprocess build-dictionary` and `preprocess tokenize`.")
        if stage != 'build-dictionary':
            tokenize_after_build = True

        if build_dict and len(vcf_files) < 2:
            print('muat preprocess: WARNING --build-dictionary given for only {} input file(s). '
                  'A dictionary is corpus-level -- build it from the WHOLE cohort in one '
                  'invocation (--input-list), or every sample ends up with its own '
                  'vocabulary.'.format(len(vcf_files)), file=sys.stderr)

        # With --build-dictionary the annotation pass runs first and tokenizing is deferred
        # until the dictionaries exist, so the dispatch below takes the annotate-only route.
        if build_dict:
            annotate_only = True

        # Dictionaries are only needed for tokenizing. Loading them under --no-tokenize
        # would also print the misleading 'using default dictionary' line for a run that
        # never consults one.
        dict_motif = dict_pos = dict_ges = None
        if annotate_only:
            pass
        elif (
            args.motif_dictionary_filepath is None or
            args.position_dictionary_filepath is None or
            args.ges_dictionary_filepath is None
        ):
            extdir = ensure_dirpath(pkg_path('extfile'))
            dict_motif = pd.read_csv(os.path.join(extdir, 'dictMutation.tsv'), sep='\t')
            dict_pos = pd.read_csv(os.path.join(extdir, 'dictChpos.tsv'), sep='\t')
            dict_ges = pd.read_csv(os.path.join(extdir, 'dictGES.tsv'), sep='\t')
            print('using default dictionary in ' + extdir + 'dict{Mutation,Chpos,GES}.tsv')
        else:
            dict_motif = pd.read_csv(resolve_path(args.motif_dictionary_filepath), sep='\t')
            dict_pos = pd.read_csv(resolve_path(args.position_dictionary_filepath), sep='\t')
            dict_ges = pd.read_csv(resolve_path(args.ges_dictionary_filepath), sep='\t')

        if args.annotated:
            # With --build-dictionary the inputs ARE the annotated corpus, so there is
            # nothing to annotate and tokenizing is deferred to the dictionary step below
            # (which needs the vocabulary before it can assign tokens).
            if not build_dict:
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
                if annotate_only:
                    preprocessing_vcf(
                        vcf_file=vcf_files,
                        genome_reference_path=genome_reference_path_hg19,
                        tmp_dir=tmp_dir,
                    )
                else:
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
                    if annotate_only:
                        preprocessing_vcf38(
                            vcf_file=vcf_files,
                            genome_reference_38_path=genome_reference_path_hg38,
                            tmp_dir=tmp_dir,
                        )
                    else:
                        preprocessing_vcf38_tokenizing(
                            vcf_file=vcf_files,
                            genome_reference_38_path=genome_reference_path_hg38,
                            tmp_dir=tmp_dir,
                            dict_motif=dict_motif,
                            dict_pos=dict_pos,
                            dict_ges=dict_ges
                        )
                else:
                    if annotate_only:
                        preprocessing_vcf38_native(
                            vcf_file=vcf_files,
                            genome_reference_38_path=genome_reference_path_hg38,
                            tmp_dir=tmp_dir,
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
                if annotate_only:
                    preprocessing_tsv(vcf_files, genome_reference_path_hg19, tmp_dir)
                else:
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
                    if annotate_only:
                        preprocessing_tsv38(vcf_files, genome_reference_path_hg38, tmp_dir)
                    else:
                        preprocessing_tsv38_tokenizing(
                            vcf_files,
                            genome_reference_path_hg38,
                            tmp_dir,
                            dict_motif,
                            dict_pos,
                            dict_ges
                        )
                else:
                    if annotate_only:
                        preprocessing_tsv38_native(vcf_files, genome_reference_path_hg38, tmp_dir)
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

        if build_dict:
            # The annotated corpus now exists in tmp_dir; derive the vocabulary from it and
            # tokenize in the same pass, so the tokens and the dictionaries cannot disagree.
            # `--annotated` (the --preannotated resume route) has no --hg19/--hg38 of its own --
            # the reference choice was made in a prior `annotate` run this invocation never
            # sees -- so genome_build_mode can't be derived here and is left unset (train
            # falls back to 'hg19', today's only prior behaviour, for that route).
            if args.annotated:
                genome_build_mode = None
            elif args.hg38 is not None:
                genome_build_mode = 'hg19' if args.liftover else 'hg38_native'
            else:
                genome_build_mode = 'hg19'
            from .dictionary import build_dictionaries
            built = build_dictionaries(
                # --annotated: the inputs themselves are the corpus, and they may live in
                # several directories, so pass the resolved list rather than a directory.
                data_dir=None if args.annotated else tmp_dir,
                files=vcf_files if args.annotated else None,
                out_dir=args.dictionary_out_dir or tmp_dir,
                which=[w.strip() for w in args.dictionary_which.split(',') if w.strip()],
                suffix=args.dictionary_suffix,
                motif_labels=args.motif_labels,
                mut_type_from=getattr(args, 'mut_type_from', None),
                # `build-dictionary` as a standalone stage stops at the dictionaries; the
                # tokenizing is then a separate (parallelisable) `tokenize` run.
                tokenize_to=tmp_dir if tokenize_after_build else None,
                genome_build_mode=genome_build_mode,
            )
            if not tokenize_after_build:
                print('\nNow tokenize with them:')
                print('  muat preprocess tokenize --input-list <same list> --tmp-dir <dir> \\')
                resolved = built.get('resolved', {})
                for flag, kind in (('--motif-dictionary-filepath', 'motif'),
                                   ('--position-dictionary-filepath', 'pos'),
                                   ('--ges-dictionary-filepath', 'ges')):
                    print('      {} {}'.format(flag, resolved.get(kind, '?')))
                return
            print('\nTrain with these dictionaries (pass ALL THREE, or the defaults silently '
                  'reintroduce the shipped ones):')
            resolved = built.get('resolved', {})
            for flag, kind in (('--motif-dictionary-filepath', 'motif'),
                               ('--position-dictionary-filepath', 'pos'),
                               ('--ges-dictionary-filepath', 'ges')):
                print('  {} {}'.format(flag, resolved.get(kind, '?')))

        # Verify something was actually written. get_motif_pos_ges() catches per-sample
        # exceptions so one bad input does not abandon a batch, and its 0/1 return was
        # never inspected -- so a run that preprocessed NOTHING still printed
        # 'preprocessed data saved in ...' and exited 0, which is indistinguishable from
        # success inside a job script. Exit non-zero instead, and say what is missing.
        produced = sorted(set(glob.glob(os.path.join(tmp_dir, '*.muat.tsv'))
                              + glob.glob(os.path.join(tmp_dir, '*.muat.tsv.gz'))
                              + glob.glob(os.path.join(tmp_dir, '*.annotate.tsv.gz'))
                              + glob.glob(os.path.join(tmp_dir, '*.gc.genic.exonic.cs.tsv.gz'))))
        n_in = len(vcf_files)
        if not produced:
            print('muat preprocess: produced no output for any of the {} input file(s). '
                  'See the errors above.'.format(n_in), file=sys.stderr)
            sys.exit(1)
        if len(produced) < n_in:
            print('muat preprocess: WARNING only {}/{} input file(s) produced output; '
                  'the rest failed (see errors above).'.format(len(produced), n_in),
                  file=sys.stderr)

    elif args.command == 'train':
        if args.subcommand == 'from-scratch':
            set_seed(getattr(args, 'seed', 1337))
            _run_train_from_scratch(args)

        elif args.subcommand == 'from-checkpoint':
            save_dir = ensure_dirpath(resolve_path(args.save_dir))
            os.makedirs(save_dir, exist_ok=True)

            load_ckpt_filepath = resolve_path(args.ckpt_filepath)
            checkpoint = load_and_check_checkpoint(load_ckpt_filepath)

            model_config = checkpoint['model_config']
            trainer_config = checkpoint['trainer_config']
            dataloader_config = checkpoint['dataloader_config']

            trainer_config.save_ckpt_dir = save_dir
            trainer_config.max_epochs = args.epoch
            trainer_config.learning_rate = args.learning_rate
            trainer_config.batch_size = args.batch_size
            trainer_config.patience = getattr(args, 'patience', 0)
            trainer_config.lr_patience = getattr(args, 'lr_patience', None)
            trainer_config.lr_factor = getattr(args, 'lr_factor', 0.5)
            trainer_config.min_lr = getattr(args, 'min_lr', 1e-7)

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
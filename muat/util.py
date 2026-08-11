import sys, gzip, datetime
import os
import errno
import shutil
import tempfile
import subprocess
import pdb
import argparse
from muat.model import *
from muat._resources import pkg_path
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from muat.dataloader import *
from muat.trainer import *
import csv
import json
import glob

def get_main_args():
    parser = argparse.ArgumentParser(description='Mutation Attention Tool')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    def _make_required_group(p):
        """Create a 'required arguments' group and promote it above the default
        'optional arguments' so --help lists required flags first."""
        g = p.add_argument_group('required arguments')
        p._action_groups.insert(1, p._action_groups.pop())
        return g

    download_parser = subparsers.add_parser(
        'download', help='Download the dataset or a reference genome.')
    download_req = _make_required_group(download_parser)
    download_req.add_argument("--download-dir", type=str, default=None, required=True,
                              help='Directory for storing the downloaded dataset.')
    # --pcawg is no longer argparse-required because --reference alone is a valid
    # invocation (and either may be given, or both). core.py errors when neither
    # is set, which keeps the "choose something" rule in one place instead of
    # splitting it across two mutually-exclusive groups.
    download_what = download_parser.add_argument_group(
        'what to download (choose at least one)')
    download_what.add_argument("--pcawg", action="store_true",
                               help="Download the PCAWG dataset.")
    download_what.add_argument("--reference", action="store_true",
                               help="Download reference genome FASTAs. Fetches both builds "
                                    "unless narrowed with --hg19 / --hg38. Files arrive "
                                    "gzipped; gunzip once if you plan repeated runs, since "
                                    "a .fa.gz is re-decompressed on every use.")
    download_what.add_argument("--hg19", action="store_true",
                               help="With --reference: hg19 only "
                                    "(Sanger PanCancer genome.fa.gz).")
    download_what.add_argument("--hg38", action="store_true",
                               help="With --reference: hg38 only "
                                    "(UCSC goldenPath hg38.fa.gz).")

    # ---------------------------------------------------------------------------------
    # preprocess: four named pipeline stages.
    #
    #   raw VCF/TSV --[annotate]--> *.annotate.tsv.gz --[tokenize]--> *.muat.tsv
    #                                       \--[build-dictionary]--> dict*.tsv
    #
    # Stages exist so that illegal combinations are UNREPRESENTABLE rather than guarded:
    # the flat flag form needed three separate contradiction checks (already-annotated
    # input + stop-before-tokenizing, build-a-dictionary + stop-before-tokenizing,
    # build-a-dictionary + supply-a-dictionary) for combinations that simply cannot be
    # expressed once each stage has its own parser and its own arguments.
    #
    # Omitting the stage runs the FULL pipeline, which keeps the 0.1.21 command line
    # (`muat preprocess --vcf --hg19 ... `) working unchanged.
    # ---------------------------------------------------------------------------------
    preprocess = subparsers.add_parser(
        'preprocess',
        help='Preprocess the dataset (raw -> annotated -> tokenized).',
        description='Run the whole pipeline by default, or one named stage: '
                    'annotate | build-dictionary | tokenize | full.')

    def _add_inputs(p, required=True):
        g = _make_required_group(p) if required else p
        inp = g.add_mutually_exclusive_group(required=required)
        inp.add_argument("--input-filepath", nargs="+", default=None,
                         help="Input file paths.")
        inp.add_argument("--input-list", type=str, default=None,
                         help="Text file listing input paths, one per line.")
        return g

    def _add_reference(p, group=None):
        g = (group or p).add_mutually_exclusive_group(required=False)
        g.add_argument("--hg19", type=str, default=None,
                       help="Path to GRCh37/hg19 (.fa or .fa.gz).")
        g.add_argument("--hg38", type=str, default=None,
                       help="Path to GRCh38/hg38 (.fa or .fa.gz).")
        p.add_argument('--liftover', action='store_true', default=False,
                       help='Only valid with --hg38: convert coordinates to GRCh37/hg19. '
                            'Required to use the pretrained checkpoints, which have an hg19 '
                            'position vocabulary. Without it, coordinates stay native GRCh38.')

    def _add_input_kind(p, group=None):
        g = (group or p).add_mutually_exclusive_group(required=False)
        g.add_argument("--vcf", action="store_true", help="Inputs are VCF files.")
        g.add_argument("--somagg", action="store_true", help="Inputs are SomAgg VCF files.")
        g.add_argument("--tsv", action="store_true", help="Inputs are TSV files.")
        return g

    def _add_dictionary_build_args(p):
        p.add_argument("--dictionary-out-dir", type=str, default=None,
                       help="Where the dictionaries are written (default: --tmp-dir).")
        p.add_argument("--dictionary-suffix", type=str, default='',
                       help="Suffix for the dictionary filenames, e.g. _hg38.")
        p.add_argument("--dictionary-which", type=str, default='pos,motif,ges',
                       help="Which dictionaries to build (default: all three). For a new "
                            "reference genome `pos` alone is usually enough -- motifs and "
                            "genic/exonic/strand categories do not depend on the build.")
        p.add_argument("--motif-labels", type=str, default='inherit',
                       choices=['inherit', 'hybrid', 'refalt'],
                       help="How mut_type is assigned to motifs. inherit (default) drops "
                            "motifs the shipped dictionary lacks; hybrid labels those from "
                            "ref/alt allele lengths; refalt derives all of them. hybrid and "
                            "refalt cannot produce SV, MEI or Neg and diverge from the shipped "
                            "convention, so a dictMutation*.provenance.txt records what "
                            "was done.")
        p.add_argument("--mut-type-from", type=str, default=None,
                       help="dictMutation.tsv to inherit mut_type labels from "
                            "(default: the shipped one).")

    def _add_dictionary_use_args(p):
        p.add_argument('--motif-dictionary-filepath', type=str, default=None,
                       help='Motif dictionary (.tsv). Defaults to the shipped one.')
        p.add_argument('--position-dictionary-filepath', type=str, default=None,
                       help='Genomic position dictionary (.tsv). Defaults to the shipped one, '
                            'which is hg19-derived -- supply your own for native GRCh38.')
        p.add_argument('--ges-dictionary-filepath', type=str, default=None,
                       help='Genic/exonic/strand dictionary (.tsv). Defaults to the shipped one.')

    def _add_full_args(p, required_inputs=True):
        """Everything the end-to-end pipeline accepts.

        required_inputs=False for the parent parser: when a stage subcommand is used, the
        input flags appear AFTER the stage word and are consumed by the stage's parser, so
        argparse would reject them as missing at parent level. The flat (stage-less) form is
        validated in _validate_preprocess_args() instead.
        """
        req = _add_inputs(p, required=required_inputs)
        _add_input_kind(p, req)
        # 0.1.21 spelling; a full run over already-annotated input is the `tokenize` stage.
        req.add_argument("--preannotated", "--annotated", dest="annotated",
                         action="store_true", help=argparse.SUPPRESS)
        _add_reference(p)
        p.add_argument("--tmp-dir", type=str, default=None,
                       help='Where outputs are written.')
        p.add_argument("--build-dictionary", action="store_true",
                       help="Derive the token dictionaries from THIS data and tokenize with "
                            "them, instead of using the shipped ones. Required for a new "
                            "reference genome. Must see the whole cohort in one invocation "
                            "(--input-list); for a large cohort use the annotate / "
                            "build-dictionary / tokenize stages instead so annotation can "
                            "run in parallel.")
        _add_dictionary_build_args(p)
        _add_dictionary_use_args(p)
        p.add_argument("--no-tokenize", "--annotate-only", dest="annotate_only",
                       action="store_true", help=argparse.SUPPRESS)

    # Flat form == full pipeline, so the released command line still works.
    _add_full_args(preprocess, required_inputs=False)

    stages = preprocess.add_subparsers(
        dest='stage', required=False, metavar='{annotate,build-dictionary,tokenize,full}',
        help='Run a single stage instead of the whole pipeline.')

    st_annotate = stages.add_parser(
        'annotate',
        help='Raw input -> *.annotate.tsv.gz. Per-sample, so safe to run as a SLURM array.')
    req = _add_inputs(st_annotate)
    _add_input_kind(st_annotate, req)
    _add_reference(st_annotate)
    st_annotate.add_argument("--tmp-dir", type=str, default=None,
                             help='Where the annotated files are written.')

    st_builddict = stages.add_parser(
        'build-dictionary',
        help='Annotated corpus -> dict{Chpos,Mutation,GES}.tsv. Corpus-level: must see every '
             'sample at once, so run it ONCE over the whole cohort.')
    _add_inputs(st_builddict)
    st_builddict.add_argument("--tmp-dir", type=str, default=None,
                              help='Fallback output directory if --dictionary-out-dir is unset.')
    _add_dictionary_build_args(st_builddict)

    st_tokenize = stages.add_parser(
        'tokenize',
        help='Annotated corpus -> *.muat.tsv, using existing dictionaries. Per-sample.')
    _add_inputs(st_tokenize)
    st_tokenize.add_argument("--tmp-dir", type=str, default=None,
                             help='Where the tokenized files are written.')
    _add_dictionary_use_args(st_tokenize)

    st_full = stages.add_parser(
        'full', help='Raw input -> tokenized output (the default when no stage is given).')
    _add_full_args(st_full)

    # Predict subparser
    # Shared input/output args used by both predict and predict-ensemble.
    # Preprocessing mode is inferred from input file suffix; see _validate_predict_inputs.
    # `required` is an argument_group on `p`; the leaf parser puts its own
    # leaf-specific required flags there too so the help displays them together.
    def _add_common_predict_args(p, required):
        inp = required.add_mutually_exclusive_group(required=True)
        inp.add_argument("--input-filepath", nargs="+",
                         help="Input paths. Accepted: .vcf{,.gz}, .maf{,.gz}, .tsv "
                              "(preprocessed first), or .muat.tsv{,.gz} (already preprocessed). "
                              "All inputs must be the same kind.")
        inp.add_argument("--input-list", type=str, default=None,
                         help="Text file listing input paths, one per line. Same suffix rules.")
        required.add_argument("--result-dir", type=str, default=None, required=True,
                              help='Result directory where the output will be written (.tsv).')

        ref = p.add_mutually_exclusive_group()
        ref.add_argument("--hg19", type=str, default=None,
                         help="Path to GRCh37/hg19 (.fa or .fa.gz). "
                              "Required when inputs are raw (.vcf/.maf/.tsv).")
        ref.add_argument("--hg38", type=str, default=None,
                         help="Path to GRCh38/hg38 (.fa or .fa.gz). "
                              "Required when inputs are raw (.vcf/.maf/.tsv).")
        p.add_argument("--tmp-dir", type=str, default=None,
                       help='Directory for storing preprocessed files (used only for raw inputs).')
        p.add_argument("--relu", action="store_true",
                       help='Apply ReLU to saved features. By default, features are '
                            'pre-ReLU (raw Linear output). Logits/predictions are unchanged.')

    # predict: single-model prediction.
    # Two sources: 'pretrained' (downloads benchmark) and 'from-checkpoint' (user .pthx).
    predict_parser = subparsers.add_parser('predict', help='Predict samples with a single model.')
    predict_source = predict_parser.add_subparsers(
        dest='source', required=True,
        help='Model source: pretrained benchmark, or your own checkpoint.')

    # pretrained: benchmark checkpoint auto-downloaded from HuggingFace.
    pre = predict_source.add_parser(
        'pretrained',
        help='Use benchmark checkpoint (auto-downloaded from HuggingFace).')
    pre_assay = pre.add_subparsers(
        dest='assay', required=True,
        help='Assay type for the benchmark.')

    pre_wgs = pre_assay.add_parser('wgs', help='Whole Genome Sequence benchmark.')
    pre_wgs_req = _make_required_group(pre_wgs)
    pre_wgs_req.add_argument("--mutation-type", type=str, required=True,
                             choices=['snv', 'snv+mnv', 'snv+mnv+indel',
                                      'snv+mnv+indel+svmei', 'snv+mnv+indel+svmei+neg'],
                             help='Selects which benchmark checkpoint to download/use.')
    _add_common_predict_args(pre_wgs, pre_wgs_req)

    pre_wes = pre_assay.add_parser('wes', help='Whole Exome Sequence benchmark.')
    pre_wes_req = _make_required_group(pre_wes)
    pre_wes_req.add_argument("--mutation-type", type=str, required=True,
                             choices=['snv', 'snv+mnv', 'snv+mnv+indel'],
                             help='Selects which benchmark checkpoint to download/use.')
    _add_common_predict_args(pre_wes, pre_wes_req)

    # from-checkpoint: user-supplied checkpoint; assay is inferred from the .pthx.
    fc = predict_source.add_parser(
        'from-checkpoint',
        help='Use your own checkpoint (.pthx); assay is inferred from the checkpoint.')
    fc_req = _make_required_group(fc)
    fc_req.add_argument("--ckpt-filepath", type=str, required=True,
                        help='Path to load the checkpoint (.pthx).')
    _add_common_predict_args(fc, fc_req)

    train_parser = subparsers.add_parser('train', help='Train the MuAt model.')
    train_subparsers = train_parser.add_subparsers(dest='subcommand', required=True, help='Available commands.')
    from_scratch = train_subparsers.add_parser('from-scratch', help='Train from scratch.')
    from_scratch_req = _make_required_group(from_scratch)
    from_scratch_req.add_argument('--mutation-type', type=str, default=None, required=True,
                    help='Mutation type; choose from {snv, snv+mnv, snv+mnv+indel, snv+mnv+indel+svmei, snv+mnv+indel+svmei+neg}.')
    from_scratch_req.add_argument("--use-motif", action="store_true", required=True,
                                  help="Use motif input.")
    from_scratch_req.add_argument('--train-split-filepath', type=str, default=None, required=True,
                    help='Training split data; example file in example_files/train_split_example.tsv.')
    from_scratch_req.add_argument('--val-split-filepath', type=str, default=None, required=True,
                    help='Internal validation split data; example file in example_files/val_split_example.tsv.')
    from_scratch_req.add_argument('--save-dir', type=str, default=None, required=True,
                    help='Directory to save the model.')

    from_scratch.add_argument("--use-position", action="store_true", help="Use genomic position input.")
    from_scratch.add_argument("--use-ges", action="store_true", help="Use genic, exonic, and strand annotation.")
    from_scratch.add_argument('--epoch', type=int, default=1,
                    help='Number of epochs (default: 5).')
    from_scratch.add_argument('--learning-rate', type=float, default=6e-4,
                    help='Learning rate (default: 6e-4).')
    from_scratch.add_argument('--batch-size', type=int, default=2,
                    help='Batch size (default: 2).')
    from_scratch.add_argument('--n-layer', type=int, default=1,
                    help='Number of attention layers (default: 1).')
    from_scratch.add_argument('--n-head', type=int, default=8,
                    help='Number of attention heads (default: 8).')
    from_scratch.add_argument('--n-emb', type=int, default=128,
                    help='Embedding dimension (default: 128).')
    from_scratch.add_argument('--mutation-sampling-size', type=int, default=5000,
                    help='Maximum number of mutations to fetch for the model (default: 5000).')
    from_scratch.add_argument("--sampling-replacement", action="store_true", help="Use sampling with replacement. Default is False")
    from_scratch.add_argument('--seed', type=int, default=1337,
                    help='Random seed for reproducible training (Python/NumPy/PyTorch + cuDNN deterministic). Default: 1337.')
    from_scratch.add_argument('--patience', type=int, default=0,
                    help='Early-stopping patience: stop if validation loss has not improved for N epochs. 0 disables.')
    from_scratch.add_argument('--lr-patience', type=int, default=None,
                    help='Reduce LR by --lr-factor after this many epochs without val-loss improvement. Default: max(1, patience//2).')
    from_scratch.add_argument('--lr-factor', type=float, default=0.5,
                    help='Multiplier applied to LR when val loss plateaus (default 0.5).')
    from_scratch.add_argument('--min-lr', type=float, default=1e-7,
                    help='Lower bound for LR scheduling (default 1e-7).')
    from_scratch.add_argument('--motif-dictionary-filepath', type=str, default=None, help='Path to the motif dictionary (.tsv).')
    from_scratch.add_argument('--position-dictionary-filepath', type=str, default=None, help='Path to the genomic position dictionary (.tsv).')
    from_scratch.add_argument('--ges-dictionary-filepath', type=str, default=None, help='Path to the genic exonic strand dictionary (.tsv).')

    from_checkpoint = train_subparsers.add_parser('from-checkpoint', help='Train from a checkpoint.')
    from_checkpoint_req = _make_required_group(from_checkpoint)
    from_checkpoint_req.add_argument("--ckpt-filepath", type=str, default=None, required=True,
                        help='Path to load the checkpoint (.pthx).')
    from_checkpoint_req.add_argument("--mutation-type", type=str, default=None, required=True,
                        help='Mutation type; choose from {snv, snv+mnv, snv+mnv+indel, snv+mnv+indel+svmei, snv+mnv+indel+svmei+neg}.')
    from_checkpoint_req.add_argument('--train-split-filepath', type=str, default=None, required=True,
                    help='Training split data; example file in example_files/train_split_example.tsv.')
    from_checkpoint_req.add_argument('--val-split-filepath', type=str, default=None, required=True,
                    help='Internal validation split data; example file in example_files/val_split_example.tsv.')
    from_checkpoint_req.add_argument('--save-dir', type=str, default=None, required=True,
                    help='Directory to save the model.')
    from_checkpoint_req.add_argument('--epoch', type=int, default=1, required=True,
                    help='Number of epochs (default: 5).')

    from_checkpoint.add_argument('--learning-rate', type=float, default=6e-4,
                    help='Learning rate (default: 6e-4).')
    from_checkpoint.add_argument('--batch-size', type=int, default=2,
                    help='Batch size (default: 2).')
    from_checkpoint.add_argument('--mutation-sampling-size', type=int, default=5000,
                    help='Maximum number of mutations to fetch for the model (default: 5000).')
    from_checkpoint.add_argument("--sampling-replacement", action="store_true", help="Use sampling with replacement.  Default is False")
    from_checkpoint.add_argument('--patience', type=int, default=0,
                    help='Early-stopping patience: stop if validation loss has not improved for N epochs. 0 disables.')
    from_checkpoint.add_argument('--lr-patience', type=int, default=None,
                    help='Reduce LR by --lr-factor after this many epochs without val-loss improvement. Default: max(1, patience//2).')
    from_checkpoint.add_argument('--lr-factor', type=float, default=0.5,
                    help='Multiplier applied to LR when val loss plateaus (default 0.5).')
    from_checkpoint.add_argument('--min-lr', type=float, default=1e-7,
                    help='Lower bound for LR scheduling (default 1e-7).')

    # predict-ensemble: averages logits across fold checkpoints.
    # Two sources: 'pretrained' (downloads benchmark bundle) and 'from-checkpoint' (user .pthx files).
    ensemble_parser = subparsers.add_parser(
        'predict-ensemble',
        help='Run ensemble prediction (averages logits across fold checkpoints).')
    ensemble_source = ensemble_parser.add_subparsers(
        dest='source', required=True,
        help='Model source: pretrained benchmark bundle, or your own checkpoints.')

    # pretrained: benchmark bundle auto-downloaded from HuggingFace.
    ens_pre = ensemble_source.add_parser(
        'pretrained',
        help='Use benchmark ensemble (auto-downloaded from HuggingFace).')
    ens_pre_assay = ens_pre.add_subparsers(
        dest='assay', required=True,
        help='Assay type for the benchmark bundle.')

    ens_pre_wgs = ens_pre_assay.add_parser('wgs', help='Whole Genome Sequence benchmark.')
    ens_pre_wgs_req = _make_required_group(ens_pre_wgs)
    ens_pre_wgs_req.add_argument("--mutation-type", type=str, required=True,
                                 choices=['snv', 'snv+mnv', 'snv+mnv+indel',
                                          'snv+mnv+indel+svmei', 'snv+mnv+indel+svmei+neg'],
                                 help='Selects which benchmark bundle to download/use.')
    _add_common_predict_args(ens_pre_wgs, ens_pre_wgs_req)

    ens_pre_wes = ens_pre_assay.add_parser('wes', help='Whole Exome Sequence benchmark.')
    ens_pre_wes_req = _make_required_group(ens_pre_wes)
    ens_pre_wes_req.add_argument("--mutation-type", type=str, required=True,
                                 choices=['snv', 'snv+mnv', 'snv+mnv+indel'],
                                 help='Selects which benchmark bundle to download/use.')
    _add_common_predict_args(ens_pre_wes, ens_pre_wes_req)

    # from-checkpoint: user-supplied ensemble; assay is inferred from each checkpoint.
    ens_fc = ensemble_source.add_parser(
        'from-checkpoint',
        help='Use your own ensemble checkpoints (.pthx); assay is inferred from each checkpoint.')
    ens_fc_req = _make_required_group(ens_fc)
    ens_fc_req.add_argument("--ckpt-filepath", nargs="+", required=True,
                            help='One .pthx per fold; logits are averaged across them.')
    _add_common_predict_args(ens_fc, ens_fc_req)

    # reproduce: run a pinned experiment from experiments.md by tag.
    reproduce_parser = subparsers.add_parser(
        'reproduce',
        help='Reproduce a pinned experiment (see experiments.md) by tag, offline-safe.')
    reproduce_parser.add_argument('tag', nargs='?', default=None,
                                  help='Experiment tag, e.g. d2. Omit with --list.')
    reproduce_parser.add_argument('--list', action='store_true',
                                  help='List available tags and exit.')
    reproduce_parser.add_argument('--cache-dir', type=str, default=None,
                                  help='Asset cache (default: $MUAT_CACHE, then ~/.cache/muat). '
                                       'On HPC use a path shared by login and compute nodes.')
    reproduce_parser.add_argument('--result-dir', type=str, default=None,
                                  help='Where to write outputs (default: ./reproduce_results/<tag>).')
    reproduce_parser.add_argument('--from-raw', action='store_true',
                                  help='Preprocess raw PCAWG data instead of using the preprocessed bundle.')
    reproduce_parser.add_argument('--allow-download', action='store_true',
                                  help='Permit downloading missing assets at run time '
                                       '(default: offline; error with a fetch hint instead).')
    reproduce_parser.add_argument('--dry-run', action='store_true',
                                  help='Resolve and print the run configuration without executing.')
    reproduce_parser.add_argument('--relu', action='store_true',
                                  help='Apply ReLU to saved features (passed through to predict).')
    reproduce_parser.add_argument('--seed', type=int, default=None,
                                  help='Override the recipe seed (default: use the seed pinned in '
                                       'experiments.json).')
    reproduce_parser.add_argument('--unseeded', action='store_true',
                                  help='Run without seeding (non-deterministic): skip set_seed and leave '
                                       'cuDNN non-deterministic. Overrides --seed and the recipe seed.')

    # fetch: stage a tag's assets into the cache (run on a node with internet).
    fetch_parser = subparsers.add_parser(
        'fetch',
        help="Download a reproduce tag's assets into the cache (run where there is internet).")
    fetch_parser.add_argument('tag', help='Experiment tag, e.g. d2.')
    fetch_parser.add_argument('--cache-dir', type=str, default=None,
                              help='Asset cache (default: $MUAT_CACHE, then ~/.cache/muat).')
    fetch_parser.add_argument('--from-raw', action='store_true',
                              help='Also stage raw PCAWG data + reference for the --from-raw run path.')

    # NOTE: there is deliberately no top-level `build-dictionary` command. Rebuilding the
    # vocabulary is not a standalone task -- it only makes sense as part of preparing data,
    # and it must be followed by tokenizing with the dictionaries just built or the two
    # disagree. So it lives on preprocess as `--build-dictionary`, which covers both the
    # from-raw route (`--vcf --hg38 --build-dictionary`) and the resume route
    # (`--preannotated --build-dictionary`, for a corpus already annotated by an array job).
    # The implementation is muat/dictionary.py:build_dictionaries(), still importable.

    args = parser.parse_args()
    _validate_predict_inputs(parser, args)
    _validate_preprocess_args(parser, args)

    return args


def _validate_preprocess_args(parser, args):
    """Per-stage requirements for `muat preprocess`.

    Enforced here rather than by argparse because the parent parser cannot mark the input
    flags required without breaking `preprocess <stage> --input-...` (the flags land on the
    stage's parser, so the parent sees none).
    """
    if getattr(args, 'command', None) != 'preprocess':
        return

    stage = getattr(args, 'stage', None) or 'full'
    args.stage = stage

    if args.input_filepath is None and args.input_list is None:
        parser.error('muat preprocess {}: one of --input-filepath or --input-list is required.'
                     .format('' if stage == 'full' else stage).replace('  ', ' '))

    if stage in ('annotate', 'full'):
        # `--preannotated` is the legacy way to say "inputs are already annotated"; that is
        # now the `tokenize` stage, and it needs no reference genome.
        if not (args.vcf or args.somagg or args.tsv or args.annotated):
            parser.error('one of --vcf, --somagg or --tsv is required to say what kind of '
                         'input this is. If the inputs are already annotated '
                         '(*.annotate.tsv[.gz]), use `muat preprocess tokenize` instead.')
        if not args.annotated and args.hg19 is None and args.hg38 is None:
            parser.error('--hg19 or --hg38 is required to annotate raw input.')
        if args.liftover and args.hg38 is None:
            parser.error('--liftover is only valid together with --hg38.')

    if stage == 'build-dictionary':
        # Guarding what the stage split cannot make unrepresentable: the vocabulary is
        # derived from the whole cohort at once, so a single-sample invocation is almost
        # always a mistake worth flagging (see the warning in core, which is not fatal).
        pass


# --- annotated-file naming --------------------------------------------------------------
#
# The annotated (pre-tokenized) file is what `preprocess --no-tokenize` emits and what
# `--preannotated` / --build-dictionary consume. It used to be named after every annotation
# stage it had been through -- `.gc.genic.exonic.cs.tsv.gz` -- which is unreadable and leaks
# implementation detail. New output is `.annotate.tsv.gz`.
#
# The old suffix is still ACCEPTED on input: data/preprocessed/ holds 1901 files under the
# old name, and any user who ran an earlier muat has the same. Never write it.
ANNOTATED_SUFFIX = '.annotate.tsv.gz'
LEGACY_ANNOTATED_SUFFIXES = ('.gc.genic.exonic.cs.tsv.gz', '.gc.genic.exonic.cs.tsv')
ANNOTATED_SUFFIXES_ACCEPTED = (ANNOTATED_SUFFIX, '.annotate.tsv') + LEGACY_ANNOTATED_SUFFIXES
# Globs for discovering annotated files in a directory (both namings).
ANNOTATED_GLOBS = ('*.annotate.tsv*', '*.gc.genic.exonic.cs.tsv*')


def annotated_path(tmp_dir, sample_name):
    """Where the annotated file for this sample is WRITTEN."""
    return ensure_dirpath(tmp_dir) + sample_name + ANNOTATED_SUFFIX


def find_annotated_path(tmp_dir, sample_name):
    """Existing annotated file for this sample, preferring the current name over the
    legacy one. Returns None if neither is present."""
    tmp_dir = ensure_dirpath(tmp_dir)
    for suffix in (ANNOTATED_SUFFIX, '.annotate.tsv') + LEGACY_ANNOTATED_SUFFIXES:
        candidate = tmp_dir + sample_name + suffix
        if os.path.exists(candidate):
            return candidate
    return None


# Input-suffix whitelist for muat predict / predict-ensemble.
PREPROCESSED_SUFFIXES = ('.muat.tsv', '.muat.tsv.gz')
RAW_SUFFIXES = ('.vcf', '.vcf.gz', '.maf', '.maf.gz', '.tsv')


def _classify_input(path):
    """Return 'preprocessed', 'raw', or None for an unsupported suffix.
    Preprocessed match is checked first because '.muat.tsv' also ends in '.tsv'."""
    if path.endswith(PREPROCESSED_SUFFIXES):
        return 'preprocessed'
    if path.endswith(RAW_SUFFIXES):
        return 'raw'
    return None


def _validate_predict_inputs(parser, args):
    """Post-parse validation for predict / predict-ensemble.

    Rules:
    - All inputs must share the same kind ('raw' or 'preprocessed'); mixed batches rejected.
    - Each input must match the whitelist suffixes.
    - --hg19/--hg38 required iff kind == 'raw'; rejected if kind == 'preprocessed'.

    Sets args.needs_preprocessing on success."""
    if getattr(args, 'command', None) not in ('predict', 'predict-ensemble'):
        return

    if args.input_list is not None:
        try:
            with open(resolve_path(args.input_list)) as f:
                paths = [line.strip() for line in f if line.strip()]
        except OSError as e:
            parser.error("could not read --input-list {!r}: {}".format(args.input_list, e))
        if not paths:
            parser.error("--input-list {!r} is empty.".format(args.input_list))
    else:
        paths = list(args.input_filepath)

    kinds = {_classify_input(p): None for p in paths}
    bad = [p for p in paths if _classify_input(p) is None]
    if bad:
        parser.error(
            "unsupported input suffix(es): " + ", ".join(bad[:3])
            + (" ..." if len(bad) > 3 else "")
            + ". Accepted: .vcf{,.gz}, .maf{,.gz}, .tsv (raw) or .muat.tsv{,.gz} (preprocessed).")

    kinds = {_classify_input(p) for p in paths}
    if len(kinds) > 1:
        parser.error(
            "mixed input kinds are not allowed: all inputs must be either raw "
            "(.vcf/.maf/.tsv) or preprocessed (.muat.tsv).")

    kind = kinds.pop()
    has_ref = bool(args.hg19 or args.hg38)

    if kind == 'raw' and not has_ref:
        parser.error("--hg19 or --hg38 is required because inputs are raw "
                     "(.vcf/.maf/.tsv) and need preprocessing.")
    if kind == 'preprocessed' and has_ref:
        parser.error("--hg19/--hg38 was given, but all inputs are already preprocessed "
                     "(.muat.tsv). Drop the reference flag.")

    args.needs_preprocessing = (kind == 'raw')


def mut_type_checkpoint_handler(mutation_type,wgs_wes):
    ckptdir = pkg_path('pkg_ckpt')
    ckptdir = ensure_dirpath(ckptdir)

    if wgs_wes == 'wgs':
        if mutation_type == 'snv':
            load_ckpt_filepath = ensure_dirpath(ckptdir+'/pcawg_wgs/snv/') + 'pcawg-wgs-snv-MuAtMotifPositionGES.pthx'
        elif mutation_type == 'snv+mnv':
            load_ckpt_filepath = ensure_dirpath(ckptdir+'/pcawg_wgs/snv+mnv/') + 'pcawg-wgs-snv+mnv-MuAtMotifPositionGESF.pthx'
        elif mutation_type == 'snv+mnv+indel':
            load_ckpt_filepath = ensure_dirpath(ckptdir+'/pcawg_wgs/snv+mnv+indel/') +'pcawg-wgs-snv+mnv+indel-MuAtMotifPositionGESF.pthx'
        elif mutation_type == 'snv+mnv+indel+svmei':
            load_ckpt_filepath = ensure_dirpath(ckptdir+'/pcawg_wgs/snv+mnv+indel+svmei/') + 'pcawg-wgs-snv+mnv+indel+svmei-MuAtMotifPositionGESF.pthx'

    elif wgs_wes == 'wes':
        if mutation_type == 'snv':
            load_ckpt_filepath = ensure_dirpath(ckptdir+'/tcga_wes/snv/') + 'tcga-wes-snv-MuAtMotifPositionGESF.pthx'
        elif mutation_type == 'snv+mnv':
            load_ckpt_filepath = ensure_dirpath(ckptdir+'/tcga_wes/snv+mnv/') + 'tcga-wes-snv+mnv-MuAtMotifPositionGESF.pthx'
        elif mutation_type == 'snv+mnv+indel':
            load_ckpt_filepath = ensure_dirpath(ckptdir+'/tcga_wes/snv+mnv+indel/') +'tcga-wes-snv+mnv+indel-MuAtMotifPositionGESF.pthx'
    print('load from ckpt ' + load_ckpt_filepath)
    return load_ckpt_filepath


def get_main_args_old():
    parser = argparse.ArgumentParser(description='Mutation Attention Tool')

    parser.add_argument("--arch", type=str, default=None,
                    help='architecture')
    parser.add_argument("--mutation-type", type=str, default=None,
                    help='mutation type, only {snv,snv+mnv,snv+mnv+indel,snv+mnv+indel+svmei,snv+mnv+indel+svmei+neg} can be applied')

    # EXECUTIION
    #preprocess
    parser.add_argument('--preprocess-vcf-hg19', action='store_true', default=False,
                        help='execute preprocess for vcf hg19')
    parser.add_argument('--preprocess-vcf-hg38', action='store_true', default=False,
                        help='execute preprocess for vcf hg38')
    parser.add_argument('--tokenizing', action='store_true', default=False,
                        help='execute tokenizing preprocessed files')
    parser.add_argument('--train', action='store_true', default=False,
                        help='execute training')
    parser.add_argument('--from-scratch', action='store_true', default=False,
                        help='execute training from scratch')
    parser.add_argument('--from-checkpoint', action='store_true', default=False,
                        help='execute training from checkpoint')

    #PREDICTION
    parser.add_argument('--predict-vcf-hg19', action='store_true', default=False,
                        help='execute prediction of vcf hg19')
    parser.add_argument('--predict-vcf-hg38', action='store_true', default=False,
                        help='execute prediction of vcf hg38')

    #INPUT
    parser.add_argument("--vcf-hg19-filepath", type=str, default=None,
                        help='List of vcf hg19')
    parser.add_argument("--vcf-hg38-filepath", type=str, default=None,
                        help="List of vcf hg38")

    parser.add_argument("--preprocessed-filepath", type=str, default=None,
                        help="List of annotated files (.annotate.tsv.gz) which contain motif position and ges to be tokenized")
    #OUTPUT
    parser.add_argument("--result-dir", type=str, default=None,
                    help='Absolut Path to save the result')
    #CHECKPOINT
    parser.add_argument('--load-ckpt-filepath', type=str, default=None,
                    help='load checkpoint complete path file')
    
    parser.add_argument('--save-ckpt-dir', type=str, default=None,
                    help='save checkpoint directory')

    #UTILS
        #preprocess
    parser.add_argument('--hg19-filepath', type=str, default=None,
                    help='Absolut Path to Human Genome Reference hg19')
    parser.add_argument('--hg38-filepath', type=str, default=None,
                    help='Absolut Path to Human Genome Reference hg38')
    parser.add_argument('--motif-dictionary-filepath', type=str, default=None,
                    help='Absolut Path to motif dictionary (.tsv)')
    parser.add_argument('--position-dictionary-filepath', type=str, default=None,
                    help='Absolut Path to genomic position dictionary (.tsv)')
    parser.add_argument('--ges-dictionary-filepath', type=str, default=None,
                    help='Absolut Path to genic exonic strand dictionary (.tsv)')

        #HYPERPARAMETERS
    parser.add_argument('--epoch', type=int, default=1,
                    help='number of epoch')
    parser.add_argument('--learning-rate', type=float, default=6e-4,
                    help='learning rate')
    parser.add_argument('--batch-size', type=int, default=1,
                    help='batch size')
    parser.add_argument('--n-layer', type=int, default=1,
                    help='attention layer')
    parser.add_argument('--n-head', type=int, default=8,
                    help='attention head')
    parser.add_argument('--n-emb', type=int, default=128,
                    help='embedding dimension') 
    parser.add_argument('--mutation-sampling-size', type=int, default=5000,
                    help='embedding dimension')

    #TRAIN RELATED
    parser.add_argument('--train-split-filepath', type=str, default=None,
                    help='training split filepath')
    parser.add_argument('--val-split-filepath', type=str, default=None,
                    help='internal validation split filepath')
    parser.add_argument('--target-dict-filepath', type=str, default=None,
                    help='target dictionary filepath') 
    parser.add_argument('--subtarget-dict-filepath', type=str, default=None,
                    help='subtarget dictionary filepath') 


    parser.add_argument('--tmp-dir', type=str, default=None,
                    help='directory to store preprocessed files')

    args = parser.parse_args()
    return args

def check_model_match(model_name,pretrained_model):
    return True

def initialize_pretrained_weight(model_name,model_config,checkpoint):

    #pdb.set_trace()
    model = get_model(model_name,model_config)
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['weight']
    filtered_pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(filtered_pretrained_dict)
    #pdb.set_trace()
    model.load_state_dict(model_dict)

    return model

def get_model(arch,model_config=None):
    if arch == 'MuAtMotif':
        return MuAtMotif(model_config)
    elif arch == 'MuAtMotifF':
        return MuAtMotifF(model_config)
    elif arch == 'MuAtMotifPosition':
        return MuAtMotifPosition(model_config)
    elif arch == 'MuAtMotifPositionF':
        return MuAtMotifPositionF(model_config)
    elif arch == 'MuAtMotifPositionGES':
        return MuAtMotifPositionGES(model_config)
    elif arch == 'MuAtMotifPositionGESF':
        return MuAtMotifPositionGESF(model_config)
    elif arch == 'MuAtMotifF_2Labels':
        return MuAtMotifF_2Labels(model_config)    
    elif arch == 'MuAtMotifPositionF_2Labels':
        return MuAtMotifPositionF_2Labels(model_config)
    elif arch == 'MuAtMotifPositionGESF_2Labels':
        return MuAtMotifPositionGESF_2Labels(model_config)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

class LabelEncoderFromCSV:
    def __init__(self, csv_file=None, class_name_col=None, class_index_col=None):
        if csv_file is not None:
            self.class_to_idx = {}
            self.idx_to_class = {}
            self._load_class_mapping(csv_file, class_name_col, class_index_col)
            self.classes_ = list(self.class_to_idx.keys())

    def _load_class_mapping(self, csv_file, class_name_col, class_index_col):
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                class_name = row[class_name_col]
                class_idx = int(row[class_index_col])
                self.class_to_idx[class_name] = class_idx
                self.idx_to_class[class_idx] = class_name

    def fit_transform(self, labels):
        return [self.class_to_idx[label] for label in labels]

    def inverse_transform(self, encoded_labels):
        return [self.idx_to_class[idx] for idx in encoded_labels]

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        obj = cls.__new__(cls)  # create an instance without calling __init__
        obj.class_to_idx = data["class_to_idx"]
        obj.idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
        obj.classes_ = data["classes_"]
        return obj

def multifiles_handler(file):
    if isinstance(file, str):
        file = [file]
    return file

def load_token_dict(checkpoint):
    dict_motif = checkpoint['motif_dict']
    dict_pos = checkpoint['pos_dict']
    dict_ges = checkpoint['ges_dict']
    return dict_motif, dict_pos, dict_ges

def load_target_handler(checkpoint):
    target_handler = checkpoint['target_handler']
    return target_handler

def mutation_type_ratio(snv, mnv, indel, sv_mei, neg,pd_motif):

    if snv + mnv + indel + sv_mei + neg != 1:
        raise ValueError("The sum of mutation types must be 1")

    if snv < 0 or mnv < 0 or indel < 0 or sv_mei < 0 or neg < 0:
        raise ValueError("Mutation types must be non-negative")

    vocabsize = 0
    vocabNisi = len(pd_motif.loc[pd_motif['mut_type']=='SNV'])
    vocabSNV = len(pd_motif.loc[pd_motif['mut_type']=='MNV'])
    vocabindel = len(pd_motif.loc[pd_motif['mut_type']=='indel']) 
    vocabSVMEI = len(pd_motif.loc[pd_motif['mut_type'].isin(['MEI','SV'])])
    vocabNormal = len(pd_motif.loc[pd_motif['mut_type']=='Normal'])

    if snv>0:
        vocabsize = vocabNisi
    if mnv>0:
        vocabsize = vocabNisi + vocabSNV
    if indel>0:
        vocabsize = vocabNisi + vocabSNV + vocabindel         
    if sv_mei>0:
        vocabsize = vocabNisi + vocabSNV + vocabindel + vocabSVMEI   
    if neg>0:
        vocabsize = vocabNisi + vocabSNV + vocabindel + vocabSVMEI + vocabNormal
    
    return {
        'snv': snv,
        'mnv': mnv,
        'indel': indel,
        'sv_mei': sv_mei,
        'neg': neg
    }, vocabsize



# translation table to map each character to a nucleotide or N
valid_dna = ''.join([chr(x) if chr(x) in 'ACGTN' else 'N' for x in range(256)])

dna_comp = {'A' : 'T', 'C' : 'G', 'G' : 'C', 'T' : 'A',
            'N' : 'N', '-' : '-', '+' : '+'}

def dna_comp_default(x):
    r = dna_comp.get(x)
    return r if r is not None else x

def read_codes():
    data = [
    ["A", "A", "A"], ["A", "C", "!"], ["A", "G", "@"], ["A", "T", "#"], ["A", "N", "N"], ["A", "-", "1"],
    ["C", "A", "$"], ["C", "C", "C"], ["C", "G", "%"], ["C", "T", "^"], ["C", "N", "N"], ["C", "-", "2"],
    ["G", "A", "&"], ["G", "C", "*"], ["G", "G", "G"], ["G", "T", "~"], ["G", "N", "N"], ["G", "-", "3"],
    ["T", "A", ":"], ["T", "C", ";"], ["T", "G", "?"], ["T", "T", "T"], ["T", "N", "N"], ["T", "-", "4"],
    ["N", "N", "N"], ["N", "-", "N"],
    ["-", "A", "5"], ["-", "C", "6"], ["-", "G", "7"], ["-", "T", "8"], ["-", "N", "N"],
    ["-", "SV_DEL", "D"], ["-", "SV_DUP", "P"], ["-", "SV_INV", "I"], ["-", "SV_BND", "B"]]

    codes = {}
    rcodes = {}
    for s in data:
        ref, alt, code = '\t'.join(s).strip().split()
        if ref not in codes:
            codes[ref] = {}
        codes[ref][alt] = code
        rcodes[code] = (ref, alt)
    rcodes['N'] = ('N', 'N')  # ->N, N>-, A>N etc all map to N, make sure that 'N'=>'N>N'
    return codes, rcodes

def ensure_dir_exists(filepath):
    """Create directory if it doesn't exist for the given filepath."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def openz(path, mode='r'):
    if path.endswith('.gz'):
        # For gzipped files, use binary mode and handle decoding manually
        if 'b' not in mode and 't' not in mode:
            mode = mode + 'b'  # Default to binary mode for gzip
        return gzip.open(path, mode)
    elif path == '-':
        if mode == 'r':
            return sys.stdin
        else:
            return sys.stdout
    else:
        # For regular files, use text mode
        if 'b' not in mode and 't' not in mode:
            mode = mode + 't'  # Default to text mode for regular files
        return open(path, mode)

def get_timestr():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def open_stream(fn):
    if fn.endswith('.gz'):
        f = gzip.open(fn, 'rt')  # 'rt' mode for text reading
        sample_name = os.path.basename(fn).split('.')[0]
    else:
        f = open(fn)
        sample_name = os.path.basename(fn).split('.')[0]
    assert(('.maf' in fn and '.vcf' in fn) == False)  # filenames should specify input type unambiguously
    return f, sample_name

def get_sample_name(fn):
    sample_name = os.path.basename(fn).split('.')[0]
    return sample_name

def gunzip_file(gz_filename):
    filename = os.path.splitext(gz_filename)[0]  # Remove .gz extension
    if os.name == "nt":  # Windows
        cmd = f'powershell -Command "gzip -d \'{gz_filename}\'"'
    else:  # Linux/macOS
        cmd = f"gunzip -c {gz_filename} > {filename}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Decompressed: {gz_filename} -> {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error decompressing {gz_filename}: {e}")

    return filename

def status(msg,verbose, lf=True, time=True):
    if verbose:
        if time:
            tstr = '[{}] '.format(get_timestr())
        else:
            tstr = ''
        sys.stderr.write('{}{}'.format(tstr, msg))
        if lf:
            sys.stderr.write('\n')
        sys.stderr.flush()


def read_reference(reffn, verbose=0):
    R = {}
    chrom = None
    seq = []
    f = None
    temp_file = None
    
    try:
        if reffn.endswith('.gz'):
            if verbose:
                sys.stderr.write('Decompressing gzipped file...\n')
            # Create temp file with same name but without .gz
            temp_path = gunzip_file(reffn)  # remove .gz extension
            f = open(temp_path)
        else:
            f = open(reffn)

        # Original reading logic
        for s in f:
            if s[0] == '>':
                if chrom is not None:
                    R[chrom] = ''.join(seq).translate(valid_dna)
                seq = []
                chrom = s[1:].strip().split()[0]
                if verbose:
                    sys.stderr.write('{} '.format(chrom))
                    sys.stderr.flush()
            else:
                seq.append(s.strip().upper())
        R[chrom] = ''.join(seq).translate(valid_dna)
        
        if verbose:
            sys.stderr.write(' done.\n')
            
    finally:
        if f:
            f.close()
    return R

def is_valid_dna(s):
    s2 = [a in 'ACGTN' for a in s]
    return len(s2) == sum(s2)

def search_best(folder):

    best = ''
    curr_acc = 0
    for x in folder:
        try:
            pd_data = pd.read_csv(x+'/finalprf.csv',index_col=0)
            acc = pd_data['acc'].unique()[0]

            if os.path.isfile(x + '/model.pthx'):
                if acc > curr_acc:
                    curr_acc=acc
                    best = x
            else:
                pass
        except:
            pass
    print(best + '/model.pthx')
    #print(str(curr_acc))
    return best + '/model.pthx', curr_acc

def ensure_dirpath(path, terminator="/"):
    path = path.replace('//', terminator)
    if path.endswith(terminator):
        return path
    else:
        path = path + terminator
    return path

def check_tmp_dir(args):
    if args.tmp_dir is None:
        tmp_dir = ensure_dirpath(os.path.abspath(os.path.join(os.getcwd(), 'data/preprocessed_local')))
        print('--tmp-dir was not defined, --tmp-dir is set to ' + str(tmp_dir))
    else:
        tmp_dir = resolve_path(args.tmp_dir)
    tmp_dir = ensure_dirpath(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir

def get_checkpoint_args():
    args = argparse.Namespace(
        arch=None,
        n_class=None,
        n_layer=1,
        n_head=8,
        n_emb=128,
        get_motif=False,
        get_position=False,
        get_ges=False,
        get_epi=False,
        motif=False,
        motif_pos=False,
        motif_pos_ges=False,
        motif_pos_ges_epi=False
    )
    return args

def resolve_path(path):
    """
    Resolve a Absolut Path to its absolute form, handling both relative and absolute paths.
    If the path is relative, it will be resolved relative to the current working directory.
    
    Args:
        path (str): The Absolut Path to resolve
        
    Returns:
        str: The resolved absolute path
    """
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))

'''
    wgs_wes = predict_parser.add_mutually_exclusive_group(required=True)
    wgs_wes.add_argument("--wgs", action="store_true", help="Run prediction for WGS")
    wgs_wes.add_argument("--wes", action="store_true", help="Run prediction for WES")
    hg19_hg38 = predict_parser.add_mutually_exclusive_group(required=True)
    hg19_hg38.add_argument("--hg19", action="store_true", help="VCF file using hg19 genome reference")
    hg19_hg38.add_argument("--hg38", action="store_true", help="VCF file using hg38 genome reference")
    mut_type_loadckpt = predict_parser.add_mutually_exclusive_group(required=True)
    mut_type_loadckpt.add_argument("--mutation-type", type=str, default=None,
                        help='mutation type, only {snv,snv+mnv,snv+mnv+indel,snv+mnv+indel+svmei,snv+mnv+indel+svmei+neg} can be applied')
    mut_type_loadckpt.add_argument("--load-ckpt-filepath", type=str, default=None,
                        help='complete file Absolut Path to load checkpoint (.pthx), --mutation-type will be adjusted accordingly when loading from ckpt')

    predict_parser.add_argument("--input-filepath", nargs="+", help="input file paths (.vcf or .vcf.gz)")
    predict_parser.add_argument("--result-dir", type=str, default=None,required=True,
                        help='result directory where the output will be written (.tsv)')
    predict_parser.add_argument("--tmp-dir", type=str, default=None,
                        help='directory for storing preprocessed files')
'''
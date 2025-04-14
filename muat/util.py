import sys, gzip, datetime
import os
import errno
import shutil
import tempfile
import subprocess
import pdb
import argparse
from muat.model import *
from pkg_resources import resource_filename
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

    download_parser = subparsers.add_parser('download', help='Download the dataset.')
    download_parser.add_argument("--pcawg", action="store_true", help="Download the PCAWG dataset.", required=True)
    download_parser.add_argument("--download-dir", type=str, default=None,required=True,
                        help='Directory for storing the downloaded dataset.')

    preprocessing = subparsers.add_parser('preprocessing', help='Preprocess the dataset.')
    vcf_somagg_tsv = preprocessing.add_mutually_exclusive_group(required=True)
    vcf_somagg_tsv.add_argument("--vcf", action="store_true", help="Preprocess VCF files.")
    vcf_somagg_tsv.add_argument("--somagg", action="store_true", help="Preprocess SomAgg VCF files.")
    vcf_somagg_tsv.add_argument("--tsv", action="store_true", help="Preprocess TSV files.")
    hg19_hg38 = preprocessing.add_mutually_exclusive_group(required=True)
    hg19_hg38.add_argument("--hg19", type=str, default=None, help="Absolut Path to GRCh37/hg19 (.fa or .fa.gz)")
    hg19_hg38.add_argument("--hg38", type=str, default=None, help="Absolut Path to GRCh38/hg38 (.fa or .fa.gz)")
    
    preprocessing.add_argument("--input-filepath", nargs="+", help="Input file paths.", required=True)

    preprocessing.add_argument("--tmp-dir", type=str, default=None, help='Directory for storing preprocessed files.')
    preprocessing.add_argument('--motif-dictionary-filepath', type=str, default=None, help='Absolut Path to the motif dictionary (.tsv).')
    preprocessing.add_argument('--position-dictionary-filepath', type=str, default=None, help='Absolut Path to the genomic position dictionary (.tsv).')
    preprocessing.add_argument('--ges-dictionary-filepath', type=str, default=None, help='Absolut Path to the genic exonic strand dictionary (.tsv).')

    #preprocessing.add_argument('--hg19-filepath', type=str, default=None, help='Absolut Path to GRCh37/hg19 (.fa or .fa.gz)',required=True)
    #preprocessing.add_argument('--hg38-filepath', type=str, default=None, help='Absolut Path to GRCh38/hg38 (.fa or .fa.gz)',required=True)

    # Predict subparser
    predict_parser = subparsers.add_parser('predict', help='Predict samples.')
    predict_subparser = predict_parser.add_subparsers(dest='command', required=True, help='Available commands.')

    wgs = predict_subparser.add_parser('wgs', help='Whole Genome Sequence.')
    hg19_hg38 = wgs.add_mutually_exclusive_group(required=True)
    hg19_hg38.add_argument("--hg19", type=str, default=None, help="Absolut Path to GRCh37/hg19 (.fa or .fa.gz)")
    hg19_hg38.add_argument("--hg38", type=str, default=None, help="Absolut Path to GRCh38/hg19 (.fa or .fa.gz)")
    hg19_hg38.add_argument("--no-preprocessing", action="store_true", help="Predict directly from preprocessed data (.token.gc.genic.exonic.cs.tsv.gz)")

    mut_type_loadckpt = wgs.add_mutually_exclusive_group(required=True)
    mut_type_loadckpt.add_argument("--mutation-type", type=str, default=None,
                        help='Mutation type; only {snv, snv+mnv, snv+mnv+indel, snv+mnv+indel+svmei, snv+mnv+indel+svmei+neg} can be applied.')
    mut_type_loadckpt.add_argument("--ckpt-filepath", type=str, default=None,
                        help='Complete file Absolut Path to load the checkpoint (.pthx). The mutation type will be adjusted accordingly when loading from the checkpoint.')

    wgs.add_argument("--input-filepath", nargs="+", help="Input file paths (.vcf or .vcf.gz) / .token.gc.genic.exonic.cs.tsv.gz for no preprocessing")
    wgs.add_argument("--result-dir", type=str, default=None, required=True,
                        help='Result directory where the output will be written (.tsv).')
    wgs.add_argument("--tmp-dir", type=str, default=None,
                        help='Directory for storing preprocessed files.')
    #wgs.add_argument('--hg19-filepath', type=str, default=None, help='Absolut Path to GRCh37/hg19 (.fa or .fa.gz)',required=True)
    #wgs.add_argument('--hg38-filepath', type=str, default=None, help='Absolut Path to GRCh38/hg38 (.fa or .fa.gz)',required=True)

    wes = predict_subparser.add_parser('wes', help='Whole Exome Sequence.')
    hg19_hg38 = wes.add_mutually_exclusive_group(required=True)
    hg19_hg38.add_argument("--hg19", type=str, default=None, help="Absolut Path to GRCh37/hg19 (.fa or .fa.gz)")
    hg19_hg38.add_argument("--hg38", type=str, default=None, help="Absolut Path to GRCh37/hg19 (.fa or .fa.gz)")
    hg19_hg38.add_argument("--no-preprocessing", action="store_true", help="Predict from preprocessed data (.token.gc.genic.exonic.cs.tsv.gz)")

    mut_type_loadckpt = wes.add_mutually_exclusive_group(required=True)
    mut_type_loadckpt.add_argument("--mutation-type", type=str, default=None,
                        help='Mutation type; only {snv, snv+mnv, snv+mnv+indel} can be applied.')
    mut_type_loadckpt.add_argument("--load-ckpt-filepath", type=str, default=None,
                        help='Complete file Absolut Path to load the checkpoint (.pthx). The mutation type will be adjusted accordingly when loading from the checkpoint.')

    wes.add_argument("--input-filepath", nargs="+", help="Input file paths (.vcf or .vcf.gz) or .token.gc.genic.exonic.cs.tsv.gz for no preprocessing")
    wes.add_argument("--result-dir", type=str, default=None, required=True,
                        help='Result directory where the output will be written (.tsv).')
    wes.add_argument("--tmp-dir", type=str, default=None,
                        help='Directory for storing preprocessed files.')
    #wes.add_argument('--hg19-filepath', type=str, default=None, help='Absolut Path to GRCh37/hg19 (.fa or .fa.gz)',required=True)
    #wes.add_argument('--hg38-filepath', type=str, default=None, help='Absolut Path to GRCh38/hg38 (.fa or .fa.gz)',required=True)

    train_parser = subparsers.add_parser('train', help='Train the MuAt model.')
    train_subparsers = train_parser.add_subparsers(dest='command', required=True, help='Available commands.')
    from_scratch = train_subparsers.add_parser('from-scratch', help='Train from scratch.')
    from_scratch.add_argument('--mutation-type', type=str, default=None, required=True,
                    help='Mutation type; choose from {snv, snv+mnv, snv+mnv+indel, snv+mnv+indel+svmei, snv+mnv+indel+svmei+neg}.')
    from_scratch.add_argument("--use-motif", action="store_true", help="Use motif input.", required=True)
    from_scratch.add_argument("--use-position", action="store_true", help="Use genomic position input.")
    from_scratch.add_argument("--use-ges", action="store_true", help="Use genic, exonic, and strand annotation.")

    from_scratch.add_argument('--train-split-filepath', type=str, default=None, required=True,
                    help='Training split data; example file in example_files/train_split_example.tsv.')
    from_scratch.add_argument('--val-split-filepath', type=str, default=None, required=True,
                    help='Internal validation split data; example file in example_files/val_split_example.tsv.')
    from_scratch.add_argument('--save-dir', type=str, default=None, required=True,
                    help='Directory to save the model.')    

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
    from_scratch.add_argument("--sampling-replacement", action="store_true", help="Use sampling with replacement.")

    from_scratch.add_argument('--motif-dictionary-filepath', type=str, default=None, help='Absolut Path to the motif dictionary (.tsv).')
    from_scratch.add_argument('--position-dictionary-filepath', type=str, default=None, help='Absolut Path to the genomic position dictionary (.tsv).')
    from_scratch.add_argument('--ges-dictionary-filepath', type=str, default=None, help='Absolut Path to the genic exonic strand dictionary (.tsv).')

    from_checkpoint = train_subparsers.add_parser('from-checkpoint', help='Train from a checkpoint.')
    from_checkpoint.add_argument("--ckpt-filepath", type=str, default=None, required=True,
                        help='Complete file Absolut Path to load the checkpoint (.pthx).')
    from_checkpoint.add_argument("--mutation-type", type=str, default=None, required=True,
                        help='Mutation type; choose from {snv, snv+mnv, snv+mnv+indel, snv+mnv+indel+svmei, snv+mnv+indel+svmei+neg}.')
    
    from_checkpoint.add_argument('--train-split-filepath', type=str, default=None, required=True,
                    help='Training split data; example file in example_files/train_split_example.tsv.')
    from_checkpoint.add_argument('--val-split-filepath', type=str, default=None, required=True,
                    help='Internal validation split data; example file in example_files/val_split_example.tsv.')
    from_checkpoint.add_argument('--save-dir', type=str, default=None, required=True,
                    help='Directory to save the model.')    

    from_checkpoint.add_argument('--epoch', type=int, default=1,required=True,
                    help='Number of epochs (default: 5).')
    from_checkpoint.add_argument('--learning-rate', type=float, default=6e-4,
                    help='Learning rate (default: 6e-4).')
    from_checkpoint.add_argument('--batch-size', type=int, default=2,
                    help='Batch size (default: 2).')
    from_checkpoint.add_argument('--mutation-sampling-size', type=int, default=5000,
                    help='Maximum number of mutations to fetch for the model (default: 5000).')
    from_checkpoint.add_argument("--sampling-replacement", action="store_true", help="Use sampling with replacement.")

    # Predict subparser
    benchmark_parser = subparsers.add_parser('benchmark', help='Run the prediction using the best MuAt ensemble models')
    benchmark_subparser = benchmark_parser.add_subparsers(dest='command', required=True, help='Available commands.')
    
    wgs = benchmark_subparser.add_parser('muat-wgs', help='MuAt Whole Genome Sequence.')
    hg19_hg38 = wgs.add_mutually_exclusive_group(required=True)
    hg19_hg38.add_argument("--hg19", type=str, default=None, help="Absolut Path to GRCh37/hg19 (.fa or .fa.gz)")
    hg19_hg38.add_argument("--hg38", type=str, default=None, help="Absolut Path to GRCh38/hg19 (.fa or .fa.gz)")
    hg19_hg38.add_argument("--no-preprocessing", action="store_true", help="Predict directly from preprocessed data (.token.gc.genic.exonic.cs.tsv.gz)")

    wgs.add_argument("--mutation-type", type=str, default=None,required=True,
                        help='Mutation type; only {snv, snv+mnv, snv+mnv+indel, snv+mnv+indel+svmei, snv+mnv+indel+svmei+neg} can be applied.')
    wgs.add_argument("--input-filepath", nargs="+", help="Input file paths (.vcf or .vcf.gz) / .token.gc.genic.exonic.cs.tsv.gz for no preprocessing")
    wgs.add_argument("--result-dir", type=str, default=None, required=True,
                        help='Result directory where the output will be written (.tsv).')
    wgs.add_argument("--tmp-dir", type=str, default=None,
                        help='Directory for storing preprocessed files.')

    wes = benchmark_subparser.add_parser('muat-wes', help='MuAt Whole Exome Sequence.')
    hg19_hg38 = wes.add_mutually_exclusive_group(required=True)
    hg19_hg38.add_argument("--hg19", type=str, default=None, help="Absolut Path to GRCh37/hg19 (.fa or .fa.gz)")
    hg19_hg38.add_argument("--hg38", type=str, default=None, help="Absolut Path to GRCh37/hg19 (.fa or .fa.gz)")
    hg19_hg38.add_argument("--no-preprocessing", action="store_true", help="Predict from preprocessed data (.token.gc.genic.exonic.cs.tsv.gz)")

    wes.add_argument("--mutation-type", type=str, default=None,required=True,
                        help='Mutation type; only {snv, snv+mnv, snv+mnv+indel} can be applied.')

    wes.add_argument("--input-filepath", nargs="+", help="Input file paths (.vcf or .vcf.gz) / .token.gc.genic.exonic.cs.tsv.gz for no preprocessing")
    wes.add_argument("--result-dir", type=str, default=None, required=True,
                        help='Result directory where the output will be written (.tsv).')
    wes.add_argument("--tmp-dir", type=str, default=None,
                        help='Directory for storing preprocessed files.')
        
    args = parser.parse_args()

    return args

def mut_type_checkpoint_handler(mutation_type,wgs_wes):
    ckptdir = resource_filename('muat','pkg_ckpt')
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
    #PREPROCESSING
    parser.add_argument('--preprocessing-vcf-hg19', action='store_true', default=False,
                        help='execute preprocessing for vcf hg19')
    parser.add_argument('--preprocessing-vcf-hg38', action='store_true', default=False,
                        help='execute preprocessing for vcf hg38')
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
                        help="List of preprocessed files (.gc.genic.exonic.cs.tsv.gz) which contain motif position and ges to be tokenized")
    #OUTPUT
    parser.add_argument("--result-dir", type=str, default=None,
                    help='Absolut Path to save the result')
    #CHECKPOINT
    parser.add_argument('--load-ckpt-filepath', type=str, default=None,
                    help='load checkpoint complete path file')
    
    parser.add_argument('--save-ckpt-dir', type=str, default=None,
                    help='save checkpoint directory')

    #UTILS
        #PREPROCESSING
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
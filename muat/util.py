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


def get_simlified_args():
        parser = argparse.ArgumentParser(description='Mutation Attention Tool')

        # DATASET
        parser.add_argument('--cwd', type=str,help='project dir')
        parser.add_argument('--dataloader', type=str, default=None,
                        help='dataloader setup, option: pcawg or tcga')
        # MODEL
        parser.add_argument('--arch', type=str, default=None,
                        help='architecture')
        # DIRECTORY
            #INPUT DATA
                #PREPROCESSING
        parser.add_argument('--raw-filepath', type=str, default=None,
                        help='path of raw vcf files')
                #TOKENIZING
        parser.add_argument('--prep-filepath', type=str, default=None,
                        help='path to preprocessed files')
                #TRAINING
        parser.add_argument('--trainfold-filepath', type=str, default=None,
                        help='path to training fold split')
        parser.add_argument('--valfold-filepath', type=str, default=None,
                        help='path to validation fold split')
                #PREDICT
        parser.add_argument('--predict-filepath', type=str, default=None,
                        help='path to predict data')
        parser.add_argument('--predict-ready-filepath', type=str, default=None,
                        help='path to clean predict data')
                
            #OUTPUT DATA
                #PREPROCESSING
        parser.add_argument('--prep-outdir', type=str, default=None,
                        help='directory of preprocessed files')
                #TOKENIZING        
        parser.add_argument('--token-outdir', type=str, default=None,
                        help='directory of tokenized files')
                #PREDICT        
        parser.add_argument('--output-pred-dir', type=str, default=None,
                        help='directory of prediction output')


        # EXECUTIION
            #PREPROCESSING
        parser.add_argument('--preprocessing', action='store_true', default=False,
                            help='execute preprocessing')
            #TOKENIZING 
        parser.add_argument('--tokenizing', action='store_true', default=False,
                            help='execute tokenizing')
            #TRAINING
        parser.add_argument('--train', action='store_true', default=False,
                            help='execute training')
            #PREDICTION
        parser.add_argument('--predict-all', action='store_true', default=False,
                            help='execute prediction all samples')
        parser.add_argument('--predict-all-noprep', action='store_true', default=False,
                            help='execute prediction all samples from the preprocessed files')

        parser.add_argument('--get-features', action='store_true', default=False,
                            help='get features from the models')
        parser.add_argument('--convert_hg38_hg19', action='store_true', default=False,
                            help='convert_hg38_hg19')

        parser.add_argument('--motif', action='store_true', default=False)
        parser.add_argument('--motif-pos', action='store_true', default=False)
        parser.add_argument('--motif-pos-ges', action='store_true', default=False)


        #CKPT SAVE
        parser.add_argument('--save-ckpt-filepath', type=str, default=None,
                        help='save checkpoint filename')
        #CKPT LOAD
        parser.add_argument('--load-ckpt-filepath', type=str, default=None,
                        help='load checkpoint complete path file')
        # HYPER PARAMS 
        parser.add_argument('--epoch', type=int, default=1,
                        help='number of epoch')
        parser.add_argument('--l-rate', type=float, default=6e-4,
                        help='learning rate')
        parser.add_argument('--n-class', type=int, default=None,
                        help='number of class')
        parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size')
        parser.add_argument('--block-size', type=int, default=1000,
                        help='block of sequence')
        parser.add_argument('--context-length', type=int, default=3,
                        help='length of sequence')
        parser.add_argument('--n-layer', type=int, default=1,
                        help='attention layer')
        parser.add_argument('--n-head', type=int, default=8,
                        help='attention head')
        parser.add_argument('--n-emb', type=int, default=128,
                        help='embedding dimension')
        parser.add_argument('--fold', type=int, default=1, 
                            help='fold')
        
        #UTILS
            #PREPROCESSING
        parser.add_argument('--hg19-path', type=str, default=None,
                        help='path to Human Genome Reference hg19')
        parser.add_argument('--hg38-path', type=str, default=None,
                        help='path to Human Genome Reference hg38')
        parser.add_argument('--dict-motif-filepath', type=str, default=None,
                        help='path to motif dictionary')
        parser.add_argument('--dict-pos-filepath', type=str, default=None,
                        help='path to position dictionary')
        parser.add_argument('--dict-ges-filepath', type=str, default=None,
                        help='path to ges dictionary')

            #TRAINING
        parser.add_argument('--classinfo-filepath', type=str, default=None,
                        help='path to class info')

        parser.add_argument('--mut-type', type=str, default='',
                        help='mutation type, only [SNV,SNV+MNV,SNV+MNV+indel,SNV+MNV+indel+SV/MEI,SNV+MNV+indel+SV/MEI+Neg] can be applied')
        parser.add_argument('--mutratio', type=str, default=None,
                        help='mutation ratio per mutation type, sum of them must be one')

        args = parser.parse_args()
        return args

def check_model_match(model_name,pretrained_model):
    return True

def initialize_pretrained_weight(model_name,model_config,checkpoint):

    model = get_model(model_name,model_config)
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['weight']
    filtered_pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(filtered_pretrained_dict)
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

def model_input(motif=True,pos=True,ges=True):
    return {
        'motif': motif,
        'pos': pos,
        'ges': ges
    }

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

def get_checkpoint_args():

    parser = argparse.ArgumentParser(description='PCAWG / TCGA experiment')
    parser.add_argument('--arch', type=str, default=None,
                        help='architecture')

    parser.add_argument('--n-class', type=int, default=None,
                        help='number of class')
    parser.add_argument('--n-layer', type=int, default=1,
                        help='attention layer')
    parser.add_argument('--n-head', type=int, default=8,
                    help='attention head')
    parser.add_argument('--n-emb', type=int, default=128,
                    help='embedding dimension')

    parser.add_argument('--get-motif', action='store_true', default=False)
    parser.add_argument('--get-position', action='store_true', default=False)
    parser.add_argument('--get-ges', action='store_true', default=False)
    parser.add_argument('--get-epi', action='store_true', default=False)

    parser.add_argument('--motif', action='store_true', default=False)
    parser.add_argument('--motif-pos', action='store_true', default=False)
    parser.add_argument('--motif-pos-ges', action='store_true', default=False)
    parser.add_argument('--motif-pos-ges-epi', action='store_true', default=False)
    args = parser.parse_args()
    return args
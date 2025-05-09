from .util import *
from .reader import *
import os
import numpy as np
import pdb
import traceback
from tqdm import tqdm
import json
import glob
import subprocess
import pandas as pd
import gzip
from muat.download import download_reference

def combine_somagg_chunks_to_platekey(sample_folder,tmp_dir):
    '''
    sample_folder : list of all sample folders
    tmp_dir: directory after combining all chunks per sample
    '''

    tmp_dir = ensure_dirpath(tmp_dir)
    sample_folder = ensure_dirpath(sample_folder)
    sample_folder = multifiles_handler(sample_folder)

    for sampfold in sample_folder:
        all_chunk = glob.glob(sampfold + '*.tsv')
        pd_persample = pd.DataFrame()
        for perchunk in all_chunk:
            pd_read = pd.read_csv(perchunk,sep='\t',low_memory=False)
            pd_persample = pd.concat([pd_persample,pd_read])
        samp_id = pd_persample['sample'].iloc[0]
        pd_persample.to_csv(tmp_dir + get_sample_name(samp_id) + '.muat.tsv', sep='\t')

def filtering_somagg_vcf(all_somagg_chunks,tmp_dir):
    '''
    all_somagg_chunks : list of all somAgg chunk vcf files
    tmp_dir : directory after filtering somagg vcf
    '''
    header_line = ''

    fns = multifiles_handler(all_somagg_chunks)

    tmp_dir = ensure_dirpath(tmp_dir)
    
    for fn in fns:
        filename_only = get_sample_name(fn)
        exportdir = tmp_dir
        os.makedirs(exportdir,exist_ok=True)
        output_file = exportdir + filename_only + '.tsv'
        #pdb.set_trace()
        
        fvcf = open(output_file, "w")
        with gzip.open(fn, 'rb') as f:
            val_start = 0
            header_line = ''
            for i, l in enumerate(f):
                variable = l.decode('utf-8')

                if variable.startswith('##'):
                    header_line = header_line + variable
                if variable.startswith('#CHROM'):
                    val_start = 1
                    colm = variable.split('\t')
                    colm_front = colm[0:8]
                    fvcf.write('PLATEKEY\t')
                    fvcf.write('\t'.join(colm_front))
                    fvcf.write('\n')
                else:
                    if val_start == 1:
                        colm_value = variable.split('\t')
                        colm_front = colm[0:8]
                        colm_b = colm[9:]
                        colm_back = []
                        for sub in colm_b:
                            colm_back.append(sub.replace("\n", ""))

                        col_vcf = '\t'.join(colm_front + ['Platekey', 'Values'])
                        condition = ['0/0', './.']
                        colm_value_front = colm_value[0:8]
                        colm_value_back = colm_value[9:]

                        for i_c, i_value in enumerate(colm_value_back):
                            if i_value.startswith('0/0') or i_value.startswith('./.'):
                                pass
                            else:
                                if i_value.split(':')[9] == 'PASS':
                                    platekey = colm_back[i_c]
                                    fvcf.write(platekey)
                                    fvcf.write('\t')
                                    fvcf.write('\t'.join(colm_value_front))
                                    fvcf.write('\n')

        fvcf.close()

        '''
        pd_read = pd.read_csv(fn, sep="\t", comment='#',low_memory=False)
        pd_read = pd_read.iloc[:,0:8]
        pd_read.columns = ['#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO']
        pd_read['PLATEKEY'] = filename_only

        pd_read = pd_read[['PLATEKEY','#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO']]
        pd_read.to_csv(output_file,sep='\t',index=False)
        '''

        #for somagg files
        #pd_read = pd.read_csv(output_file,sep='\t',low_memory=False)



def preprocessing_tsv38_tokenizing(tsv_file,genome_reference_38_path,tmp_dir,dict_motif,dict_pos,dict_ges):
    '''
    Preprocess tsv file with GRCh38 and tokenize the motif, pos, and ges
    '''
    tsv_file = multifiles_handler(tsv_file)
    preprocessing_tsv38(tsv_file,genome_reference_38_path,tmp_dir)

    all_preprocessed_vcf = []

    tmp_dir = ensure_dirpath(tmp_dir)

    for x in tsv_file:
        if os.path.exists(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            all_preprocessed_vcf.append(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    #pdb.set_trace()
    tokenizing(dict_motif,dict_pos,dict_ges,all_preprocessed_vcf,tmp_dir)
    #pdb.set_trace()
    all_tokenized = []
    for x in all_preprocessed_vcf:
        if os.path.exists(tmp_dir + get_sample_name(x) + '.muat.tsv'):
            all_tokenized.append(tmp_dir + get_sample_name(x) + '.muat.tsv')
    
    for x in all_tokenized:
        pd_file = pd.read_csv(x,sep='\t',low_memory=False)
        #pdb.set_trace()
        all_samples = pd_file['sample'].unique().tolist()

        for samp in all_samples:
            persamp = pd_file.loc[pd_file['sample']==samp]
            os.makedirs(tmp_dir + samp,exist_ok=True)
            persamp_path = ensure_dirpath(tmp_dir + samp)
            persamp.to_csv(persamp_path + get_sample_name(x) + '.tsv',sep='\t',index=False)
    #os.remove(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    #os.remove(tmp_dir + get_sample_name(x) + '.token.gc.genic.exonic.cs.tsv.gz')

def preprocessing_vcf38_tokenizing(vcf_file,genome_reference_38_path,tmp_dir,dict_motif,dict_pos,dict_ges):
    '''
    Preprocess vcf file with GRCh38 and tokenize the motif, pos, and ges
    '''
    vcf_file = multifiles_handler(vcf_file)
    preprocessing_vcf38(vcf_file,genome_reference_38_path,tmp_dir)

    all_preprocessed_vcf = []

    tmp_dir = ensure_dirpath(tmp_dir)

    for x in vcf_file:
        if os.path.exists(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            all_preprocessed_vcf.append(tmp_dir  + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    #pdb.set_trace()
    tokenizing(dict_motif,dict_pos,dict_ges,all_preprocessed_vcf,tmp_dir)
    #pdb.set_trace()

def preprocessing_tsv38(tsv_file,genome_reference_38_path,tmp_dir,verbose=True):
    '''
    Preprocess tsv file with GRCh38
    '''
    genome_ref38 = read_reference(genome_reference_38_path, verbose=verbose)
    fns = multifiles_handler(tsv_file)
    fns = [resolve_path(x) for x in fns]

    for i, fn in enumerate(fns):
        digits = int(np.ceil(np.log10(len(fns))))
        fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
        # get motif
        f, sample_name = open_stream(fn)
        vr = SomAggTSVReader(f=f, pass_only=True, type_snvs=False)
        status('Writing mutation sequences...', verbose)
        process_input(vr, sample_name, None,tmp_dir,genome_ref38=genome_ref38,liftover=True,verbose=verbose)        
        f.close()

def preprocessing_vcf38(vcf_file,genome_reference_38_path,tmp_dir,verbose=True):
    '''
    Preprocess vcf file with GRCh38
    '''

    if not os.path.exists(genome_reference_38_path):
        print('reference file not found')
        genome_reference_dir = os.path.dirname(genome_reference_38_path)
        print('Downloading reference file to ' + genome_reference_dir)
        download_reference(genome_reference_dir,hg19=False,hg38=True)
        genome_reference_38_path = ensure_dirpath(genome_reference_dir) + 'hg38.fa.gz'

    genome_ref38 = read_reference(genome_reference_38_path, verbose=verbose)
    fns = multifiles_handler(vcf_file)
    fns = [resolve_path(x) for x in fns]

    for i, fn in enumerate(fns):
        digits = int(np.ceil(np.log10(len(fns))))
        fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
        get_motif_pos_ges(fn, None, tmp_dir, genome_ref38=genome_ref38, liftover=True, verbose=verbose)

def preprocessing_vcf_tokenizing(vcf_file,genome_reference_path,tmp_dir,dict_motif,dict_pos,dict_ges):
    '''
    Preprocess vcf file and tokenize the motif, pos, and ges
    '''
    vcf_file = multifiles_handler(vcf_file)
    vcf_file = [resolve_path(x) for x in vcf_file]

    preprocessing_vcf(vcf_file,genome_reference_path,tmp_dir)
    #pdb.set_trace()
    all_preprocessed_vcf = []

    tmp_dir = ensure_dirpath(tmp_dir)
    for x in vcf_file:
        if os.path.exists(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            all_preprocessed_vcf.append(tmp_dir + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    #pdb.set_trace()
    tokenizing(dict_motif,dict_pos,dict_ges,all_preprocessed_vcf,tmp_dir)
    

def get_motif_pos_ges(fn,genome_ref,tmp_dir,genome_ref38=None,liftover=False,verbose=True):
    """
    Preprocess to get the motif from the vcf file
    Args:
        fn: str, path to vcf file
        genome_ref: reference genome variable from read_reference
        tmp_dir: str, path to temporary directory for storing preprocessed files
        liftover: bool, if True, liftover the vcf file from GRCh38 to GRCh37
    """

    tmp_dir = ensure_dirpath(tmp_dir)

    try:
        # get motif
        f, sample_name = open_stream(fn)
        vr = get_reader(f)
        status('Writing mutation sequences...', verbose)
        process_input(vr, sample_name, genome_ref,tmp_dir,genome_ref38=genome_ref38,liftover=liftover,verbose=verbose)
        f.close()
        return 1
    except Exception as e:
        print(f"Error: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return 0
    

def preprocessing_vcf(vcf_file,genome_reference_path,tmp_dir,info_column=None,verbose=True):
    """
    Preprocess one or more VCF files
    Args:
        vcf_file: str or list of str, path(s) to VCF file(s)
        genome_reference_path: str, path to genome reference file
        tmp_dir: str, path to temporary directory for storing preprocessed files
    """

    # Check if reference files exist
    if not os.path.exists(genome_reference_path):
        print('reference file not found')
        genome_reference_dir = os.path.dirname(genome_reference_path)
        print('downloading reference file to ' + genome_reference_dir)
        download_reference(genome_reference_dir,hg19=True,hg38=False)
        genome_reference_path = ensure_dirpath(genome_reference_dir) + 'hg19.fa.gz'

    genome_ref = read_reference(genome_reference_path,verbose=verbose)   

    fns = multifiles_handler(vcf_file)
    fns = [resolve_path(x) for x in fns]
    #pdb.set_trace()
    tmp_dir = ensure_dirpath(tmp_dir)

    for i,fn in enumerate(fns):
        digits = int(np.ceil(np.log10(len(fns))))
        fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
        get_motif_pos_ges(fn,genome_ref,tmp_dir,verbose=verbose)

    return 1

def load_dict(dict_path):
    with open(dict_path, 'r') as f:
        dict_data = json.load(f)
    return dict_data['dict_motif'], dict_data['dict_pos'], dict_data['dict_ges']


def create_dictionary(prep_path,pos_bin_size=1000000,save_dict_path=None):
    '''
    Create a dictionary of preprocessed vcf file and histology abbreviation
    Args:
        prep_path: str or list of str, path(s) to preprocessed vcf file(s)
    '''   # Convert single file path to list for consistent handling
    if isinstance(prep_path, str):
        prep_path = [prep_path]

    dict_motif = set()
    dict_pos = set()    
    dict_ges = set()

    for path in tqdm(prep_path, desc="Generating token", unit="file"):
        #load tsv.gz file
        df = pd.read_csv(path, sep='\t',compression='gzip',low_memory=False)
        #get aliquot_id
        motif = set(df['seq'].to_list())
        ps = (df['pos'] / pos_bin_size).apply(np.floor).astype(int).astype(str)

        chrom = df['chrom'].astype(str)
        chrompos = chrom + '_' + ps
        df['chrompos'] = chrompos
        chrompos = df['chrompos'].unique()
        df['ges'] = df['genic'].astype(str) + '_' + df['exonic'].astype(str) + '_' + df['strand'].astype(str)
        ges = df['ges'].unique()

        df.to_csv(path, sep='\t',compression='gzip',index=False)

        dict_motif.update(motif)
        dict_pos.update(chrompos)
        dict_ges.update(ges)

    # Save all dictionaries as a single JSON file
    combined_dict = {
        'dict_motif': dict_motif,
        'dict_pos': dict_pos,
        'dict_ges': dict_ges
    }

    # Example of converting a set to a list before serialization
    if 'dict_motif' in combined_dict:
        combined_dict['dict_motif'] = list(combined_dict['dict_motif'])  # Convert set to list
    if 'dict_pos' in combined_dict:
        combined_dict['dict_pos'] = list(combined_dict['dict_pos'])  # Convert set to list
    if 'dict_ges' in combined_dict:
        combined_dict['dict_ges'] = list(combined_dict['dict_ges'])  # Convert set to list

    if save_dict_path is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_dict_path), exist_ok=True)
        
        with open(save_dict_path, 'w') as f:
            json.dump(combined_dict, f)
    
    return dict_motif, dict_pos, dict_ges

def tokenizing(dict_motif, dict_pos, dict_ges,all_preprocessed_vcf,tmp_dir,pos_bin_size=1000000):
    '''
    Tokenizing the motif, pos, and ges
    '''
    for path in tqdm(all_preprocessed_vcf, desc="Tokenizing", unit="file"):

        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t',compression='gzip',low_memory=False)

            ps = (df['pos'] / pos_bin_size).apply(np.floor).astype(int).astype(str)
            chrom = df['chrom'].astype(str)
            chrompos = chrom + '_' + ps
            df['chrompos'] = chrompos        
            df['ges'] = df['genic'].astype(str) + '_' + df['exonic'].astype(str) + '_' + df['strand'].astype(str)
    
            df = df.merge(dict_motif, left_on='seq', right_on='seq', how='left')
            df = df.merge(dict_pos, left_on='chrompos', right_on='chrompos', how='left')
            df = df.merge(dict_ges, left_on='ges', right_on='ges', how='left')

            token_file = ensure_dirpath(tmp_dir) + get_sample_name(path) + '.muat.tsv'
            df.to_csv(token_file, sep='\t',index=False)
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


def combine_samplefolders_tosingle_tsv(sample_folder,tmp_dir):
    '''
    sample_folder : list of all sample folders
    tmp_dir: directory after combining all chunks per sample
    '''
    
    all_chunk = glob.glob(os.path.join(sample_folder,'*.tsv'))

    pd_persample = pd.DataFrame()
    for perchunk in all_chunk:
        pd_read = pd.read_csv(perchunk,sep='\t',low_memory=False)
        pd_persample.append(pd_read)
    pd_persample = pd_persample[['#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO','PLATEKEY']]
    pd_persample.columns = ['CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO','PLATEKEY']
    pd_persample.to_csv(tmp_dir + '/' + get_sample_name(sample) + '.somagg.tsv.gz', sep='\t',compression='gzip')

def split_chunk_persample(filtered_chunk,tmp_dir):
    pd_file = pd.read_csv(filtered_chunk,sep='\t',low_memory=False)
    sample_list = pd_file['PLATEKEY'].unique()

    #split samples per chunk
    for sample in sample_list:
        os.makedirs(tmp_dir + '/' + sample,exist_ok=True)
        pd_samp_chunk = pd_file.loc[pd_file['PLATEKEY']==sample]
        pd_samp_chunk.to_csv(tmp_dir + '/' + sample + '/' + get_sample_name(chunk_file) + '.tsv',sep='\t',index=False)


def filtering_somagg_vcf(all_somagg_chunks,tmp_dir):
    '''
    all_somagg_chunks : list of all somAgg chunk vcf files
    tmp_dir : directory after filtering somagg vcf
    '''

    header_line = ''

    fns = multifiles_handler(all_somagg_chunks)
    
    for fn in fns:
        filename_only = get_sample_name(fn)
        exportdir = tmp_dir
        output_file = exportdir + '/' + filename_only + '.tsv'
        
        fvcf = open(outputfile, "w")
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

def preprocessing_tsv38_tokenizing(tsv_file,genome_reference_38_path,genome_reference_19_path,tmp_dir,dict_motif,dict_pos,dict_ges):
    '''
    Preprocess tsv file with GRCh38 and tokenize the motif, pos, and ges
    '''
    tsv_file = multifiles_handler(tsv_file)
    preprocessing_tsv38(tsv_file,genome_reference_38_path,genome_reference_19_path,tmp_dir)

    all_preprocessed_vcf = []

    for x in tsv_file:
        if os.path.exists(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            all_preprocessed_vcf.append(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    #pdb.set_trace()
    tokenizing(dict_motif,dict_pos,dict_ges,all_preprocessed_vcf)
    #pdb.set_trace()



def preprocessing_vcf38_tokenizing(vcf_file,genome_reference_38_path,genome_reference_19_path,tmp_dir,dict_motif,dict_pos,dict_ges):
    '''
    Preprocess vcf file with GRCh38 and tokenize the motif, pos, and ges
    '''
    vcf_file = multifiles_handler(vcf_file)
    preprocessing_vcf38(vcf_file,genome_reference_38_path,genome_reference_19_path,tmp_dir)

    all_preprocessed_vcf = []

    for x in vcf_file:
        if os.path.exists(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            all_preprocessed_vcf.append(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    #pdb.set_trace()
    tokenizing(dict_motif,dict_pos,dict_ges,all_preprocessed_vcf)
    #pdb.set_trace()

def preprocessing_tsv38(tsv_file,genome_reference_38_path,genome_reference_19_path,tmp_dir,verbose=True):
    '''
    Preprocess tsv file with GRCh38
    '''
    genome_ref38 = read_reference(genome_reference_38_path, verbose=verbose)
    genome_ref19 = read_reference(genome_reference_19_path, verbose=verbose)

    fns = multifiles_handler(tsv_file)

    for i, fn in enumerate(fns):
        digits = int(np.ceil(np.log10(len(fns))))
        fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
        get_motif_pos_ges(fn, genome_ref19, tmp_dir, genome_ref38=genome_ref38, liftover=True, verbose=verbose)

def preprocessing_vcf38(vcf_file,genome_reference_38_path,genome_reference_19_path,tmp_dir,verbose=True):
    '''
    Preprocess vcf file with GRCh38
    '''

    # Check if reference files exist
    if not os.path.exists(genome_reference_19_path):
        raise FileNotFoundError(
            "Reference files not found. Please download from:\n"
            "- GRCh37/hg19: https://ftp.sanger.ac.uk/pub/project/PanCancer/genomehg19.fa.gz\n"
            f"and place them in: genome_reference_19_path"
        )

    if not os.path.exists(genome_reference_38_path):
        raise FileNotFoundError(
            "Reference files not found. Please download from:\n"
            "- GRCh38/hg38: http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa\n"
            f"and place them in: genome_reference_38_path"
        )

    genome_ref38 = read_reference(genome_reference_38_path, verbose=verbose)
    genome_ref19 = read_reference(genome_reference_19_path, verbose=verbose)

    fns = multifiles_handler(vcf_file)

    for i, fn in enumerate(fns):
        digits = int(np.ceil(np.log10(len(fns))))
        fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
        get_motif_pos_ges(fn, genome_ref19, tmp_dir, genome_ref38=genome_ref38, liftover=True, verbose=verbose)

def preprocessing_vcf_tokenizing(vcf_file,genome_reference_path,tmp_dir,dict_motif,dict_pos,dict_ges):
    '''
    Preprocess vcf file and tokenize the motif, pos, and ges
    '''
    vcf_file = multifiles_handler(vcf_file)
    preprocessing_vcf(vcf_file,genome_reference_path,tmp_dir)
    #pdb.set_trace()
    all_preprocessed_vcf = []
    for x in vcf_file:
        if os.path.exists(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz'):
            all_preprocessed_vcf.append(tmp_dir + '/' + get_sample_name(x) + '.gc.genic.exonic.cs.tsv.gz')
    tokenizing(dict_motif,dict_pos,dict_ges,all_preprocessed_vcf)
    

def get_motif_pos_ges(fn,genome_ref,tmp_dir,genome_ref38=None,liftover=False,verbose=True):
    """
    Preprocess to get the motif from the vcf file
    Args:
        fn: str, path to vcf file
        genome_ref: reference genome variable from read_reference
        tmp_dir: str, path to temporary directory for storing preprocessed files
        liftover: bool, if True, liftover the vcf file from GRCh38 to GRCh37
    """

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
        raise FileNotFoundError(
            "Reference files not found. Please download from:\n"
            "- GRCh37/hg19: https://ftp.sanger.ac.uk/pub/project/PanCancer/genomehg19.fa.gz\n"
            "- GRCh38/hg38: http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa\n"
            f"and place them in: genome_reference_path"
        )

    genome_ref = read_reference(genome_reference_path,verbose=verbose)   

    fns = multifiles_handler(vcf_file)

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

def tokenizing(dict_motif, dict_pos, dict_ges,all_preprocessed_vcf,pos_bin_size=1000000):
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

            df.to_csv(path, sep='\t',compression='gzip',index=False)
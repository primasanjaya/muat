from .util import *
from .reader import *
import os
import numpy as np
import pdb
import traceback
from tqdm import tqdm
import json

def preprocessing_vcf_tokenizing(vcf_file,genome_reference_path,tmp_dir,dict_motif,dict_pos,dict_ges):
    '''
    Preprocess vcf file and tokenize the motif, pos, and ges
    '''

    preprocessing_vcf(vcf_file,genome_reference_path,tmp_dir)
    


    get_motif_pos_ges(vcf_file,genome_reference_path,tmp_dir)
    create_dictionary(tmp_dir,dict_motif,dict_pos,dict_ges)
    tokenizing(dict_motif,dict_pos,dict_ges,tmp_dir)
    

def get_motif_pos_ges(fn,genome_ref,tmp_dir,verbose=True):
    """
    Preprocess to get the motif from the vcf file
    Args:
        fn: str, path to vcf file
        genome_ref: reference genome variable from read_reference
        tmp_dir: str, path to temporary directory for storing preprocessed files
    """

    try:

        # get motif
        f, sample_name = open_stream(fn)
        vr = get_reader(f)
        status('Writing mutation sequences...', verbose)
        process_input(vr, sample_name, genome_ref,tmp_dir)
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

    # Convert single file path to list for consistent handling
    if isinstance(vcf_file, str):
        vcf_file = [vcf_file]
    
    fns = vcf_file
    is_getmotif = 0
    is_getpos = 0
    is_getges = 0

    for i,fn in enumerate(fns):
        digits = int(np.ceil(np.log10(len(fns))))
        fmt = '{:' + str(digits) + 'd}/{:' + str(digits) + 'd} {}: '
        get_motif_pos_ges(fn,genome_ref,tmp_dir,verbose=verbose)

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
from .util import *
from .reader import *
import os
import numpy as np
import pdb
import traceback


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

    


        









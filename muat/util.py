import sys, gzip, datetime
import os
import errno
import shutil
import tempfile
import subprocess
import pdb

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

def gunzip_file(gz_filename):
    filename = os.path.splitext(gz_filename)[0]  # Remove .gz extension
    if os.name == "nt":  # Windows
        cmd = f'powershell -Command "gzip -d \'{gz_filename}\'"'
    else:  # Linux/macOS
        cmd = f"gunzip -c '{gz_filename}' > '{filename}'"
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
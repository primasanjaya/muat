from .util import *
import numpy as np
import pandas as pd
from natsort import natsort_keygen 
from collections import deque
from pkg_resources import resource_filename
import os, subprocess,re, sys,itertools

import pdb

accepted_pos_h19 = ['1',
'2',
'3',
'4',
'5',
'6',
'7',
'8',
'9',
'10',
'11',
'12',
'13',
'14',
'15',
'16',
'17',
'18',
'19',
'20',
'21',
'22',
'X',
'Y']

try:
    import swalign
    align_scoring = swalign.NucleotideScoringMatrix(2, -1)
    aligner = swalign.LocalAlignment(align_scoring, globalalign=True)
    support_complex = True
except:
    sys.stderr.write('Warning: module swalign not installed: complex variants ignored\n')
    sys.stderr.write('To install swalign: conda install bioconda::swalign\n')
    support_complex = False

re_ext_cigar = re.compile(r'(\d+)([MXID])')

def align(ref, alt, mutation_code):
    alignment = aligner.align(ref, alt)
    ix_r = ix_a = 0
    s = []
    for seg_length, seg_type in re_ext_cigar.findall(alignment.extended_cigar_str):
        seg_length = int(seg_length)
        # seg_type is M, X, D or I
        if seg_type == 'M' or seg_type == 'X':
            s.extend([mutation_code[ref[ix_r + i]][alt[ix_a + i]] for i in range(seg_length)])
            ix_r += seg_length
            ix_a += seg_length
        elif seg_type == 'D':
            s.extend([mutation_code[ref[ix_r + i]]['-'] for i in range(seg_length)])
            ix_r += seg_length
        elif seg_type == 'I':
            s.extend([mutation_code['-'][alt[ix_a + i]] for i in range(seg_length)])
            ix_a += seg_length
        else:
            assert(0)  # invalid seg_type
    return s


def get_reader(f, type_snvs=False,pass_only=True):
    if '.vcf' in f.name:
        vr = VCFReader(f=f, pass_only=pass_only, type_snvs=type_snvs)
    else:
        raise Exception('Unsupported file type: {}\n'.format(f.name))
    return vr

def get_context(v, prev_buf, next_buf, ref_genome,
                mutation_code, reverse_code, context=3):
    """Retrieve sequence context around the focal variant v, incorporate surrounding variants into
    the sequence."""
#    chrom, pos, fref, falt, vtype, _ = mut  # discard sample_id
    #assert(context & (context - 1) == 0)
    flank = (context * 2) // 2 - 1
#    print('get_context', chrom, pos, fref, falt, args.context)
    if v.pos - flank - 1 < 0 or \
        (v.pos + flank >= len(ref_genome[v.chrom])):
        return None
    
    seq = ref_genome[v.chrom][v.pos - flank - 1:v.pos + flank]
#    print('seqlen', len(seq))
    seq = list(seq)
    fpos = len(seq) // 2  # position of the focal mutation
    #for c2, p2, r2, a2, vt2, _ in itertools.chain(prev_buf, next_buf):
    for v2 in itertools.chain(prev_buf, next_buf):
        #if args.nope:
        #    if v.pos != v2.pos or (v.pos == v2.pos and (v.ref != v2.ref or v.alt != v2.alt)):
        #        continue
        # make sure that the mutation stays in the middle of the sequence!
        assert(v2.chrom == v.chrom)
        tp = v2.pos - v.pos + flank
        if tp < 0 or tp >= len(seq):
            continue
#        print(c2, p2, r2, a2, vt2, len(r2), len(a2), len(seq))
        if v2.vtype == Variant.SNV:
            seq[tp] = mutation_code[v2.ref][v2.alt]
        elif v2.vtype == Variant.DEL:
            for i, dc in enumerate(v2.ref):
#                    print('DEL', i, dc, mutation_code[r2[i + 1]]['-'])
                seq[tp] = mutation_code[dc]['-']
                tp += 1
                if tp == len(seq):
                    break
            if v.pos == v2.pos:
#                    print('ADJ, del', fpos, (len(r2) - 1) / 2)
                fpos += len(v2.ref) // 2  # adjust to the deletion midpoint
        elif v2.vtype == Variant.INS:
            seq[tp] = seq[tp] + ''.join([mutation_code['-'][ic] for ic in v2.alt])
            if v2.pos < v.pos:      # adjust to the insertion midpoint
                # v2 is before focal variant - increment position by insertion length
                fpos += len(v2.alt)
            elif v2.pos == v.pos:
                # v2 is the focal variant - increment position by half of insertion length
                fpos += int(np.ceil(1.0 * len(v2.alt) / 2))
        elif v2.vtype == Variant.COMPLEX:
            if support_complex:
                m = align(v2.ref, v2.alt, mutation_code)  # determine mutation sequence
                if len(m) + tp >= len(seq):  # too long allele to fit into the context; increase context length
                    return None
                for i in range(len(m)):  # insert mutation sequence into original
                    seq[tp] = m[i]
                    tp += 1
                n_bp_diff = len(v2.alt) - len(v2.ref)
                if n_bp_diff > 0: # inserted bases? add to the end of the block, insertions are unrolled below
                    seq[tp - 1] = seq[tp - 1] + ''.join(v2.alt[len(v2.ref):])
                # we need to adjust the midpoint according to whether block is before or at the current midpoint
                if v2.pos < v.pos:
                    fpos += max(0, n_bp_diff)
                elif v2.pos == v.pos:
                    fpos += (len(m) - 1) // 2
        elif v2.vtype in Variant.MEI_TYPES:
            seq[tp] = seq[tp] + mutation_code['-'][v2.vtype]
            if v2.pos <= v.pos:  # handle SV breakpoints as insertions
                fpos += 1
        elif v2.vtype in Variant.SV_TYPES:
            seq[tp] = seq[tp] + mutation_code['-'][v2.vtype]
            if v2.pos <= v.pos:  # handle SV breakpoints as insertions
                fpos += 1
        elif v2.vtype == Variant.NOM:
            pass  # no mutation, do nothing (a negative datapoint)
        else:
            raise Exception('Unknown variant type: {}'.format(v2))
    # unroll any insertions and deletions (list of lists -> list)
    seq = [x for sl in list(map(lambda x: list(x), seq)) for x in sl]
    #print('seq2', seq)
    n = len(seq)
    # reverse complement the sequence if the reference base of the substitution is not C or T,
    # or the first inserted/deleted base is not C or T.
    # we transform both nucleotides and mutations here
#    print('UNRL fpos={}, seq={}, f="{}", seqlen={}'.format(fpos, ''.join(seq), seq[fpos], len(seq)))
    lfref, lfalt = len(v.ref), len(v.alt)
    if (lfref == 1 and lfalt == 1 and v.ref in 'AG') or \
       ((v.alt not in Variant.SV_TYPES) and (v.alt not in Variant.MEI_TYPES) and \
            ((lfref > 1 and v.ref[1] in 'AG') or (lfalt > 1 and v.alt[1]))):
        # dna_comp_default returns the input character for non-DNA characters (SV breakpoints)
        seq = [mutation_code[dna_comp_default(reverse_code.get(x)[0])]\
            [dna_comp_default(reverse_code.get(x)[1])] for x in seq][::-1]
        fpos = n - fpos - 1
#        print('REVC', fref, falt, 'fpos={}, seq={}, f="{}", seqlen={}'.format(fpos, ''.join(seq), seq[fpos], len(seq)))
    target_len = 2**int(np.floor(np.log2(context)))
    # trim sequence to length 2^n for max possible n
    #target_len = 2**int(np.floor(np.log2(n)))
    #trim = (n - target_len) / 2.0
    seq = ''.join(seq[max(0, fpos - int(np.floor(target_len / 2))):min(n, fpos + int(np.ceil(target_len / 2)))])
#    print('TRIM seqlen={}, tgtlen={}, seq={}, mid="{}"'.format(len(seq), target_len, ''.join(seq), seq[len(seq) // 2]))
    if len(seq) != target_len:
        return None
    return seq[3:6]

def process_input(vr, sample_name, ref_genome,tmp_dir,genome_ref38=None,liftover=False,verbose=True,context=8):
    """A sweepline algorithm to insert mutations into the sequence flanking the focal mutation."""

    infotag = ''
    report_interval = 1000
    process = []
    output_file = tmp_dir + sample_name + '.tsv.gz'
    ensure_dir_exists(output_file)
    o = gzip.open(output_file, 'wt')
    o.write('chrom\tpos\tref\talt\tsample\tseq{}\n'.format(infotag))

    mutation_code, reverse_code = read_codes()

    warned_invalid_chrom = False
    prev_buf, next_buf = [], []
    i = 0
    n_var = n_flt = n_invalid = n_invalid_chrom = n_ok = 0
    for variant in vr:
        n_var += 1
        if report_interval > 0 and (n_var % report_interval) == 0:
            status('{} variants processed'.format(n_var), True)
        if liftover:
            if variant.chrom not in genome_ref38 and variant.chrom != VariantReader.EOF:
                if warned_invalid_chrom == False:
                    sys.stderr.write('Warning: a chromosome found in data not present in reference: {}\nCheck your reference and vcf file compatibility'.format(variant.chrom))
                    warned_invalid_chrom = True
                    #pdb.set_trace()
                n_invalid_chrom += 1
                continue
        else:
            if variant.chrom not in ref_genome and variant.chrom != VariantReader.EOF:
                if warned_invalid_chrom == False:
                    sys.stderr.write('Warning: a chromosome found in data not present in reference: {}\nCheck your reference and vcf file compatibility'.format(variant.chrom))
                    warned_invalid_chrom = True
                n_invalid_chrom += 1
                continue

        while len(next_buf) > 0 and (next_buf[0].chrom != variant.chrom or next_buf[0].pos < variant.pos - context):
            while len(prev_buf) > 0 and prev_buf[0].pos < next_buf[0].pos - context:
                prev_buf.pop(0)
            if liftover:
                ctx = get_context(next_buf[0], prev_buf, next_buf, genome_ref38,
                            mutation_code, reverse_code,context=context)
            else:
                ctx = get_context(next_buf[0], prev_buf, next_buf, ref_genome,
                            mutation_code, reverse_code,context=context)
            if ctx is not None:
                # mutation not in the end of the chromosome and has full-len context
                next_buf[0].seq = ctx
                o.write(str(next_buf[0]) + '\n')
                n_ok += 1
            else:
                n_invalid += 1
            prev_buf.append(next_buf.pop(0))
        if len(prev_buf) > 0 and prev_buf[0].chrom != variant.chrom:
            prev_buf = []
        if variant.sample_id is None:
            variant.sample_id = sample_name   # id specific on per-file basis
        next_buf.append(variant)
    n_var -= 1  # remove terminator
    if verbose:
        n_all = vr.get_n_accepted() + vr.get_n_filtered()
        sys.stderr.write('{}/{} processed variants, {} filtered, {} invalid, {} missing chromosome\n'.\
            format(n_ok, n_all, vr.get_n_filtered(), n_invalid, n_invalid_chrom))
        sys.stderr.flush()

    o.close()

    #pdb.set_trace()

    pd_motif = pd.read_csv(output_file,sep='\t',low_memory=False,compression='gzip') 
    pd_motif  = pd_motif.sort_values(by=['chrom','pos'],key=natsort_keygen())
    pd_motif.to_csv(output_file,sep='\t',index=False, compression="gzip")
    process.append('motif')

    if liftover:
        from pyliftover import LiftOver

        liftover_chain = resource_filename('muat', 'pkg_data/genomic_tracks/hg38ToHg19.over.chain.gz')
        lo = LiftOver(liftover_chain)
        
        pd_hg38 = pd.read_csv(output_file,sep='\t',low_memory=False) 
        chrom_pos = []

        for i in range(len(pd_hg38)):
            try:
                row = pd_hg38.iloc[i]
                chrom_38 = row['chrom']
                pos_38 = row['pos']
                ref = row['ref']
                alt = row['alt']
                sample = row['sample']
                seq = row['seq']
                #gc1kb = row['gc1kb']
                hg19chrompos = lo.convert_coordinate(chrom_38, pos_38)
                chrom = hg19chrompos[0][0][3:]
                pos = hg19chrompos[0][1]
                chrom_pos.append((chrom,pos,ref,alt,sample,seq,chrom_38,pos_38))
            except:
                print('cant be converted at pos ' +str(row['chrom']) +':'+ str(row['pos']))
        pd_hg19 = pd.DataFrame(chrom_pos)

        #pdb.set_trace()
        pd_hg19.columns = ['chrom','pos','ref','alt','sample','seq','chrom_38','pos_38']
        #natural sort
        pd_hg19 = pd_hg19.loc[pd_hg19['chrom'].isin(accepted_pos_h19)]
        pd_hg19 = pd_hg19.sort_values(by=['chrom','pos'], key=natsort_keygen())
        pd_hg19.to_csv(output_file,sep='\t',index=False, compression="gzip")

     #next gc content
    input_gc = output_file
    output_gc = tmp_dir + sample_name + '.gc.tsv.gz'
    
    pd_sort = pd.read_csv(input_gc,sep='\t',low_memory=False,compression="gzip")
    #remove nan genic
    pd_sort['chrom'] = pd_sort['chrom'].astype('string')
    pd_sort = pd_sort.loc[pd_sort['chrom'].isin(accepted_pos_h19)]
    pd_sort = pd_sort.sort_values(by=['chrom','pos'],key=natsort_keygen())
    pd_sort.to_csv(output_gc,sep='\t',index=False, compression="gzip")
    process.append('gc')

    genic_regions_file = resource_filename('muat', 'pkg_data/genomic_tracks/h37/Homo_sapiens.GRCh37.87.genic.genomic.bed.gz')
    annotate_with_bed_sh = resource_filename('muat', 'pkg_shell/annotate_mutations_with_bed.sh')
    # Make the shell script executable
    #os.chmod(annotate_with_bed_sh, 0o755)
    # Genic region
    output_genic = tmp_dir + sample_name + '.gc.genic.tsv.gz'
    syntax_genic = annotate_with_bed_sh + '\
    ' + output_gc + ' \
    ' + genic_regions_file + ' \
    '+ output_genic + '\
    genic'

    #syntax_test = /Users/primasan/Documents/0a9c9db0-c623-11e3-bf01-24c6515278c0.gc.tsv.gz /Users/primasan/Documents/work/muat/pkg_data/genomic_tracks/h37/Homo_sapiens.GRCh37.87.genic.genomic.bed.gz /Users/primasan/Documents/0a9c9db0-c623-11e3-bf01-24c6515278c0.genic.gc.tsv.gz genic
    
    # Run the shell script and capture the result
    result = subprocess.run(syntax_genic, shell=True, capture_output=True, text=True)
    pd_sort = pd.read_csv(output_genic,sep='\t',low_memory=False,compression="gzip")
    #remove nan genic
    pd_sort = pd_sort[~pd_sort['genic'].isna()]
    pd_sort['chrom'] = pd_sort['chrom'].astype('string')
    pd_sort = pd_sort.loc[pd_sort['chrom'].isin(accepted_pos_h19)]
    pd_sort = pd_sort.sort_values(by=['chrom', 'pos'],key=natsort_keygen())
    pd_sort.to_csv(output_genic,sep='\t',index=False, compression="gzip")
    process.append('genic')

    #exon regions
    output_exon = tmp_dir + sample_name + '.gc.genic.exonic.tsv.gz'
    exonic_regions_file = resource_filename('muat', 'pkg_data/genomic_tracks/h37/Homo_sapiens.GRCh37.87.exons.genomic.bed.gz')

    syntax_exonic = annotate_with_bed_sh + '\
    ' + output_genic + ' \
    ' + exonic_regions_file + ' \
    ' + output_exon + ' \
    exonic'
    subprocess.run(syntax_exonic, shell=True)

    pd_sort = pd.read_csv(output_exon,sep='\t',low_memory=False,compression="gzip") 
    pd_sort = pd_sort[~pd_sort['exonic'].isna()]
    pd_sort['chrom'] = pd_sort['chrom'].astype('string')
    pd_sort = pd_sort.loc[pd_sort['chrom'].isin(accepted_pos_h19)]
    pd_sort = pd_sort.sort_values(by=['chrom', 'pos'],key=natsort_keygen())
    pd_sort.to_csv(output_exon,sep='\t',index=False, compression="gzip")
    process.append('exonic')

    #strand
    output_cs = tmp_dir + sample_name + '.gc.genic.exonic.cs.tsv.gz'
    annotation = resource_filename('muat', 'pkg_data/genomic_tracks/h37/Homo_sapiens.GRCh37.87.transcript_directionality.bed.gz')

    o = openz(output_cs, 'wt')

    hdr = openz(output_exon).readline().strip().decode("utf-8").split('\t')
    n_cols = len(hdr)
    sys.stderr.write('{} columns in input\n'.format(n_cols))
    if hdr[0] == 'chrom':
        sys.stderr.write('Header present\n')
        o.write('{}\tstrand\n'.format('\t'.join(hdr)))
    else:
        sys.stderr.write('Header absent\n')
        hdr = None
    sys.stderr.write('Reading reference: ')
    #reference = read_reference(args.ref, verbose=True)
    # Set the appropriate command based on OS
    if sys.platform == "darwin":  # macOS
        zcat_cmd = "gzcat"
        cmd = "bedmap --sweep-all --faster --delim '\t' --multidelim '\t' --echo --echo-map  <(gzcat {muts}|grep -v \"^chrom\"|awk 'BEGIN{{FS=OFS=\"\t\"}} {{$2 = $2 OFS $2+1}} 1') <(gzcat {annot})".format(annot=annotation, muts=output_exon)
    else:  # Linux
        zcat_cmd = "zcat"
        cmd = "bedmap --sweep-all --faster --delim '\t' --multidelim '\t' --echo --echo-map  <(gunzip -c {muts}|grep -v \"^chrom\"|awk 'BEGIN{{FS=OFS=\"\t\"}} {{$2 = $2 OFS $2+1}} 1') <(zcat {annot})".format(annot=annotation, muts=output_exon)
    #pdb.set_trace()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
    prev_chrom = prev_pos = None
    seen_chroms = set()
    n = 0
    for s in p.stdout:
        v = s.strip().decode("utf-8").split('\t')
        # v is mutation bed + extra columns when the mutation overlaps with a transcript
        # extra columns: chrom,start,end,dirs where dirs is either 1) +, 2) -, 3) +,-
        n_pos = n_neg = 0 
        if len(v) == n_cols + 1:
            pass  # no overlap
        else:
            try:
                strands = v[n_cols + 1 + 3]  # +1 for extra bed END column, +3 to get the strand from [chrom, start, end, strand]
                if strands not in ['+', '-', '+;-']:
                    raise Exception('Unknown strand directionality {} at \n{}'.format(strands, s))
                if strands == '+':
                    n_pos = 1
                elif strands == '-':
                    n_neg = 1
                else:
                    n_pos = n_neg = 1
            except:
                pass

#        n_pos = len(filter(lambda x: x == '+', strands))
#        n_neg = len(filter(lambda x: x == '-', strands))
#        st = None
        #pdb.set_trace()
        if liftover:
            chrom, pos,chrom_38, pos_38  = v[0], int(v[1]), v[7],int(v[8])
        else:
            chrom, pos = v[0], int(v[1])
        ref, alt = v[3], v[4]

        if chrom != prev_chrom:
            if chrom in seen_chroms:
                sys.stderr.write('Input is not sorted (chromosome order): {}:{}\n'.format(chrom, pos))
                sys.exit(1)
            seen_chroms.add(chrom)
            prev_chrom = chrom
        else:
            if pos < prev_pos:
                sys.stderr.write('Input is not sorted (position order): {}:{}\n'.format(chrom, pos))
                sys.exit(1)
        prev_pos = pos
       
        if liftover:
            base = genome_ref38[chrom_38][pos_38]
        else:
            base = ref_genome[chrom][pos]
        #pdb.set_trace()

        if n_pos > 0:
            if n_neg > 0:
                st = '?'
            else:
                if base in ['C', 'T']:
                    st = '+'
                else:
                    st = '-'
        else:
            if n_neg > 0:
                if base in ['C', 'T']:
                    st = '-'
                else:
                    st = '+'
            else:
                st = '='
        o.write('{}\t{}\t{}\t{}\n'.format(chrom, pos, '\t'.join(v[3:n_cols + 1]), st))
        n += 1
        if (n % 1000000) == 0:
            sys.stdout.write('{}: {} mutations written\n'.format(datetime.datetime.now(), n))
    o.flush()
    o.close()
    
    pd_sort = pd.read_csv(output_cs,sep='\t',low_memory=False,compression="gzip") 
    pd_sort = pd_sort[~pd_sort['strand'].isna()]

    pd_sort['chrom'] = pd_sort['chrom'].astype('string')
    pd_sort = pd_sort.loc[pd_sort['chrom'].isin(accepted_pos_h19)]
    pd_sort = pd_sort.sort_values(by=['chrom', 'pos'],key=natsort_keygen())
    preprocessed_mutation = len(pd_sort)
    pd_sort.to_csv(output_cs,sep='\t',index=False, compression="gzip")
    process.append('strand')
    if process == ['motif','gc','genic','exonic','strand']:        #task complete
        try:
            os.remove(tmp_dir + sample_name + '.vcf')
        except:
            pass
        os.remove(tmp_dir + sample_name + '.tsv.gz')
        os.remove(tmp_dir + sample_name + '.gc.tsv.gz')
        os.remove(tmp_dir + sample_name + '.gc.genic.tsv.gz')
        os.remove(tmp_dir + sample_name + '.gc.genic.exonic.tsv.gz')



class Variant(object):
    SNV = 'SNV'        # single-nucleotide variant
    DEL = 'DEL'        # (small) deletion
    INS = 'INS'        # (small) insertion
    COMPLEX = 'CX'     # complex
    SV_DEL = 'SV_DEL'  # deletion
    SV_DUP = 'SV_DUP'  # duplication
    SV_INV = 'SV_INV'  # inversion
    SV_BND = 'SV_BND'  # breakend
    MEI_L1 = 'MEI_L1'  # LINE1 insertion
    MEI_ALU = 'MEI_ALU'# ALU insertion
    MEI_SVA = 'MEI_SVA'# SVA insertion
    MEI_PG = 'MEI_PG'  # Pseudogene insertion (?)
    UNKNOWN = 'UNKNOWN'
    NOM = 'NOM'        # NO Mutation
    SNV_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    INDEL_TYPES = ['DEL', 'INS']
    SV_TYPES = [SV_DEL, SV_DUP, SV_INV, SV_BND]
    MEI_TYPES = [MEI_L1, MEI_ALU, MEI_SVA, MEI_PG]
    ALL_TYPES = [SNV, COMPLEX, UNKNOWN] + INDEL_TYPES + SV_TYPES + MEI_TYPES
    ALL_TYPES_SNV = SNV_TYPES + INDEL_TYPES + SV_TYPES + MEI_TYPES + [COMPLEX, UNKNOWN]
    SVCLASSES = {'DEL' : SV_DEL, 'DUP' : SV_DUP, 'INV' : SV_INV, 'TRA' : SV_BND}
    MEICLASSES = {'L1': MEI_L1, 'Alu': MEI_ALU, 'SVA': MEI_SVA, 'PG': MEI_PG }
    def __init__(self, chrom, pos, ref, alt, vtype=None, sample_id=None,
                 seq=None, extras=None):
        '''`alt` is either nucleotide sequence for SNVs and indels, or one of SV/MEI_TYPES.'''
        self.chrom = chrom
        self.pos = int(pos)
        self.ref = ref
        self.alt = alt
        self.seq = seq
        self.extras = extras
        if vtype is None:
            self.vtype = Variant.variant_type(ref, alt)
        else:
            self.vtype = vtype
        self.sample_id = sample_id
    def __str__(self):
        s = [self.chrom, self.pos, self.ref, self.alt, self.sample_id, self.seq]
        if self.extras is not None:
            s.extend(list(map(str, self.extras)))
        return '\t'.join(map(str, s))
    @staticmethod
    def variant_type(ref, alt, type_snvs=True):
        if len(ref) == 1 and len(alt) == 1:
            if type_snvs:
                if ref in 'AG':
                    ref, alt = dna_comp[ref], dna_comp[alt]
                return '{}>{}'.format(ref, alt)
            else:
                return Variant.SNV
        elif alt in Variant.SV_TYPES:
            return alt
        elif alt in Variant.MEI_TYPES:
            return alt
        elif len(alt) == 0:
            return Variant.DEL
        elif len(ref) == 0:
            return Variant.INS
        else:
            return Variant.COMPLEX
    @staticmethod
    def parse(line):
        '''Parse mutation data generated by process_input()'''
        v = line.strip().split('\t')
        chrom, pos, ref, alt, sample, seq = v[:6]
        extras = v[6:] if len(v) > 6 else None
        return Variant(chrom, pos, ref, alt, None, sample, seq, extras=extras)


class VariantReader(object):
    EOF = 'EOF'  # terminator token to signal end of input
    def __init__(self, f, pass_only=True, type_snvs=False, *args, **kwargs):
        self.f = f
        self.pass_only = pass_only
        self.type_snvs = type_snvs
        self.n_acc = self.n_flt = self.n_ref_alt_equal = 0
        self.eof = False
        self.prev_chrom = self.prev_pos = self.prev_sample = None
        self.seen_samples = set()
        self.seen_chroms = set()
    def get_n_accepted(self):
        return self.n_acc
    def get_n_filtered(self):
        return self.n_flt
    def update_pos(self, sample, chrom, pos):
        #assert(sample is not None and sample != "")
        if sample == self.prev_sample and chrom == self.prev_chrom and pos < self.prev_pos:
            sys.stderr.write('Error: input not sorted by position: {}:{}:{}\n'.format(sample, chrom, pos))
            sys.exit(1)
        if self.prev_sample != sample:
            if sample in self.seen_samples:
                sys.stderr.write('Error: sample already seen in input before: {}:{}:{}\n'.format(sample, chrom, pos))
                sys.exit(1)
            self.seen_samples.add(sample)
            self.seen_chroms = set()
        if self.prev_chrom != chrom:
            if chrom in self.seen_chroms:
                sys.stderr.write('Error: chromosomes not sorted: {}:{}:{}\n'.format(sample, chrom, pos))
                sys.exit(1)
            self.seen_chroms.add(chrom)
        self.prev_sample = sample
        self.prev_chrom = chrom
        self.prev_pos = pos

    @staticmethod
    def format(variant):
        raise NotImplementedError()
    @staticmethod
    def get_file_suffix():
        raise NotImplementedError()
    @staticmethod
    def get_file_sort_cmd(infn, hdrfn, outfn):
        raise NotImplementedError()
    def get_file_header(self):
        raise NotImplementedError()
    def __iter__(self):
        return self


class VCFReader(VariantReader):
    FILTER_PASS = ['.', 'PASS']
    SVCLASS_TO_SVTYPE = {'DEL' : Variant.SV_DEL, 'DUP' : Variant.SV_DUP,
                         't2tINV' : Variant.SV_INV, 't2hINV' : Variant.SV_INV,
                         'h2hINV' : Variant.SV_INV, 'h2tINV' : Variant.SV_INV,
                         'INV' : Variant.SV_INV, 'TRA' : Variant.SV_BND,
                         'L1': Variant.MEI_L1, 'Alu': Variant.MEI_ALU,
                         'SVA': Variant.MEI_SVA, 'PG': Variant.MEI_PG}
    SVTYPE_TO_SVCLASS = {Variant.SV_DEL : 'DEL', Variant.SV_DUP : 'DUP',
                         Variant.SV_INV : 'INV', Variant.SV_BND : 'TRA',
                         Variant.MEI_L1 : 'L1', Variant.MEI_ALU : 'Alu',
                         Variant.MEI_SVA: 'SVA'}


    def __init__(self, *args, **kwargs):
        super(VCFReader, self).__init__(*args, **kwargs)
        self.hdr = None
    def __next__(self):
        while 1:
            if self.eof:
                raise StopIteration()
            v = self.f.readline()
            if v.startswith('#'):
                if v.startswith('#CHROM'):
                    self.hdr = v
                continue
            if v == '':
                self.eof = True
                return Variant(chrom=VariantReader.EOF, pos=0, ref='N', alt='N')
            v = v.rstrip('\n').split('\t')
            chrom, pos, ref, alt, flt, info = v[0], int(v[1]), v[3], v[4], v[6], v[7]

            self.update_pos(None, chrom, pos)
            if self.pass_only and flt not in VCFReader.FILTER_PASS:
                self.n_flt += 1
                continue
            if ref == alt and ref != '':
                ref = alt = ''  # "T>T" -> "(null)>(null)"

            if alt[0] in '[]' or alt[-1] in '[]':  # SV, e.g., ]18:27105494]T
                info = dict([a for a in [c.split('=') for c in info.split(';')] if len(a) == 2])
                svc = info.get('SVCLASS', None)
                if svc is None:
                    sys.stderr.write('Warning: missing SVCLASS: {}:{}\n'.format(chrom, pos))
                    continue
                alt = VCFReader.SVCLASS_TO_SVTYPE.get(svc, None)
                if alt is None:
                    sys.stderr.write('Warning: unknown SVCLASS: {}:{}:{}\n'.format(chrom, pos, svc))
                    continue
            else:
                if is_valid_dna(ref) == False or is_valid_dna(alt) == False:
                    sys.stderr.write('Warning: invalid nucleotide sequence: {}:{}: {}>{}\n'.format(chrom, pos, ref, alt))
                    continue
                # canonize indels by removing anchor bases
                if len(ref) == 1 and len(alt) > 1:   # insertion
                    ref, alt = '', alt[1:]
                elif len(ref) > 1 and len(alt) == 1: # deletion
                    ref, alt = ref[1:], ''
                    pos += 1

            self.n_acc += 1
            # return None as sample id; vcf filename provides the sample id instead
            return Variant(chrom=chrom, pos=pos, ref=ref, alt=alt,
                           vtype=Variant.variant_type(ref, alt, self.type_snvs))

    def get_file_header(self):
        return '{}'.format(self.hdr.rstrip('\n'))

    @staticmethod
    def format(variant):
        "Convert the input variant into a string accepted by this reader"
        if variant.vtype in Variant.SV_TYPES or variant.vtype in Variant.MEI_TYPES:
            info = 'SVCLASS={}'.format(SVTYPE_TO_SVCLASS[variant.vtype])
        else:
            info = ''
        return '{}\t{}\t.\t{}\t{}\t.\t.\t{}\n'.format(variant.chrom, variant.pos, variant.ref, variant.alt, info)

    @staticmethod
    def get_file_suffix():
        return 'vcf'

    @staticmethod
    def get_file_sort_cmd(infn, hdrfn, outfn):
        return "/bin/bash -c \"cat {} <(LC_ALL=C sort -t $'\\t' -k1,1 -k2n,2 {}) >{}\"".format(hdrfn, infn, outfn)

class SomAggTSVReader(VariantReader):
    FILTER_PASS = ['.', 'PASS']
    SVCLASS_TO_SVTYPE = {'DEL' : Variant.SV_DEL, 'DUP' : Variant.SV_DUP,
                         't2tINV' : Variant.SV_INV, 't2hINV' : Variant.SV_INV,
                         'h2hINV' : Variant.SV_INV, 'h2tINV' : Variant.SV_INV,
                         'INV' : Variant.SV_INV, 'TRA' : Variant.SV_BND,
                         'L1': Variant.MEI_L1, 'Alu': Variant.MEI_ALU,
                         'SVA': Variant.MEI_SVA, 'PG': Variant.MEI_PG}
    SVTYPE_TO_SVCLASS = {Variant.SV_DEL : 'DEL', Variant.SV_DUP : 'DUP',
                         Variant.SV_INV : 'INV', Variant.SV_BND : 'TRA',
                         Variant.MEI_L1 : 'L1', Variant.MEI_ALU : 'Alu',
                         Variant.MEI_SVA: 'SVA'}

    def __init__(self, *args, **kwargs):
        super(SomAggTSVReader, self).__init__(*args, **kwargs)
        self.hdr = None

    def __next__(self):
        while True:
            if self.eof:
                raise StopIteration()
            v = self.f.readline()
            
            if v.startswith('PLATEKEY'):
                self.hdr = v
                continue
            if v == '':
                self.eof = True
                return Variant(chrom=VariantReader.EOF, pos=0, ref='N', alt='N')
            
            v = v.rstrip('\n').split('\t')
            sample_id, chrom, pos, ref, alt, flt, info = v[0], v[1], int(v[2]), v[4], v[5], v[7], v[8]

            self.update_pos(None, chrom, pos)
            if self.pass_only and flt not in SomAggTSVReader.FILTER_PASS:
                self.n_flt += 1
                continue
            if ref == alt and ref != '':
                ref = alt = ''  # "T>T" -> "(null)>(null)"

            if alt[0] in '[]' or alt[-1] in '[]':  # SV, e.g., ]18:27105494]T
                info = dict([a for a in [c.split('=') for c in info.split(';')] if len(a) == 2])
                svc = info.get('SVCLASS', None)
                if svc is None:
                    sys.stderr.write('Warning: missing SVCLASS: {}:{}\n'.format(chrom, pos))
                    continue
                alt = SomAggTSVReader.SVCLASS_TO_SVTYPE.get(svc, None)
                if alt is None:
                    sys.stderr.write('Warning: unknown SVCLASS: {}:{}:{}\n'.format(chrom, pos, svc))
                    continue
            else:
                if is_valid_dna(ref) == False or is_valid_dna(alt) == False:
                    sys.stderr.write('Warning: invalid nucleotide sequence: {}:{}: {}>{}\n'.format(chrom, pos, ref, alt))
                    continue
                # canonize indels by removing anchor bases
                if len(ref) == 1 and len(alt) > 1:   # insertion
                    ref, alt = '', alt[1:]
                elif len(ref) > 1 and len(alt) == 1: # deletion
                    ref, alt = ref[1:], ''
                    pos += 1

            self.n_acc += 1
            # return None as sample id; vcf filename provides the sample id instead
            return Variant(chrom=chrom, pos=pos, ref=ref, alt=alt,
                           vtype=Variant.variant_type(ref, alt, self.type_snvs),sample_id=sample_id)

    def get_file_header(self):
        return '{}'.format(self.hdr.rstrip('\n'))

    @staticmethod
    def format(variant):
        "Convert the input variant into a string accepted by this reader"
        if variant.vtype in Variant.SV_TYPES or variant.vtype in Variant.MEI_TYPES:
            info = 'SVCLASS={}'.format(SomAggTSVReader.SVTYPE_TO_SVCLASS[variant.vtype])
        else:
            info = ''
        return '{}\t{}\t.\t{}\t{}\t.\t.\t{}\n'.format(variant.chrom, variant.pos, variant.ref, variant.alt, info)

    @staticmethod
    def get_file_suffix():
        return 'vcf'

    @staticmethod
    def get_file_sort_cmd(infn, hdrfn, outfn):
        return "/bin/bash -c \"cat {} <(LC_ALL=C sort -t $'\\t' -k1,1 -k2n,2 {}) >{}\"".format(hdrfn, infn, outfn)


class MAFReader(VariantReader):
    col_chrom_names = ['chromosome', 'chrom', 'chr']
    col_pos_names = ['position', 'chromosome_start', 'pos']
    col_ref_names = ['mutated_from_allele', 'ref', 'reference_genome_allele']
    col_alt_names = ['mutated_to_allele', 'alt']
    col_filter_names = ['filter']
    col_sample_names = ['sample']

    @staticmethod
    def find_col_ix(names, col_to_ix, fail_on_error=True):
        for n in names:
            if n in col_to_ix:
                return col_to_ix[n]
            if n.capitalize() in col_to_ix:
                return col_to_ix[n.capitalize()]
            if n.upper() in col_to_ix:
                return col_to_ix[n.upper()]
        if fail_on_error:
            sys.stderr.write('Could not find column(s) "{}" in header\n'.format(','.join(names)))
            sys.exit(1)
        else:
            return None

    def __init__(self, extra_columns=None, fake_header=False, *args, **kwargs):
        super(MAFReader, self).__init__(*args, **kwargs)
        if fake_header:
            self.hdr = MAFReader.create_fake_header(kwargs)
        else:
            self.hdr = self.f.readline().rstrip('\n')
        col_to_ix = dict([(x, i) for i, x in enumerate(self.hdr.split('\t'))])
        self.col_chrom_ix = MAFReader.find_col_ix(MAFReader.col_chrom_names, col_to_ix)
        self.col_pos_ix = MAFReader.find_col_ix(MAFReader.col_pos_names, col_to_ix)
        self.col_ref_ix = MAFReader.find_col_ix(MAFReader.col_ref_names, col_to_ix)
        self.col_alt_ix = MAFReader.find_col_ix(MAFReader.col_alt_names, col_to_ix)
        self.col_filter_ix = MAFReader.find_col_ix(MAFReader.col_filter_names, col_to_ix, fail_on_error=False)
        if kwargs['args'].sample_id:
            self.col_sample_ix = MAFReader.find_col_ix([kwargs['args'].sample_id], col_to_ix)
        else:
            self.col_sample_ix = MAFReader.find_col_ix(MAFReader.col_sample_names, col_to_ix)

        if extra_columns:
            self.extra_columns = list(map(col_to_ix.get, extra_columns))
            if None in self.extra_columns:
                raise Exception('Extra column(s) {} not found in input header\n{}'.format(\
                    extra_columns, self.extra_columns))
        else:
            self.extra_columns = []

    def __next__(self):
        while 1:
            if self.eof:
                raise StopIteration()
            v = self.f.readline()
            if v == '':
                self.eof = True
                return Variant(chrom=VariantReader.EOF, pos=0, ref='N', alt='N')
            v = v.rstrip('\n').split('\t')
            try:
                chrom, pos, ref, alt, sample = v[self.col_chrom_ix], \
                    int(v[self.col_pos_ix]), v[self.col_ref_ix], v[self.col_alt_ix], \
                    v[self.col_sample_ix]
                assert(sample != "")
                if self.col_filter_ix is None:
                    flt = 'PASS'
                else:
                    flt = v[self.col_filter_ix]
            except:
                print(v)
                raise
            self.update_pos(sample, chrom, pos)
            if self.pass_only and flt != 'PASS':
                self.n_flt += 1
                continue
            if ref == alt and ref != '':
                ref = alt = ''  # "T>T" -> "(null)>(null)"
            if (alt not in Variant.SV_TYPES) and (alt not in Variant.MEI_TYPES):
                # canonize indels
                if ref == '-' and len(alt) > 0:
                    ref = ''
                elif len(ref) > 0 and alt == '-':
                    alt = ''
            self.n_acc += 1
            extras = [v[ix] for ix in self.extra_columns]
            return Variant(chrom=chrom, pos=pos, ref=ref, alt=alt,
                           vtype=Variant.variant_type(ref, alt, type_snvs=self.type_snvs),
                           sample_id=sample, extras=extras)

    def format(self, variant):
        "Convert the input variant into a string accepted by this reader"
        v = ['.' for _ in range(len(self.hdr.split('\t')))]
        v[self.col_chrom_ix], v[self.col_pos_ix], \
            v[self.col_ref_ix], v[self.col_alt_ix], \
            v[self.col_filter_ix], v[self.col_sample_ix] = \
            variant.chrom, variant.pos, variant.ref, variant.alt, \
            'PASS', variant.sample_id
        for i, ix in enumerate(self.extra_columns):
            v[ix] = variant.info[i]
        return '{}\n'.format('\t'.join(map(str, v)))

    @staticmethod
    def create_fake_header(kwargs):
        if kwargs['args'].sample_id:
            sample_id = kwargs['args'].sample_id
        else:
            sample_id = MAFReader.col_sample_names[0]
        hdr = [MAFReader.col_chrom_names[0], MAFReader.col_pos_names[0],
               MAFReader.col_ref_names[0], MAFReader.col_alt_names[0],
               sample_id, MAFReader.col_filter_names[0]]
        return '\t'.join(hdr)

    @staticmethod
    def get_file_suffix():
        return 'maf'

    def get_file_header(self):
        return self.hdr

    def get_file_sort_cmd(self, infn, hdrfn, outfn, header=False):
        'Sort by sample, and then by chromosome and position'
        if header:
            return "/bin/bash -c \"cat {hdrfn} <(tail -n +2|LC_ALL=C sort -t $'\\t' -k{sample_ix},{sample_ix} -k{chrom_ix},{chrom_ix} -k{pos_ix}g,{pos_ix} {infn}) >{outfn}\"".format(\
                hdrfn=hdrfn, 
                sample_ix=self.col_sample_ix + 1, chrom_ix=self.col_chrom_ix + 1, pos_ix=self.col_pos_ix + 1,
                infn=infn, outfn=outfn)
        else:
            return "/bin/bash -c \"cat {hdrfn} <(LC_ALL=C sort -t $'\\t' -k{sample_ix},{sample_ix} -k{chrom_ix},{chrom_ix} -k{pos_ix}g,{pos_ix} {infn}) >{outfn}\"".format(\
                hdrfn=hdrfn, 
                sample_ix=self.col_sample_ix + 1, chrom_ix=self.col_chrom_ix + 1, pos_ix=self.col_pos_ix + 1,
                infn=infn, outfn=outfn)
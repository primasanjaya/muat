"""Build merged per-sample VCFs (SNV+MNV+indel+SV) for the d1_snvmnvindelsv tag.

Combines three PCAWG-open sources per aliquot UUID into ONE sorted, gzipped VCF that
the existing, UNMODIFIED `muat preprocess --vcf --hg19 ...` CLI can consume as-is:

  - snv_mnv.vcf.gz / indel.vcf.gz (data/PCAWG/consensus_snv_indel/.../snv_mnv|indel):
    already standard VCF, muat.reader.VCFReader parses these natively (indel anchor-base
    stripping is already implemented -- see reader.py's VCFReader.__next__).
  - *.bedpe.gz (data/PCAWG/consensus_sv/icgc/open): PCAWG's structural-variant format,
    which VCFReader does NOT read directly. No new reader class is needed though:
    VCFReader already parses VCF-embedded SV breakend notation (ALT starting/ending with
    a bracket + an INFO SVCLASS= tag), and does not actually parse the bracket's mate-
    position text for classification -- only the SVCLASS tag matters. So each BEDPE row
    is converted into TWO synthetic VCF records (one per breakpoint side, chrom1:start1
    and chrom2:start2), matching how get_context() already treats SV variants as
    single-position breakpoint events, not mated pairs. svclass values observed in this
    cohort (DEL/DUP/TRA/h2hINV/t2tINV) already have exact entries in
    VCFReader.SVCLASS_TO_SVTYPE -- no new mapping needed.

No muat package code is touched by this script -- it only prepares valid merged VCF
inputs; the standard CLI does everything else. Dictionaries are NOT rebuilt: the
shipped defaults (muat/extfile/dict{Mutation,Chpos,GES}.tsv) already carry indel/SV/MEI
vocabulary (1160/233/33 rows respectively) from whatever corpus originally built them --
this cohort's own d1 recipe just never exercised those categories.

Usage:
    python muat/pkg_reproduce/make_snvmnvindelsv_vcfs.py \
        --uuid-list <file, one aliquot UUID per line> \
        --out-dir data/pcawg_full_mutations/vcf
"""
import argparse
import gzip
import os
import sys

from natsort import natsort_keygen

PCAWG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'PCAWG')
SNV_DIR = os.path.join(PCAWG, 'consensus_snv_indel', 'final_consensus_snv_indel_passonly_icgc.public', 'snv_mnv')
INDEL_DIR = os.path.join(PCAWG, 'consensus_snv_indel', 'final_consensus_snv_indel_passonly_icgc.public', 'indel')
SV_DIR = os.path.join(PCAWG, 'consensus_sv', 'icgc', 'open')

ACCEPTED_CHROMS = set(str(c) for c in list(range(1, 23)) + ['X', 'Y'])

VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##INFO=<ID=SVCLASS,Number=1,Type=String,Description=\"SV class, from PCAWG consensus BEDPE\">\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)


def read_vcf_data_lines(path):
    """Yield (chrom, pos, line) for the PASS/'.'-filter data rows of a plain VCF."""
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            v = line.rstrip('\n').split('\t')
            chrom, pos, ref, alt, flt = v[0], int(v[1]), v[3], v[4], v[6]
            if flt not in ('.', 'PASS'):
                continue
            if chrom not in ACCEPTED_CHROMS:
                continue
            yield chrom, pos, '\t'.join([chrom, str(pos), '.', ref, alt, '.', '.', '.'])


def read_sv_as_synthetic_vcf_lines(path):
    """Convert one BEDPE row into two synthetic VCF SV-breakend records (one per side)."""
    with gzip.open(path, 'rt') as f:
        header = f.readline().rstrip('\n').split('\t')
        idx = {name: i for i, name in enumerate(header)}
        for line in f:
            v = line.rstrip('\n').split('\t')
            if len(v) < len(header):
                continue
            svclass = v[idx['svclass']]
            for side in ('1', '2'):
                chrom = v[idx['chrom' + side]]
                # BEDPE start is 0-based; VCF POS is 1-based.
                pos = int(v[idx['start' + side]]) + 1
                if chrom not in ACCEPTED_CHROMS:
                    continue
                # ALT just needs to start/end with a bracket to be detected as an SV row
                # by VCFReader -- the bracket's contents are never parsed for
                # classification, only the INFO SVCLASS tag is.
                info = 'SVCLASS={}'.format(svclass)
                yield chrom, pos, '\t'.join([chrom, str(pos), '.', 'N', ']N:0]N', '.', '.', info])


def build_sample_vcf(uuid, out_dir):
    snv_path = os.path.join(SNV_DIR, '{}.consensus.20160830.somatic.snv_mnv.vcf.gz'.format(uuid))
    indel_path = os.path.join(INDEL_DIR, '{}.consensus.20161006.somatic.indel.vcf.gz'.format(uuid))
    sv_path = os.path.join(SV_DIR, '{}.pcawg_consensus_1.6.161116.somatic.sv.bedpe.gz'.format(uuid))

    for p in (snv_path, indel_path, sv_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    rows = []
    rows.extend(read_vcf_data_lines(snv_path))
    rows.extend(read_vcf_data_lines(indel_path))
    rows.extend(read_sv_as_synthetic_vcf_lines(sv_path))

    key = natsort_keygen()
    rows.sort(key=lambda r: key((r[0], r[1])))

    out_path = os.path.join(out_dir, '{}.vcf.gz'.format(uuid))
    with gzip.open(out_path, 'wt') as f:
        f.write(VCF_HEADER)
        for _, _, line in rows:
            f.write(line + '\n')
    return out_path, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--uuid-list', required=True, help='text file, one aliquot UUID per line')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.uuid_list) as f:
        uuids = [l.strip() for l in f if l.strip()]

    n_ok = n_fail = 0
    for i, uuid in enumerate(uuids):
        try:
            out_path, n_rows = build_sample_vcf(uuid, args.out_dir)
            n_ok += 1
        except FileNotFoundError as e:
            print('SKIP {}: missing source file {}'.format(uuid, e), file=sys.stderr)
            n_fail += 1
            continue
        if (i + 1) % 100 == 0 or (i + 1) == len(uuids):
            print('{}/{} done ({} rows in last sample)'.format(i + 1, len(uuids), n_rows))

    print('done: {} ok, {} failed, output in {}'.format(n_ok, n_fail, args.out_dir))


if __name__ == '__main__':
    main()

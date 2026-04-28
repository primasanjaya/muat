#!/usr/bin/env python3
"""
Generate GRCh38 annotation BED files for muat from Ensembl GTF.

Produces (same format as the existing h37 BEDs):
  Homo_sapiens.GRCh38.87.genic.genomic.bed.gz
  Homo_sapiens.GRCh38.87.exons.genomic.bed.gz
  Homo_sapiens.GRCh38.87.transcript_directionality.bed.gz

Coordinate convention (matches h37 BEDs):
  GTF 1-based inclusive coords are used directly — no 0-base conversion.
  This is consistent with the annotation pipeline where VCF positions are
  also used directly as BED start coords (awk '$2 = $2 OFS $2+1').

Usage:
  python generate_h38_beds.py --gtf Homo_sapiens.GRCh38.87.gtf.gz --out ./h38/
  python generate_h38_beds.py --download --out ./h38/

Dependencies: bgzip, tabix  (available in muat-env)
"""

import sys
import os
import gzip
import argparse
import subprocess
import urllib.request
from collections import defaultdict

ENSEMBL_RELEASE = 87
GTF_URL = (
    f"https://ftp.ensembl.org/pub/release-{ENSEMBL_RELEASE}/gtf/homo_sapiens/"
    f"Homo_sapiens.GRCh38.{ENSEMBL_RELEASE}.gtf.gz"
)

CHROMS = [
    '1','2','3','4','5','6','7','8','9','10','11',
    '12','13','14','15','16','17','18','19','20','21','22','X','Y'
]

# GRCh38 standard chromosome sizes (1-based, = number of bases)
CHROM_SIZES = {
    '1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555,
    '5': 181538259, '6': 170805979, '7': 159345973, '8': 145138636,
    '9': 138394717, '10': 133797422, '11': 135086622, '12': 133275309,
    '13': 114364328, '14': 107043718, '15': 101991189, '16': 90338345,
    '17': 83257441,  '18': 80373285,  '19': 58617616,  '20': 64444167,
    '21': 46709983,  '22': 50818468,  'X': 156040895,  'Y': 57227415,
}


def merge_intervals(intervals):
    """Merge overlapping or adjacent 1-based inclusive intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:  # overlapping or adjacent
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def parse_gtf(gtf_path):
    """
    Parse Ensembl GTF. Returns:
      genes       : dict  chrom -> [(start, end)]           1-based inclusive
      exons       : dict  chrom -> [(start, end)]           1-based inclusive
      transcripts : dict  chrom -> [(start, end, strand)]   1-based inclusive
    """
    genes       = defaultdict(list)
    exons       = defaultdict(list)
    transcripts = defaultdict(list)

    print(f"Parsing GTF: {gtf_path}", flush=True)
    opener = gzip.open if gtf_path.endswith('.gz') else open
    n = 0
    with opener(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 8:
                continue
            chrom   = parts[0]
            feature = parts[2]
            start   = int(parts[3])   # 1-based inclusive, kept as-is
            end     = int(parts[4])   # 1-based inclusive, kept as-is
            strand  = parts[6]

            if chrom not in CHROM_SIZES:
                continue

            if feature == 'gene':
                genes[chrom].append((start, end))
            elif feature == 'exon':
                exons[chrom].append((start, end))
            elif feature == 'transcript':
                transcripts[chrom].append((start, end, strand))

            n += 1
            if n % 1_000_000 == 0:
                print(f"  {n:,} lines parsed...", flush=True)

    print(f"Done. {n:,} feature lines across target chromosomes.", flush=True)
    return genes, exons, transcripts


def write_bgzip_tabix(rows, out_path):
    """Write rows to bgzip file and index with tabix."""
    tmp = out_path[:-3] + '.tmp.bed'   # strip .gz -> .tmp.bed
    with open(tmp, 'w') as f:
        f.writelines(rows)
    subprocess.run(['bgzip', '-f', tmp], check=True)
    os.rename(tmp + '.gz', out_path)
    subprocess.run(['tabix', '-p', 'bed', out_path], check=True)


def make_binary_partition_bed(features, out_path):
    """
    Write a full-genome binary partition BED.
    col5 = 1 inside merged feature intervals, 0 outside.
    Format: chrom start end . 0/1  (coords match GTF 1-based, no conversion)
    The first non-feature region starts at 0 (BED convention for chrom start).
    """
    print(f"Writing {os.path.basename(out_path)} ...", flush=True)
    rows = []
    for chrom in CHROMS:
        size   = CHROM_SIZES[chrom]
        merged = merge_intervals(features.get(chrom, []))
        pos    = 0   # tracks current position (0 = before base 1)

        for start, end in merged:
            if start - 1 > pos:                          # non-feature gap before this feature
                rows.append(f"{chrom}\t{pos}\t{start - 1}\t.\t0\n")
            rows.append(f"{chrom}\t{start}\t{end}\t.\t1\n")
            pos = end + 1

        if pos <= size:                                   # trailing non-feature after last gene
            rows.append(f"{chrom}\t{pos}\t{size}\t.\t0\n")

    write_bgzip_tabix(rows, out_path)
    print(f"  -> {out_path}", flush=True)


def make_directionality_bed(transcripts, out_path):
    """
    Write transcript directionality BED covering only transcribed regions.
    Format: chrom start end strand   (strand: +, -, or +;-)
    Overlapping transcripts with mixed strands get '+;-'.
    """
    print(f"Writing {os.path.basename(out_path)} ...", flush=True)
    rows = []

    for chrom in CHROMS:
        if chrom not in transcripts:
            continue

        # Events: (position, type, idx, strand)
        # type 0 = end (processed before starts at same pos), type 1 = start
        events = []
        for i, (start, end, strand) in enumerate(transcripts[chrom]):
            events.append((start, 1, i, strand))   # start event
            events.append((end,   0, i, strand))   # end event

        events.sort(key=lambda x: (x[0], x[1]))   # end events before starts at same pos

        active_pos = set()   # indices of active + transcripts
        active_neg = set()   # indices of active - transcripts
        prev_pos   = None

        for pos, etype, idx, strand in events:
            # Emit segment [prev_pos, pos] using currently active set
            if prev_pos is not None and prev_pos < pos and (active_pos or active_neg):
                if active_pos and active_neg:
                    st = '+;-'
                elif active_pos:
                    st = '+'
                else:
                    st = '-'
                rows.append(f"{chrom}\t{prev_pos}\t{pos}\t{st}\n")

            if etype == 1:   # start
                if strand == '+':
                    active_pos.add(idx)
                else:
                    active_neg.add(idx)
            else:            # end
                active_pos.discard(idx)
                active_neg.discard(idx)

            prev_pos = pos

    write_bgzip_tabix(rows, out_path)
    print(f"  -> {out_path}", flush=True)


def download_gtf(out_dir):
    dest = os.path.join(out_dir, os.path.basename(GTF_URL))
    if os.path.exists(dest):
        print(f"GTF already exists: {dest}", flush=True)
        return dest
    print(f"Downloading:\n  {GTF_URL}", flush=True)
    urllib.request.urlretrieve(GTF_URL, dest)
    print(f"Saved to {dest}", flush=True)
    return dest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--gtf',      help='Path to Ensembl GRCh38 GTF (.gtf or .gtf.gz)')
    src.add_argument('--download', action='store_true',
                     help=f'Download Ensembl release {ENSEMBL_RELEASE} GTF automatically')
    parser.add_argument('--out', required=True,
                        help='Output directory (created if needed)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gtf_path = download_gtf(args.out) if args.download else args.gtf

    genes, exons, transcripts = parse_gtf(gtf_path)

    prefix = os.path.join(args.out, f"Homo_sapiens.GRCh38.{ENSEMBL_RELEASE}")
    make_binary_partition_bed(genes,       f"{prefix}.genic.genomic.bed.gz")
    make_binary_partition_bed(exons,       f"{prefix}.exons.genomic.bed.gz")
    make_directionality_bed(transcripts,   f"{prefix}.transcript_directionality.bed.gz")

    print(f"\nAll done. Files in: {args.out}")


if __name__ == '__main__':
    main()

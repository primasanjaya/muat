#!/usr/bin/env python
"""One-time fix for the hg38 genic/exonic/transcript_directionality BED tracks
shipped under muat/pkg_data/genomic_tracks/h38/.

Bug found via documentation/README_public_hg38.md's demo cohort: a sample with
5 valid, in-gene variants came out of annotation with zero rows. Root cause:
these three tracks are sorted in plain numeric chromosome order
(1,2,3,...,9,10,11,...,22,X,Y), but BEDOPS `bedmap` requires its own canonical
order (1,10,11,...,19,2,20,21,22,3,4,...,9,X,Y) -- `sort-bed --check-sort`
rejects the shipped files exactly where chromosome "10" starts. bedmap does
NOT error on this; it silently desyncs and returns no overlap, which
reader.py's `pd_sort[~pd_sort['genic'].isna()]` (and the exonic/strand
equivalents) then reads as "not in a gene" and drops -- a silent mutation-loss
bug with no error anywhere in the pipeline. Measured impact: ~0.6% of
mutations lost in aggregate on a 590-sample cohort, up to 100% for individual
low-mutation-count samples (which is what crashed training's DataLoader).

The hg19 equivalents (h37/*.bed.gz) are already correctly sorted and are left
untouched -- they underpin the published, checksum-locked d1-d4 results, and
were not implicated by this bug (this is an hg38_native-only code path; see
`require_sorted_bed()` in reader.py, added alongside this fix as a preflight
guard so a future track regeneration mistake fails loudly instead of
recurring silently).

Also drops the one zero-length interval found in the exons track
(`2  166473892  166473892  .  1`, an off-by-one artifact from however the
track was generated) rather than guessing a replacement coordinate -- there
is no original source file here to re-derive it from, and it is 1 row out of
615,980.

Usage
-----
    python muat/pkg_reproduce/fix_hg38_genomic_tracks.py --check-only   # report, no writes
    python muat/pkg_reproduce/fix_hg38_genomic_tracks.py                # fix in place
"""
import argparse
import gzip
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKS_DIR = os.path.normpath(os.path.join(HERE, "..", "pkg_data", "genomic_tracks", "h38"))
FILES = [
    "Homo_sapiens.GRCh38.87.genic.genomic.bed.gz",
    "Homo_sapiens.GRCh38.87.exons.genomic.bed.gz",
    "Homo_sapiens.GRCh38.87.transcript_directionality.bed.gz",
]


def check_sort(path):
    """Return (is_sorted, message)."""
    result = subprocess.run(
        "gunzip -c {} | sort-bed --check-sort -".format(path),
        shell=True, capture_output=True, text=True)
    return result.returncode == 0, (result.stderr or result.stdout or "").strip()


def fix_one(path):
    n_in = n_dropped = 0
    tmp_plain = path + ".tmp.bed"
    with gzip.open(path, "rt") as fh, open(tmp_plain, "w") as out:
        for line in fh:
            n_in += 1
            f = line.rstrip("\n").split("\t")
            start, end = int(f[1]), int(f[2])
            if end <= start:
                n_dropped += 1
                continue
            out.write(line)

    tmp_sorted = path + ".tmp.sorted.bed"
    subprocess.run("sort-bed {} > {}".format(tmp_plain, tmp_sorted), shell=True, check=True)
    os.remove(tmp_plain)

    tmp_gz = path + ".tmp.bed.gz"
    subprocess.run("bgzip -c {} > {}".format(tmp_sorted, tmp_gz), shell=True, check=True)
    os.remove(tmp_sorted)
    os.replace(tmp_gz, path)

    tbi = path + ".tbi"
    if os.path.exists(tbi):
        os.remove(tbi)
    subprocess.run("tabix -p bed {}".format(path), shell=True, check=True)

    return n_in, n_dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true",
                    help="report sort status of each track without modifying anything")
    args = ap.parse_args()

    any_bad = False
    for fname in FILES:
        path = os.path.join(TRACKS_DIR, fname)
        ok, msg = check_sort(path)
        print("{:60s} {}".format(fname, "OK" if ok else "NOT SORTED / MALFORMED"))
        if not ok:
            any_bad = True
            print("  " + msg.replace("\n", "\n  "))

    if args.check_only:
        sys.exit(1 if any_bad else 0)

    for fname in FILES:
        path = os.path.join(TRACKS_DIR, fname)
        ok, _ = check_sort(path)
        if ok:
            continue
        n_in, n_dropped = fix_one(path)
        ok_after, msg_after = check_sort(path)
        print("{}: {} rows in, {} malformed dropped, re-sorted -> {}".format(
            fname, n_in, n_dropped, "OK" if ok_after else "STILL BAD: " + msg_after))
        if not ok_after:
            raise SystemExit("fix did not converge for {}".format(path))


if __name__ == "__main__":
    main()

"""Quantify hg38->hg19 liftover ambiguity for SV/MEI breakpoints (reviewer R1-minor-6).

Reviewer 1 asked us to "specify whether preprocessing from hg38 to hg19 could
introduce coordinate ambiguities for SVs and MEIs." muat's actual production
liftover step (reader.py, `process_input(..., liftover=True)`) lifts hg38 input
DOWN to hg19 using the shipped chain `muat/pkg_data/genomic_tracks/
hg38ToHg19.over.chain.gz`, then runs the hg19-trained checkpoint. We have no
genuinely hg38-native somatic SV/MEI calls available open-access (TCGA's public
MAFs are SNV-only; TCGA's real Manta SV VCFs and Genomics England's SV calls are
both controlled-access) -- see the muat-revision discussion. So this script
answers the question with a ROUND-TRIP liftover instead of needing new data:

    real PCAWG hg19 breakpoint --[hg19->hg38, UCSC chain]--> hg38 coordinate
                               --[hg38->hg19, muat's OWN shipped chain]--> hg19'

The first leg only exists to synthesize a plausible hg38 starting point (PCAWG's
SV calls are natively hg19 -- see muat-d1-snvmnvindelsv). The SECOND leg is the
literal chain file and direction muat's own preprocessing code uses in
production, so failures/multi-mapping/drift on that leg are a direct measurement
of the reviewer's concern, not a proxy.

Per breakpoint SIDE (a BEDPE row has two: chrom1/start1, chrom2/start2), classify:
  FWD_FAIL / FWD_MULTI   -- hg19->hg38 leg failed / landed on >1 chain (synthesis
                             step only; not itself a muat production concern)
  REV_FAIL / REV_MULTI   -- hg38->hg19 leg (muat's real chain) failed / ambiguous
  CHROM_MISMATCH         -- round-tripped back to a different chromosome
  EXACT / NEAR / DRIFT   -- round-tripped position ==, within NEAR_BP of, or
                             farther than NEAR_BP from the original real hg19 pos

ANNOTATION EXTENSION: for every breakpoint whose round-trip was NOT a clean EXACT
match, also queries muat's own shipped hg19 (h37) genic/exonic/transcript-strand
BED tracks -- the SAME tracks `reader.py` annotates real mutations with -- at the
original position and (when one exists) the round-tripped position, via `tabix`
point queries (equivalent to bedmap's point-overlap semantics, just index-based
instead of a sweep). This answers the sharper question the coordinate-drift
numbers alone can't: does the ambiguity actually CHANGE what muat's GES tokenizer
would have annotated, or does it just drift within the same (e.g. intergenic)
region? NOTE: this does not replicate reader.py's SNV-specific pyrimidine/purine
strand-folding (that needs a reference base, meaningless for a bare breakpoint) --
it reports the raw genic flag / exonic flag / transcript-strand overlap, which is
the genuinely comparable, coordinate-only piece of that annotation.

Usage:
    python muat/pkg_reproduce/liftover_sv_qc.py \
        --bedpe-glob 'data/PCAWG/consensus_sv/icgc/open/*.bedpe.gz' \
        --forward-chain data/liftover_qc/chains/hg19ToHg38.over.chain.gz \
        --out-dir data/liftover_qc
    python muat/pkg_reproduce/liftover_sv_qc.py --limit 20   # quick smoke test
"""
import argparse
import glob
import gzip
import os
import subprocess

from pyliftover import LiftOver

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BEDPE_GLOB = os.path.join(REPO, "data/PCAWG/consensus_sv/icgc/open/*.bedpe.gz")
DEFAULT_FORWARD_CHAIN = os.path.join(REPO, "data/liftover_qc/chains/hg19ToHg38.over.chain.gz")
DEFAULT_REVERSE_CHAIN = os.path.join(REPO, "muat/pkg_data/genomic_tracks/hg38ToHg19.over.chain.gz")
DEFAULT_OUT_DIR = os.path.join(REPO, "data/liftover_qc")
DEFAULT_GENIC_BED = os.path.join(REPO, "muat/pkg_data/genomic_tracks/h37/Homo_sapiens.GRCh37.87.genic.genomic.bed.gz")
DEFAULT_EXONIC_BED = os.path.join(REPO, "muat/pkg_data/genomic_tracks/h37/Homo_sapiens.GRCh37.87.exons.genomic.bed.gz")
DEFAULT_STRAND_BED = os.path.join(REPO, "muat/pkg_data/genomic_tracks/h37/Homo_sapiens.GRCh37.87.transcript_directionality.bed.gz")

NEAR_BP = 5  # small round-trip shifts (a few bp) are a known liftOver/assembly-patch
             # artifact, not a real ambiguity -- only flag drift beyond this as DRIFT
# Categories worth an annotation lookup -- EXACT (359,865/359,946 breakpoints) is
# skipped: identical position guarantees identical annotation, and re-querying it
# would dominate runtime for zero information.
ANNOTATE_CATEGORIES = {"NEAR", "DRIFT", "CHROM_MISMATCH", "REV_FAIL", "REV_MULTI", "FWD_FAIL", "FWD_MULTI"}
# Categories where a valid, different round-tripped hg19 position exists to compare against.
HAS_BACK_POSITION = {"NEAR", "DRIFT", "CHROM_MISMATCH"}


class TabixAnnotator:
    """Point-overlap lookups against a tabix-indexed BED, mirroring bedmap's
    point-overlap semantics (query = the single 0-based base [pos, pos+1))."""

    def __init__(self, genic_bed, exonic_bed, strand_bed):
        self.genic_bed = genic_bed
        self.exonic_bed = exonic_bed
        self.strand_bed = strand_bed
        self._cache = {}

    def _query(self, bed_path, chrom, pos):
        key = (bed_path, chrom, pos)
        if key in self._cache:
            return self._cache[key]
        region = "{}:{}-{}".format(chrom, pos + 1, pos + 1)  # tabix region is 1-based inclusive
        try:
            out = subprocess.check_output(["tabix", bed_path, region], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            out = b""
        rows = [r.split("\t") for r in out.decode("utf-8").splitlines() if r]
        self._cache[key] = rows
        return rows

    def genic(self, chrom, pos):
        rows = self._query(self.genic_bed, chrom, pos)
        return rows[0][4] if rows else "NA"

    def exonic(self, chrom, pos):
        rows = self._query(self.exonic_bed, chrom, pos)
        return rows[0][4] if rows else "NA"

    def strand(self, chrom, pos):
        rows = self._query(self.strand_bed, chrom, pos)
        vals = sorted({r[3] for r in rows})
        if not vals:
            return "none"
        if len(vals) > 1:
            return "+;-"
        return vals[0]

    def annotate(self, chrom, pos):
        return self.genic(chrom, pos), self.exonic(chrom, pos), self.strand(chrom, pos)


def _chr(chrom):
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else "chr" + chrom


def _unchr(chrom):
    return chrom[3:] if chrom.startswith("chr") else chrom


def iter_breakpoint_sides(bedpe_path):
    """Yield (sample, sv_id, svclass, side, chrom, pos) for both sides of every row.

    `pos` is BEDPE's 0-based `start`, matching pyliftover's expected 0-based input.
    """
    sample = os.path.basename(bedpe_path).split(".")[0]
    opener = gzip.open if bedpe_path.endswith(".gz") else open
    with opener(bedpe_path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in fh:
            row = line.rstrip("\n").split("\t")
            sv_id = row[idx["sv_id"]]
            svclass = row[idx["svclass"]]
            yield (sample, sv_id, svclass, 1, row[idx["chrom1"]], int(row[idx["start1"]]))
            yield (sample, sv_id, svclass, 2, row[idx["chrom2"]], int(row[idx["start2"]]))


def classify(lo_fwd, lo_rev, chrom, pos):
    """Returns (category, hg38_chrom, hg38_pos, back_chrom, back_pos).

    back_chrom/back_pos (the round-tripped hg19 coordinate, bare-chrom form) are
    only meaningful for NEAR/DRIFT/CHROM_MISMATCH -- None otherwise.
    """
    fwd = lo_fwd.convert_coordinate(_chr(chrom), pos)
    if not fwd:
        return "FWD_FAIL", None, None, None, None
    if len(fwd) > 1:
        return "FWD_MULTI", None, None, None, None
    hg38_chrom, hg38_pos, _, _ = fwd[0]

    rev = lo_rev.convert_coordinate(hg38_chrom, hg38_pos)
    if not rev:
        return "REV_FAIL", hg38_chrom, hg38_pos, None, None
    if len(rev) > 1:
        return "REV_MULTI", hg38_chrom, hg38_pos, None, None
    back_chrom, back_pos, _, _ = rev[0]
    back_chrom = _unchr(back_chrom)

    if back_chrom != _unchr(chrom):
        return "CHROM_MISMATCH", hg38_chrom, hg38_pos, back_chrom, back_pos

    drift = abs(back_pos - pos)
    if drift == 0:
        return "EXACT", hg38_chrom, hg38_pos, back_chrom, back_pos
    if drift <= NEAR_BP:
        return "NEAR", hg38_chrom, hg38_pos, back_chrom, back_pos
    return "DRIFT", hg38_chrom, hg38_pos, back_chrom, back_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bedpe-glob", default=DEFAULT_BEDPE_GLOB)
    ap.add_argument("--forward-chain", default=DEFAULT_FORWARD_CHAIN,
                     help="hg19->hg38 chain, used ONLY to synthesize a starting hg38 "
                          "coordinate since PCAWG's SV calls are natively hg19")
    ap.add_argument("--reverse-chain", default=DEFAULT_REVERSE_CHAIN,
                     help="hg38->hg19 chain -- muat's OWN shipped production chain")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N bedpe files (smoke test)")
    ap.add_argument("--no-annotate", action="store_true",
                     help="skip the genic/exonic/strand annotation-flip check (liftover metrics only)")
    ap.add_argument("--genic-bed", default=DEFAULT_GENIC_BED)
    ap.add_argument("--exonic-bed", default=DEFAULT_EXONIC_BED)
    ap.add_argument("--strand-bed", default=DEFAULT_STRAND_BED)
    args = ap.parse_args()

    for path, label in ((args.forward_chain, "forward"), (args.reverse_chain, "reverse")):
        if not os.path.isfile(path):
            raise SystemExit("missing {} chain: {}".format(label, path))

    lo_fwd = LiftOver(args.forward_chain)
    lo_rev = LiftOver(args.reverse_chain)
    annotator = None
    if not args.no_annotate:
        for path, label in ((args.genic_bed, "genic"), (args.exonic_bed, "exonic"), (args.strand_bed, "strand")):
            if not os.path.isfile(path):
                raise SystemExit("missing {} BED: {}".format(label, path))
        annotator = TabixAnnotator(args.genic_bed, args.exonic_bed, args.strand_bed)

    files = sorted(glob.glob(args.bedpe_glob))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("no bedpe files matched {}".format(args.bedpe_glob))

    os.makedirs(args.out_dir, exist_ok=True)
    detail_path = os.path.join(args.out_dir, "liftover_sv_qc_detail.tsv")
    counts = {}          # category -> count
    counts_by_class = {}  # (svclass, category) -> count
    annot_rows = []       # rows needing an annotation-flip report

    detail_cols = ["sample", "sv_id", "svclass", "side", "chrom", "pos", "category", "hg38_chrom", "hg38_pos"]
    if annotator is not None:
        detail_cols += ["back_chrom", "back_pos",
                         "genic_before", "exonic_before", "strand_before",
                         "genic_after", "exonic_after", "strand_after", "annotation_changed"]

    with open(detail_path, "w") as out:
        out.write("\t".join(detail_cols) + "\n")
        for fi, path in enumerate(files, 1):
            for sample, sv_id, svclass, side, chrom, pos in iter_breakpoint_sides(path):
                category, hg38_chrom, hg38_pos, back_chrom, back_pos = classify(lo_fwd, lo_rev, chrom, pos)
                counts[category] = counts.get(category, 0) + 1
                counts_by_class[(svclass, category)] = counts_by_class.get((svclass, category), 0) + 1
                row = [sample, sv_id, svclass, str(side), chrom, str(pos), category,
                       hg38_chrom or "", str(hg38_pos) if hg38_pos is not None else ""]

                if annotator is not None:
                    if category in ANNOTATE_CATEGORIES:
                        g0, e0, s0 = annotator.annotate(chrom, pos)
                        if category in HAS_BACK_POSITION:
                            g1, e1, s1 = annotator.annotate(back_chrom, back_pos)
                            changed = (g0, e0, s0) != (g1, e1, s1)
                        else:
                            g1 = e1 = s1 = "NA"  # no valid round-tripped position to compare (REV_FAIL/*_MULTI/FWD_FAIL)
                            changed = ""         # not applicable, not "unchanged"
                        annot_rows.append({
                            "svclass": svclass, "category": category,
                            "genic_before": g0, "exonic_before": e0, "strand_before": s0,
                            "genic_after": g1, "exonic_after": e1, "strand_after": s1,
                            "annotation_changed": changed,
                        })
                        row += [back_chrom or "", str(back_pos) if back_pos is not None else "",
                                g0, e0, s0, g1, e1, s1, str(changed)]
                    else:
                        row += [""] * 9

                out.write("\t".join(row) + "\n")
            if fi % 200 == 0:
                print("... {}/{} files".format(fi, len(files)))

    total = sum(counts.values())
    summary_path = os.path.join(args.out_dir, "liftover_sv_qc_summary.tsv")
    with open(summary_path, "w") as out:
        out.write("category\tcount\tfraction\n")
        for cat in ["EXACT", "NEAR", "DRIFT", "CHROM_MISMATCH", "REV_FAIL", "REV_MULTI", "FWD_FAIL", "FWD_MULTI"]:
            c = counts.get(cat, 0)
            out.write("{}\t{}\t{:.6f}\n".format(cat, c, c / total if total else 0.0))
        out.write("TOTAL\t{}\t1.0\n".format(total))

    by_class_path = os.path.join(args.out_dir, "liftover_sv_qc_by_svclass.tsv")
    svclasses = sorted({sv for sv, _ in counts_by_class})
    categories = ["EXACT", "NEAR", "DRIFT", "CHROM_MISMATCH", "REV_FAIL", "REV_MULTI", "FWD_FAIL", "FWD_MULTI"]
    with open(by_class_path, "w") as out:
        out.write("svclass\t" + "\t".join(categories) + "\ttotal\n")
        for sv in svclasses:
            row = [counts_by_class.get((sv, c), 0) for c in categories]
            out.write(sv + "\t" + "\t".join(str(x) for x in row) + "\t{}\n".format(sum(row)))

    print("files processed :", len(files))
    print("breakpoint sides:", total)
    for cat in ["EXACT", "NEAR", "DRIFT", "CHROM_MISMATCH", "REV_FAIL", "REV_MULTI", "FWD_FAIL", "FWD_MULTI"]:
        c = counts.get(cat, 0)
        print("  {:<15s} {:>8d}  ({:.3%})".format(cat, c, c / total if total else 0.0))
    print("wrote:", detail_path)
    print("wrote:", summary_path)
    print("wrote:", by_class_path)

    if annotator is not None:
        annot_path = os.path.join(args.out_dir, "liftover_sv_qc_annotation.tsv")
        comparable = [r for r in annot_rows if r["category"] in HAS_BACK_POSITION]
        changed = [r for r in comparable if r["annotation_changed"]]
        with open(annot_path, "w") as out:
            out.write("category\tn_queried\tn_comparable\tn_annotation_changed\tfraction_changed\n")
            for cat in sorted({r["category"] for r in annot_rows}):
                cat_rows = [r for r in annot_rows if r["category"] == cat]
                cat_comparable = [r for r in cat_rows if cat in HAS_BACK_POSITION]
                cat_changed = [r for r in cat_comparable if r["annotation_changed"]]
                frac = len(cat_changed) / len(cat_comparable) if cat_comparable else float("nan")
                out.write("{}\t{}\t{}\t{}\t{:.4f}\n".format(
                    cat, len(cat_rows), len(cat_comparable), len(cat_changed), frac))
        print()
        print("annotation-flip check (genic/exonic/strand at muat's own hg19 tracks):")
        print("  breakpoints queried    :", len(annot_rows))
        print("  comparable (before+after):", len(comparable))
        print("  annotation ACTUALLY changed:", len(changed),
              "({:.3%} of comparable)".format(len(changed) / len(comparable) if comparable else 0.0))
        print("wrote:", annot_path)


if __name__ == "__main__":
    main()

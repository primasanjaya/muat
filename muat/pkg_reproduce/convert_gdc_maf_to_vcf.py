#!/usr/bin/env python
"""Convert per-aliquot GDC 'masked somatic mutation' MAF.gz files (from
fetch_public_hg38_cohort.py) into minimal per-sample VCFs that muat's VCFReader
accepts directly via `muat preprocess --vcf --hg38 <ref>` (native, no
--liftover -- see documentation/README_public_hg38.md).

Scope limitation (measured, not silent): only SNP rows (single-base REF/ALT)
are converted. MAF's insertion/deletion convention needs the reference FASTA
to recover the shared anchor base that VCF requires, which this script does
not do; on a 6-project TCGA cohort this dropped indels (~3.6%) and stray
multi-base substitutions (3 rows out of ~160k). Both counts are printed so the
loss is visible, not swallowed.

Usage
-----
    python muat/pkg_reproduce/convert_gdc_maf_to_vcf.py \\
        --manifest data/hg38_public_demo/gdc_manifest.json \\
        --maf-dir data/hg38_public_demo/gdc_maf \\
        --out-dir data/hg38_public_demo/vcf \\
        --labels-out data/hg38_public_demo/labels.json
"""
import argparse
import gzip
import json
import os
import sys

KEEP_TYPES = {"SNP"}
VCF_HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def convert_one(maf_path, out_path):
    """Return (n_kept, n_skipped_indel, n_skipped_other, n_skipped_nonsomatic)."""
    stats = [0, 0, 0, 0]
    with gzip.open(maf_path, "rt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            header = line.rstrip("\n").split("\t")
            break
        if header is None:
            print("EMPTY MAF (no header):", maf_path, file=sys.stderr)
            return (0, 0, 0, 0)
        col = {name: i for i, name in enumerate(header)}
        need = ["Chromosome", "Start_Position", "Reference_Allele",
                "Tumor_Seq_Allele2", "Variant_Type", "Mutation_Status"]
        for n in need:
            if n not in col:
                raise SystemExit("missing column {} in {}".format(n, maf_path))

        rows = []
        for line in f:
            v = line.rstrip("\n").split("\t")
            vtype = v[col["Variant_Type"]]
            mstatus = v[col["Mutation_Status"]]
            if mstatus and mstatus != "Somatic":
                stats[3] += 1
                continue
            chrom = v[col["Chromosome"]]
            pos = int(v[col["Start_Position"]])
            ref = v[col["Reference_Allele"]]
            alt = v[col["Tumor_Seq_Allele2"]]
            if vtype not in KEEP_TYPES or len(ref) != 1 or len(alt) != 1 or ref == alt:
                if vtype in ("INS", "DEL"):
                    stats[1] += 1
                else:
                    stats[2] += 1
                continue
            rows.append((chrom, pos, ref, alt))
            stats[0] += 1

    # Sort by chrom (grouped, any stable order) then pos ascending: VCFReader
    # requires no chrom to be revisited once left, and ascending pos within one.
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(out_path, "w") as out:
        out.write(VCF_HEADER)
        for chrom, pos, ref, alt in rows:
            out.write("{}\t{}\t.\t{}\t{}\t.\tPASS\t.\n".format(chrom, pos, ref, alt))
    return tuple(stats), len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="gdc_manifest.json from the fetch step")
    ap.add_argument("--maf-dir", required=True, help="directory of downloaded <file_id>.maf.gz")
    ap.add_argument("--out-dir", required=True, help="directory to write per-sample VCFs into")
    ap.add_argument("--labels-out", required=True,
                    help="output JSON: sample_id/vcf_path/project/case_id/n_variants per sample")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = json.load(open(args.manifest))

    totals = [0, 0, 0, 0]
    labels = []
    for m in manifest:
        maf_path = os.path.join(args.maf_dir, m["file_id"] + ".maf.gz")
        sample_id = m["submitter_id"].replace(":", "_")
        # Absolute, regardless of --out-dir: muat train/predict are typically invoked from a
        # neutral directory (to avoid shadowing an installed package with repo source), so a
        # relative path recorded here would silently fail to resolve at train/predict time.
        out_path = os.path.abspath(os.path.join(args.out_dir, sample_id + ".vcf"))
        stats, n_rows = convert_one(maf_path, out_path)
        for i in range(4):
            totals[i] += stats[i]
        labels.append({
            "sample_id": sample_id,
            "vcf_path": out_path,
            "project": m["project"],
            "case_id": m["case_id"],
            "n_variants": n_rows,
        })

    with open(args.labels_out, "w") as f:
        json.dump(labels, f, indent=1)

    zero_variant = sum(1 for l in labels if l["n_variants"] == 0)
    print("kept={} skipped_indel={} skipped_other={} skipped_nonsomatic={}".format(*totals))
    print("wrote {} per-sample VCFs to {} ({} have zero variants after filtering -- "
          "drop these before building splits)".format(len(labels), args.out_dir, zero_variant))


if __name__ == "__main__":
    main()

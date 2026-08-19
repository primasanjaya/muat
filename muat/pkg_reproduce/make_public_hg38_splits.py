#!/usr/bin/env python
"""Deterministically split the public hg38 demo cohort (see
documentation/README_public_hg38.md) into patient-level train/val/test sets.

Each sample in labels.json is already exactly one distinct GDC case (dedup'd by
fetch_public_hg38_cohort.py), so there is no cross-split patient leakage to
guard against beyond dropping zero-variant samples (a sample can end up with
zero rows after convert_gdc_maf_to_vcf.py's SNP-only filtering).

Splitting is per-class stratified with a fixed seed (pure-Python
``random.Random``, no numpy) so re-running this on the same labels.json
reproduces byte-identical split files.

Outputs
-------
all_vcfs.txt
    every kept sample's raw VCF path, one per line -- the --input-list for the
    single `muat preprocess --vcf --hg38 ... --build-dictionary` call that must
    see the whole cohort at once.
{train,val,test}_split.tsv
    columns: prep_path (=<tokenized-dir>/<sample_id>.muat.tsv) / class_name /
    class_index -- ready for `muat train from-scratch` / `muat predict`.
class_index.json
    project -> class_index mapping used above.

Usage
-----
    python muat/pkg_reproduce/make_public_hg38_splits.py \\
        --labels data/hg38_public_demo/labels.json \\
        --tokenized-dir data/hg38_public_demo/preprocessed \\
        --out-dir data/hg38_public_demo \\
        --seed 1337 --train-frac 0.7 --val-frac 0.15
"""
import argparse
import csv
import hashlib
import json
import os
import random


def _stratified_split(rows, seed, train_frac, val_frac):
    """rows: list of dict with at least 'project'. Returns (train, val, test)."""
    by_class = {}
    for r in rows:
        by_class.setdefault(r["project"], []).append(r)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls in sorted(by_class):
        members = sorted(by_class[cls], key=lambda r: r["sample_id"])  # stable input order
        rng.shuffle(members)
        n = len(members)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train.extend(members[:n_train])
        val.extend(members[n_train:n_train + n_val])
        test.extend(members[n_train + n_val:])
        print("{}: total={} train={} val={} test={}".format(
            cls, n, n_train, n_val, n - n_train - n_val))
    return train, val, test


def _write_split(path, members, class_index, tokenized_dir):
    members = sorted(members, key=lambda r: (class_index[r["project"]], r["sample_id"]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["prep_path", "class_name", "class_index"])
        for r in members:
            # Absolute regardless of --tokenized-dir: muat train/predict are typically
            # invoked from a neutral directory (to avoid shadowing an installed package
            # with repo source), so a relative path here would silently fail to resolve.
            prep_path = os.path.abspath(os.path.join(tokenized_dir, r["sample_id"] + ".muat.tsv"))
            w.writerow([prep_path, r["project"], class_index[r["project"]]])
    return path


def _sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="labels.json from convert_gdc_maf_to_vcf.py")
    ap.add_argument("--tokenized-dir", required=True,
                    help="directory the preprocess/tokenize step will write <sample_id>.muat.tsv into")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    n_zero = sum(1 for l in labels if l["n_variants"] == 0)
    rows = [l for l in labels if l["n_variants"] > 0]
    if n_zero:
        print("dropping {} zero-variant sample(s) after SNP-only filtering".format(n_zero))

    projects = sorted(set(r["project"] for r in rows))
    class_index = {p: i for i, p in enumerate(projects)}
    print("classes:", class_index)

    train, val, test = _stratified_split(rows, args.seed, args.train_frac, args.val_frac)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "all_vcfs.txt"), "w") as f:
        for r in rows:
            f.write(r["vcf_path"] + "\n")
    with open(os.path.join(args.out_dir, "class_index.json"), "w") as f:
        json.dump(class_index, f, indent=1)

    outs = {}
    for name, members in (("train", train), ("val", val), ("test", test)):
        path = os.path.join(args.out_dir, "{}_split.tsv".format(name))
        outs[name] = _write_split(path, members, class_index, args.tokenized_dir)

    print("total used: {} of {}".format(len(rows), len(labels)))
    for which, path in outs.items():
        print("  {:5} -> {}  sha256={}".format(which, os.path.basename(path), _sha256(path)[:16]))


if __name__ == "__main__":
    main()

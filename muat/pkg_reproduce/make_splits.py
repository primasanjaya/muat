#!/usr/bin/env python
"""Deterministically regenerate the Group-D (open-access PCAWG) reproduce splits.

The split is *defined by code + a fixed seed*, not by a frozen mystery file: given
the same `pcawg_open_labels.tsv` and `--seed`, this writes bit-identical
train/val/test TSVs on any platform (pure-Python ``random.Random`` Mersenne
Twister, no numpy). That is the reproducibility guarantee for `muat reproduce`.

Inputs
------
pcawg_open_labels.tsv : the frozen ground-truth label manifest
    columns: sample <tab> class_name <tab> class_index
    (`sample` is the PCAWG aliquot UUID; this file *is* the benchmark universe.)

Outputs (written next to this script, under splits/)
---------------------------------------------------
pcawg_open_train.tsv / pcawg_open_val.tsv / pcawg_open_test.tsv
    columns: prep_path <tab> class_name <tab> class_index
    `prep_path` is the basename only (``<sample>.muat.tsv``); `muat reproduce`
    remaps it to <cache>/<bundle>/<basename> at run time, so the files stay
    portable across machines.

Usage
-----
    python muat/pkg_reproduce/make_splits.py            # seed 1337, 80/10/10
    python muat/pkg_reproduce/make_splits.py --seed 1337 --test-frac 0.1 --val-frac 0.1
"""
import argparse
import csv
import hashlib
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(HERE, "splits")
LABELS = os.path.join(SPLITS_DIR, "pcawg_open_labels.tsv")

# The preprocessed file shipped in the data bundle, per sample. The bundle uses
# the friendly ``.muat.tsv`` name (plain TSV) rather than the long internal
# ``.token.gc.genic.exonic.cs.tsv.gz``; see make_bundle.py.
SAMPLE_SUFFIX = ".muat.tsv"


def _read_labels(path):
    """Return list of (sample, class_name, class_index), sorted deterministically."""
    rows = []
    with open(path) as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            rows.append((row["sample"], row["class_name"], int(row["class_index"])))
    # canonical input order: (class_index, sample) — independent of file order
    return sorted(rows, key=lambda t: (t[2], t[0]))


def _stratified_split(rows, seed, test_frac, val_frac):
    """Per-class stratified split. Deterministic given (rows, seed)."""
    by_class = {}
    for sample, name, idx in rows:
        by_class.setdefault(idx, []).append((sample, name, idx))

    rng = random.Random(seed)
    train, val, test = [], [], []
    for idx in sorted(by_class):
        members = sorted(by_class[idx], key=lambda t: t[0])  # stable input
        rng.shuffle(members)
        n = len(members)
        # at least one sample per class in test and val when the class allows it
        n_test = max(1, round(test_frac * n)) if n >= 3 else (1 if n >= 1 else 0)
        n_val = max(1, round(val_frac * n)) if n >= 3 else 0
        if n_test + n_val >= n:  # never starve train for tiny classes
            n_test = min(n_test, max(0, n - 1))
            n_val = min(n_val, max(0, n - 1 - n_test))
        test.extend(members[:n_test])
        val.extend(members[n_test:n_test + n_val])
        train.extend(members[n_test + n_val:])
    return train, val, test


def _write_split(path, members):
    """Write a split TSV (prep_path basename, class_name, class_index)."""
    # sort output by (class_index, sample) so the file is deterministic regardless
    # of the order classes were appended in.
    members = sorted(members, key=lambda t: (t[2], t[0]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["prep_path", "class_name", "class_index"])
        for sample, name, idx in members:
            w.writerow([sample + SAMPLE_SUFFIX, name, idx])
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
    ap.add_argument("--labels", default=LABELS, help="label manifest TSV")
    ap.add_argument("--out-dir", default=SPLITS_DIR, help="where to write the splits")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    rows = _read_labels(args.labels)
    train, val, test = _stratified_split(rows, args.seed, args.test_frac, args.val_frac)

    os.makedirs(args.out_dir, exist_ok=True)
    outs = {
        "train": _write_split(os.path.join(args.out_dir, "pcawg_open_train.tsv"), train),
        "val": _write_split(os.path.join(args.out_dir, "pcawg_open_val.tsv"), val),
        "test": _write_split(os.path.join(args.out_dir, "pcawg_open_test.tsv"), test),
    }

    n = len(rows)
    print("labels: {} samples, {} classes, seed={}".format(
        n, len({r[2] for r in rows}), args.seed))
    print("split : train={} val={} test={}".format(len(train), len(val), len(test)))
    for which, path in outs.items():
        print("  {:5} -> {}  sha256={}".format(which, os.path.basename(path),
                                               _sha256(path)[:16]))


if __name__ == "__main__":
    main()

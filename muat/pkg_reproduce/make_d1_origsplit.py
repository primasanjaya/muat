"""Build the d1 reproduce splits from the ORIGINAL muat example_files partition.

`muat reproduce d1` trains on the split files named in experiments.json. To make
reproduce d1 regenerate the original-partition checkpoint -- i.e. match a
`muat train from-scratch` run on example_files/local_{train,val}_split_muat1.tsv
-- this converts those example splits into the reproduce *shipping* format:

  * prep_path -> BASENAME ``<uuid>.muat.tsv`` (remapped to the data bundle at run
    time by reproduce._resolve_split_rows; the bundle holds the decompressed
    ``.muat.tsv`` token files, which are byte-identical to the example splits'
    ``<uuid>.token.gc.genic.exonic.cs.tsv.gz`` after gunzip).
  * class_name / class_index preserved VERBATIM, so label_1 is identical.
  * row ORDER preserved, so seeded DataLoader shuffling reproduces bit-for-bit.

Output: splits/pcawg_orig_train.tsv, splits/pcawg_orig_val.tsv

Run: python muat/pkg_reproduce/make_d1_origsplit.py
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXAMPLE = os.path.join(REPO, "example_files")
OUT = os.path.join(HERE, "splits")

SRC = {
    "train": os.path.join(EXAMPLE, "local_train_split_muat1.tsv"),
    "val": os.path.join(EXAMPLE, "local_val_split_muat1.tsv"),
}
DST = {
    "train": os.path.join(OUT, "pcawg_orig_train.tsv"),
    "val": os.path.join(OUT, "pcawg_orig_val.tsv"),
}
# reproduce d2-d6 (predict/test) evaluate on the val split itself -- they point
# their `test` at pcawg_orig_val.tsv in experiments.json, so no separate test
# file is generated here.

_TOKEN_SUFFIX = ".token.gc.genic.exonic.cs.tsv.gz"


def to_bundle_basename(prep_path):
    """`/abs/<uuid>.token.gc.genic.exonic.cs.tsv.gz` -> `<uuid>.muat.tsv`."""
    base = os.path.basename(prep_path.strip())
    if base.endswith(_TOKEN_SUFFIX):
        uuid = base[: -len(_TOKEN_SUFFIX)]
    elif base.endswith(".muat.tsv"):
        uuid = base[: -len(".muat.tsv")]
    else:
        uuid = base  # leave as-is; the bundle-coverage check will flag it loudly
    return uuid + ".muat.tsv"


def convert(which):
    src, dst = SRC[which], DST[which]
    with open(src) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = [dict(r) for r in reader if (r.get("prep_path") or "").strip()]
    os.makedirs(OUT, exist_ok=True)
    cls_map = {}
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["prep_path", "class_name", "class_index"])
        for r in rows:
            cn = r["class_name"]
            # train/val use 'class_index'; tolerate the 'class_indexa' typo too.
            ci = r.get("class_index", r.get("class_indexa"))
            w.writerow([to_bundle_basename(r["prep_path"]), cn, ci])
            cls_map.setdefault(cn, ci)
    print("{:5s}: {} rows -> {}".format(which, len(rows), dst))
    return cls_map


def main():
    maps = {}
    for which in ("train", "val"):
        m = convert(which)
        for cn, ci in m.items():
            if cn in maps and maps[cn] != ci:
                print(
                    "WARNING: class_index mismatch for {!r}: {} vs {}".format(
                        cn, maps[cn], ci
                    ),
                    file=sys.stderr,
                )
            maps[cn] = ci
    print("distinct classes: {}".format(len(maps)))


if __name__ == "__main__":
    main()

"""Build the 90:10 reproduce splits from the ORIGINAL muat example_files partition.

Companion to make_d1_origsplit.py (which builds the 80:20 pcawg_orig_{train,val}).
This converts the 90:10 example_files partition into the reproduce *shipping* format:

  train = example_files/local_train_split9_muat1.tsv  (1630)
  test  = example_files/local_test_split1_muat1.tsv   (182, held-out)

The two are a verified-clean stratified ~90:10 partition of the 1812-sample / 17-class
benchmark (train n test = 0, union = the full 1812). Conversion rules match the 80:20
converter exactly:

  * prep_path -> BASENAME ``<uuid>.muat.tsv`` (remapped to the data bundle at run time
    by reproduce._resolve_split_rows; the bundle holds the decompressed ``.muat.tsv``
    token files, byte-identical to ``<uuid>.token.gc.genic.exonic.cs.tsv.gz`` gunzipped).
  * class_name / class_index preserved VERBATIM (headers were fixed: the old
    ``class_indexa`` typo is now ``class_index``).
  * row ORDER preserved, so seeded DataLoader shuffling reproduces bit-for-bit.

Output: splits/pcawg_orig9010_train.tsv, splits/pcawg_orig9010_test.tsv

Run: python muat/pkg_reproduce/make_d1_9010_origsplit.py
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXAMPLE = os.path.join(REPO, "example_files")
OUT = os.path.join(HERE, "splits")

SRC = {
    "train": os.path.join(EXAMPLE, "local_train_split9_muat1.tsv"),
    "test": os.path.join(EXAMPLE, "local_test_split1_muat1.tsv"),
}
DST = {
    "train": os.path.join(OUT, "pcawg_orig9010_train.tsv"),
    "test": os.path.join(OUT, "pcawg_orig9010_test.tsv"),
}

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
            # headers fixed to 'class_index'; tolerate the old 'class_indexa' just in case.
            ci = r.get("class_index", r.get("class_indexa"))
            w.writerow([to_bundle_basename(r["prep_path"]), cn, ci])
            cls_map.setdefault(cn, ci)
    print("{:5s}: {} rows -> {}".format(which, len(rows), dst))
    return cls_map


def main():
    maps = {}
    for which in ("train", "test"):
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

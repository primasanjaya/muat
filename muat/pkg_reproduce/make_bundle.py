#!/usr/bin/env python
"""Build the open-access PCAWG data bundle for `muat reproduce`.

The internal preprocessing pipeline emits one
``<sample>.token.gc.genic.exonic.cs.tsv.gz`` (gzipped) per aliquot. For the
distributable Zenodo bundle we repackage these under the friendly, plain-text
name ``<sample>.muat.tsv`` so users aren't confronted with the long internal
extension. Content is identical — same columns — just decompressed and renamed.

This stages exactly the benchmark samples listed in ``splits/pcawg_open_labels.tsv``
(1,812 samples / 17 tumour types) into an output directory ready to be tar-gzipped
and uploaded as the ``pcawg_open_preprocessed`` asset.

Usage
-----
    python muat/pkg_reproduce/make_bundle.py \
        --src data/preprocessed \
        --out /tmp/pcawg_open_muat

    # then package + checksum for Zenodo:
    tar -czf pcawg_open_muat.tar.gz -C /tmp pcawg_open_muat
    sha256sum pcawg_open_muat.tar.gz
"""
import argparse
import csv
import gzip
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "splits", "pcawg_open_labels.tsv")

SRC_SUFFIX = ".token.gc.genic.exonic.cs.tsv.gz"
OUT_SUFFIX = ".muat.tsv"


def _samples(labels_path):
    with open(labels_path) as fh:
        return [row["sample"] for row in csv.DictReader(fh, delimiter="\t")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/preprocessed",
                    help="directory holding the *.token.gc.genic.exonic.cs.tsv.gz files")
    ap.add_argument("--out", required=True,
                    help="output bundle directory (will hold <sample>.muat.tsv files)")
    ap.add_argument("--labels", default=LABELS, help="benchmark label manifest")
    args = ap.parse_args()

    samples = _samples(args.labels)
    os.makedirs(args.out, exist_ok=True)

    written, missing = 0, []
    for s in samples:
        src = os.path.join(args.src, s + SRC_SUFFIX)
        if not os.path.exists(src):
            missing.append(s)
            continue
        dst = os.path.join(args.out, s + OUT_SUFFIX)
        # decompress .gz -> plain .muat.tsv (identical content)
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        written += 1

    print("bundle: wrote {}/{} samples to {}".format(written, len(samples), args.out))
    if missing:
        print("WARNING: {} sample(s) missing from {} (e.g. {}). The bundle is "
              "INCOMPLETE — d2-d6 will report missing assets for these."
              .format(len(missing), args.src, missing[0]))


if __name__ == "__main__":
    main()

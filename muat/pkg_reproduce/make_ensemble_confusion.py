"""Rebuild example_files/ensemble_confusion_matrix{,_rownorm}.tsv.

NO MODEL IS RUN HERE. The MuAt ensemble was run elsewhere (long before this
script); its output is the checked-in file ``example_files/ensemble_prediction.tsv``.
This script only joins those predictions to ground-truth labels and cross-tabulates
them, which is why it needs no checkpoint, no GPU and no token data.

Inputs
------
``example_files/ensemble_prediction.tsv``
    Ensemble output: ``sample`` (aliquot UUID) + 24 per-class probability columns +
    ``prediction`` (the argmax of those 24 -- verified, not re-derived here). 1901
    rows. Carries NO ground-truth column, hence the join below. Note the 24-class
    taxonomy is the FULL PCAWG one, wider than the 17-class open benchmark.

``muat/pkg_reproduce/splits/pcawg_open_labels.tsv``
    Frozen ground truth for the open benchmark: ``sample`` / ``class_name`` /
    ``class_index``. 1812 rows, 17 classes.

Output
------
``ensemble_confusion_matrix.tsv``          raw counts, rows = true, cols = predicted
``ensemble_confusion_matrix_rownorm.tsv``  same, each row divided by its own total

The matrix is 17 x 21, not square: the rows are the 17 true benchmark classes, but
the ensemble can predict into any of its 24, and it used 4 classes that are outside
the benchmark taxonomy (ColoRect-AdenoCA, Kidney-ChRCC, Lung-AdenoCA,
Uterus-AdenoCA). Those off-taxonomy predictions are always wrong by construction and
are kept as columns so the errors stay visible instead of being silently dropped.

IMPORTANT -- the headline accuracy this yields (0.9603, 1740/1812) is measured over
ALL 1812 samples, i.e. the ensemble's own training material. It is almost certainly
IN-SAMPLE and is NOT comparable to the held-out single-model numbers (~0.79-0.88).
Do not cite it as held-out performance without first confirming the ensemble's
provenance (trained-on-everything vs cross-validated).

Run (pkg_reproduce has no __init__.py, so use a file path, not -m):

    python muat/pkg_reproduce/make_ensemble_confusion.py
"""

import argparse
import os

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PRED = os.path.join(_REPO, "example_files", "ensemble_prediction.tsv")
DEFAULT_LABELS = os.path.join(
    _REPO, "muat", "pkg_reproduce", "splits", "pcawg_open_labels.tsv"
)
DEFAULT_OUT_DIR = os.path.join(_REPO, "example_files")


def build(pred_path, labels_path):
    """Return (counts, rownorm, accuracy, n_matched) as DataFrames/scalars."""
    pred = pd.read_csv(pred_path, sep="\t")
    lab = pd.read_csv(labels_path, sep="\t")
    for col, path in (("sample", pred_path), ("prediction", pred_path)):
        if col not in pred.columns:
            raise ValueError("{}: missing required column {!r}".format(path, col))
    for col in ("sample", "class_name"):
        if col not in lab.columns:
            raise ValueError("{}: missing required column {!r}".format(labels_path, col))

    merged = lab.merge(pred[["sample", "prediction"]], on="sample", how="inner")
    if len(merged) != len(lab):
        # Every benchmark sample should appear in the ensemble output; a shortfall
        # means the two files have drifted apart and the matrix would be partial.
        raise ValueError(
            "only {} of {} labelled samples found in {} -- refusing to build a "
            "partial confusion matrix.".format(len(merged), len(lab), pred_path)
        )

    counts = pd.crosstab(merged["class_name"], merged["prediction"])
    counts.index.name = "true\\pred"
    # Row-normalise by each true class's own support (rows sum to 1.0).
    rownorm = counts.div(counts.sum(axis=1), axis=0)
    accuracy = float((merged["class_name"] == merged["prediction"]).mean())
    return counts, rownorm, accuracy, len(merged)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--predictions", default=DEFAULT_PRED)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)

    counts, rownorm, accuracy, n = build(args.predictions, args.labels)
    os.makedirs(args.out_dir, exist_ok=True)

    counts_path = os.path.join(args.out_dir, "ensemble_confusion_matrix.tsv")
    rownorm_path = os.path.join(args.out_dir, "ensemble_confusion_matrix_rownorm.tsv")
    counts.to_csv(counts_path, sep="\t")
    rownorm.to_csv(rownorm_path, sep="\t", float_format="%.4f")

    correct = int(round(accuracy * n))
    print("matched {} samples | {} true classes x {} predicted classes".format(
        n, counts.shape[0], counts.shape[1]))
    print("accuracy = {:.4f} ({}/{})  <-- IN-SAMPLE, see module docstring".format(
        accuracy, correct, n))
    off = [c for c in counts.columns if c not in set(counts.index)]
    if off:
        print("off-taxonomy predicted classes (always wrong): {}".format(", ".join(off)))
    print("wrote {}".format(counts_path))
    print("wrote {}".format(rownorm_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

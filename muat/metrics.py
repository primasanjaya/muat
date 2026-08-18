"""Classification metrics for MuAt validation/test logit tables.

Computes top-1 / top-3 / top-5 accuracy, per-class precision / recall / F1 /
support, and macro & weighted averages from a ``*_first_logits.tsv`` table --
the format written by ``trainer.py`` during validation: one column per class
holding the logit, followed by a ``target_name`` column (the true label) and a
``sample`` column. The predicted class is the argmax over the class-logit
columns; top-k accuracy asks whether the true class is among the k
highest-scoring classes.

This addresses the revision request for recall / F1 / per-class metrics beyond
the single top-1 accuracy currently logged in ``evaluation.tsv``. It relies only
on ``sklearn.metrics`` and ``pandas``, both already MuAt dependencies, so it adds
no new requirement and stays import-light.

Run standalone:

    python -m muat.metrics path/to/best_val_first_logits.tsv --out-dir path/to/out

or call :func:`metrics_from_logits` from other code.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Columns in a *_first_logits.tsv that are NOT class-logit columns.
# trainer.py writes 'target_name' (true label); predict.py writes 'prediction'.
_META_COLS = ("target_name", "sample", "prediction")

# Which top-k accuracies to report. k values larger than the number of classes
# are dropped (top-17 of 17 classes is trivially 1.0 and not worth printing).
DEFAULT_TOPK = (1, 3, 5)


def load_logits_table(path, require_labels=True):
    """Read a ``*_first_logits.tsv`` and return ``(y_true, y_pred, class_names, logits)``.

    ``class_names`` are the logit-column headers in file order (= class-index
    order). ``y_pred`` is the argmax over those columns; ``y_true`` is the
    ``target_name`` column; ``logits`` is the raw (n_samples, n_classes) score
    matrix, kept so top-k accuracy can be computed. Raises ``ValueError`` if
    there is no ``target_name`` column (e.g. a predict output, which carries no
    ground truth) or no class columns.

    ``require_labels=False`` allows a predict-mode table (no ``target_name``,
    e.g. inference on unlabeled data) to load anyway, with ``y_true`` filled
    with ``None`` -- for callers that only need predictions/logits, not accuracy.
    """
    df = pd.read_csv(path, sep="\t")
    has_labels = "target_name" in df.columns
    if not has_labels and require_labels:
        raise ValueError(
            "{}: no 'target_name' column found -- cannot compute metrics without "
            "ground-truth labels (is this a predict output rather than a "
            "validation logits table?).".format(path)
        )
    class_names = [c for c in df.columns if c not in _META_COLS]
    if not class_names:
        raise ValueError("{}: no class-logit columns found.".format(path))
    logits = df[class_names].to_numpy()
    y_pred = [class_names[i] for i in logits.argmax(axis=1)]
    y_true = df["target_name"].astype(str).tolist() if has_labels else [None] * len(df)
    return y_true, y_pred, class_names, logits


def compute_topk_accuracy(y_true, logits, class_names, topk=DEFAULT_TOPK):
    """Return ``{k: accuracy}`` -- the fraction of rows whose true class is among
    the ``k`` highest-scoring classes.

    Ranking is done on the raw logits. This is equivalent to ranking on softmax
    probabilities (softmax is monotonic, so it does not change the order), which
    is why no probability conversion is needed. Rows whose ``target_name`` is not
    one of ``class_names`` count as incorrect at every k. ``k`` values exceeding
    the class count are skipped.
    """
    n_classes = len(class_names)
    index_of = {c: i for i, c in enumerate(class_names)}
    # Column index of the true class per row, or -1 when the label is unknown.
    true_idx = np.array([index_of.get(str(t), -1) for t in y_true])
    # Descending rank order of the class columns for each row.
    order = np.argsort(-logits, axis=1, kind="stable")
    out = {}
    for k in topk:
        if k > n_classes:
            continue
        topk_idx = order[:, :k]
        hit = (topk_idx == true_idx[:, None]).any(axis=1)
        out[int(k)] = float(hit.mean()) if len(hit) else 0.0
    return out


def compute_metrics(y_true, y_pred, class_names, logits=None, topk=DEFAULT_TOPK):
    """Return a dict of per-class and aggregate classification metrics.

    Per-class precision/recall/F1/support is reported for every label in
    ``class_names`` (so classes absent from ``y_true`` still appear, with
    support 0), plus overall accuracy and macro / weighted averages.
    ``zero_division=0`` keeps undefined precision/recall at 0 rather than
    raising. When ``logits`` is given, a ``topk`` entry maps each k to its
    top-k accuracy (``topk[1]`` equals ``accuracy``).
    """
    labels = list(class_names)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        {
            "class_name": c,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, c in enumerate(labels)
    ]
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_samples": len(y_true),
        "n_classes": len(labels),
        "macro": {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
        "weighted": {"precision": float(wp), "recall": float(wr), "f1": float(wf)},
        "per_class": per_class,
        "labels": labels,
    }
    if logits is not None:
        out["topk"] = compute_topk_accuracy(y_true, logits, labels, topk=topk)
    return out


def write_metrics(metrics, out_dir, prefix="", y_true=None, y_pred=None):
    """Write metric tables into ``out_dir`` and return the list of paths written.

    Always writes ``{prefix}per_class_metrics.tsv`` (one row per class plus
    macro_avg / weighted_avg rows) and ``{prefix}metrics_summary.tsv``. If both
    ``y_true`` and ``y_pred`` are given, also writes
    ``{prefix}confusion_matrix.tsv`` (rows = true class, cols = predicted).
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []

    per_class_path = os.path.join(out_dir, "{}per_class_metrics.tsv".format(prefix))
    with open(per_class_path, "w") as fh:
        fh.write("class_name\tprecision\trecall\tf1\tsupport\n")
        for row in metrics["per_class"]:
            fh.write(
                "{class_name}\t{precision:.6f}\t{recall:.6f}\t{f1:.6f}\t{support}\n".format(
                    **row
                )
            )
        for avg in ("macro", "weighted"):
            m = metrics[avg]
            fh.write(
                "{}_avg\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n".format(
                    avg, m["precision"], m["recall"], m["f1"], metrics["n_samples"]
                )
            )
    written.append(per_class_path)

    summary_path = os.path.join(out_dir, "{}metrics_summary.tsv".format(prefix))
    with open(summary_path, "w") as fh:
        fh.write("metric\tvalue\n")
        fh.write("accuracy\t{:.6f}\n".format(metrics["accuracy"]))
        for k in sorted(metrics.get("topk", {})):
            fh.write("top{}_accuracy\t{:.6f}\n".format(k, metrics["topk"][k]))
        fh.write("macro_f1\t{:.6f}\n".format(metrics["macro"]["f1"]))
        fh.write("weighted_f1\t{:.6f}\n".format(metrics["weighted"]["f1"]))
        fh.write("macro_recall\t{:.6f}\n".format(metrics["macro"]["recall"]))
        fh.write("macro_precision\t{:.6f}\n".format(metrics["macro"]["precision"]))
        fh.write("weighted_recall\t{:.6f}\n".format(metrics["weighted"]["recall"]))
        fh.write("weighted_precision\t{:.6f}\n".format(metrics["weighted"]["precision"]))
        fh.write("n_samples\t{}\n".format(metrics["n_samples"]))
        fh.write("n_classes\t{}\n".format(metrics["n_classes"]))
    written.append(summary_path)

    if y_true is not None and y_pred is not None:
        labels = metrics["labels"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_path = os.path.join(out_dir, "{}confusion_matrix.tsv".format(prefix))
        with open(cm_path, "w") as fh:
            fh.write("true\\pred\t" + "\t".join(labels) + "\n")
            for i, c in enumerate(labels):
                fh.write(c + "\t" + "\t".join(str(int(x)) for x in cm[i]) + "\n")
        written.append(cm_path)

    return written


def metrics_from_logits(logits_path, out_dir=None, prefix=""):
    """Load a logits table, compute metrics, and optionally write them out.

    Returns the metrics dict from :func:`compute_metrics`. When ``out_dir`` is
    given, also writes the per-class / summary / confusion-matrix tables there.
    """
    y_true, y_pred, class_names, logits = load_logits_table(logits_path)
    metrics = compute_metrics(y_true, y_pred, class_names, logits=logits)
    if out_dir:
        write_metrics(metrics, out_dir, prefix=prefix, y_true=y_true, y_pred=y_pred)
    return metrics


def _format_report(metrics):
    """Return a human-readable per-class table as a string (for stdout)."""
    lines = []
    lines.append(
        "{:<22}{:>10}{:>10}{:>10}{:>10}".format(
            "class", "precision", "recall", "f1", "support"
        )
    )
    for row in metrics["per_class"]:
        lines.append(
            "{class_name:<22}{precision:>10.4f}{recall:>10.4f}{f1:>10.4f}{support:>10}".format(
                **row
            )
        )
    lines.append("")
    lines.append(
        "{:<22}{:>10}{:>10.4f}{:>10.4f}{:>10}".format(
            "macro avg", "", metrics["macro"]["recall"], metrics["macro"]["f1"],
            metrics["n_samples"],
        )
    )
    lines.append(
        "{:<22}{:>10}{:>10.4f}{:>10.4f}{:>10}".format(
            "weighted avg", "", metrics["weighted"]["recall"],
            metrics["weighted"]["f1"], metrics["n_samples"],
        )
    )
    lines.append("")
    if metrics.get("topk"):
        lines.append(
            "  |  ".join(
                "top-{} acc = {:.4f}".format(k, metrics["topk"][k])
                for k in sorted(metrics["topk"])
            )
        )
    lines.append(
        "accuracy = {:.4f}  |  macro-F1 = {:.4f}  |  weighted-F1 = {:.4f}  "
        "({} samples, {} classes)".format(
            metrics["accuracy"], metrics["macro"]["f1"], metrics["weighted"]["f1"],
            metrics["n_samples"], metrics["n_classes"],
        )
    )
    return "\n".join(lines)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute per-class precision/recall/F1, accuracy, and a "
        "confusion matrix from a MuAt *_first_logits.tsv validation table."
    )
    parser.add_argument("logits", help="path to a *_first_logits.tsv table")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory to write per_class_metrics.tsv / metrics_summary.tsv / "
        "confusion_matrix.tsv (default: alongside the logits file)",
    )
    parser.add_argument(
        "--prefix", default="", help="filename prefix for the written tables"
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.logits))
    metrics = metrics_from_logits(args.logits, out_dir=out_dir, prefix=args.prefix)
    print(_format_report(metrics))
    print("\nwrote tables to {}".format(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

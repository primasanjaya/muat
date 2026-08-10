"""Aggregate the 10 same-seed d1 training repeats into one report.

For each repeat's result directory this computes top-1 / top-3 / top-5 accuracy
plus precision / recall / F1 (macro and weighted), then reports the per-repeat
table together with mean +/- sd across repeats.

The d1 setup is an 80:20 split: train 1449, test 363. The REPORTED result is the
BEST EPOCH (project decision), and the published d1 checkpoint is the matching
``best_ckpt.pthx`` -- so the reported number and every downstream inference run
(d2-d6) refer to the same model.

``best_val_first_logits.tsv``   BEST epoch -- THE REPORTED RESULT. Pairs with
                                ``best_ckpt.pthx``.
``val_first_logits.tsv``        FINAL epoch. Printed underneath as a secondary
                                reference only; pairs with ``model.pthx``.

Recorded for the write-up: "best" is the epoch with the highest accuracy on these
same 363 samples (trainer.py:324), so the test set informed epoch selection. The
figure is therefore a best-epoch result rather than a selection-free one. This is
noted once here so the distinction is not lost later; it is not a defect and needs
no rerun.

Because every repeat runs the identical command under the identical seed, the
expected result is ZERO variance. The script therefore also runs a determinism
check per evaluation: it compares the class-logit matrix and the predicted
labels of every repeat against repeat 1 and reports whether they are
bit-identical. The ``sample`` / prep_path column is deliberately excluded from
that comparison -- it records which directory the data was read from, so it
differs between runs without meaning the model differed.

pkg_reproduce has no ``__init__.py``, so run it as a file path, not with ``-m``:

    python muat/pkg_reproduce/aggregate_repeats.py \
        --glob 'data/reproduce_results/d1_rep*_278400' --expect 10
"""

import argparse
import glob as globmod
import os
import sys

import numpy as np

# Import muat.metrics from the repo this file lives in, without needing muat installed.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from muat.metrics import (  # noqa: E402
    DEFAULT_TOPK,
    compute_metrics,
    load_logits_table,
)

# (filename, short key, human description). Order matters: the first entry is the
# headline evaluation, printed first and used for the script's exit status.
EVALUATIONS = [
    (
        "best_val_first_logits.tsv",
        "best_epoch",
        "REPORTED RESULT -- best epoch on the 363-sample test set (pairs with best_ckpt.pthx)",
    ),
    (
        "val_first_logits.tsv",
        "final_epoch",
        "secondary reference -- final epoch (pairs with model.pthx); not the reported figure",
    ),
]


def collect_runs(patterns, logits_name):
    """Return sorted (label, logits_path) pairs for every result dir that has logits."""
    dirs = []
    for pat in patterns:
        dirs.extend(globmod.glob(pat))
    runs, missing = [], []
    for d in sorted(set(dirs)):
        path = os.path.join(d, logits_name)
        if os.path.isfile(path):
            runs.append((os.path.basename(d.rstrip("/")), path))
        else:
            missing.append(d)
    return runs, missing


def summarise(path):
    """Compute the reported metric set for one repeat."""
    y_true, y_pred, class_names, logits = load_logits_table(path)
    m = compute_metrics(y_true, y_pred, class_names, logits=logits)
    row = {
        "n_samples": m["n_samples"],
        "n_classes": m["n_classes"],
        "macro_precision": m["macro"]["precision"],
        "macro_recall": m["macro"]["recall"],
        "macro_f1": m["macro"]["f1"],
        "weighted_precision": m["weighted"]["precision"],
        "weighted_recall": m["weighted"]["recall"],
        "weighted_f1": m["weighted"]["f1"],
    }
    for k in sorted(m.get("topk", {})):
        row["top{}".format(k)] = m["topk"][k]
    return row, logits, y_pred, y_true


def determinism_check(runs, logit_mats, preds):
    """Compare every repeat against the first; return (list of per-run status, all_ok)."""
    base_logits, base_pred = logit_mats[0], preds[0]
    statuses, all_ok = [], True
    for i, (label, _) in enumerate(runs):
        if i == 0:
            statuses.append((label, "reference", "reference"))
            continue
        same_shape = logit_mats[i].shape == base_logits.shape
        logit_ok = same_shape and np.array_equal(logit_mats[i], base_logits)
        pred_ok = preds[i] == base_pred
        if not (logit_ok and pred_ok):
            all_ok = False
        statuses.append(
            (
                label,
                "IDENTICAL" if logit_ok else ("shape differs" if not same_shape else "DIFFERS"),
                "identical" if pred_ok else "DIFFERS",
            )
        )
    return statuses, all_ok


def report_one(patterns, logits_name, description, expect, tsv_rows):
    """Print the full report for one evaluation.

    Appends its rows to ``tsv_rows`` and returns
    ``(found_any, determinism_ok, metric_cols)``; ``metric_cols`` is None when
    nothing was found.
    """
    runs, missing = collect_runs(patterns, logits_name)
    print()
    print("#" * 78)
    print("# {}".format(description))
    print("# source: {}".format(logits_name))
    print("#" * 78)
    for d in missing:
        print("skipping {} (no {} - run unfinished or failed?)".format(d, logits_name))
    if not runs:
        print("no result directories contained {} - skipping this evaluation.".format(
            logits_name))
        return False, True, None
    if expect is not None and len(runs) != expect:
        print(
            "WARNING: found {} repeat(s), expected {}. Numbers below cover only the "
            "completed runs.".format(len(runs), expect)
        )

    rows, logit_mats, preds = [], [], []
    for label, path in runs:
        row, logits, y_pred, _ = summarise(path)
        row["run"] = label
        rows.append(row)
        logit_mats.append(logits)
        preds.append(y_pred)

    metric_cols = [c for c in ("top1", "top3", "top5") if c in rows[0]]
    metric_cols += [
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]

    print()
    print("--- per-repeat metrics ({} run(s), {} samples, {} classes) ---".format(
        len(rows), rows[0]["n_samples"], rows[0]["n_classes"]))
    print("{:<28}".format("run") + "".join("{:>20}".format(c) for c in metric_cols))
    for row in rows:
        print(
            "{:<28}".format(row["run"][:27])
            + "".join("{:>20.6f}".format(row[c]) for c in metric_cols)
        )

    print()
    print("--- mean +/- sd across {} repeat(s) ---".format(len(rows)))
    stats = {}
    for c in metric_cols:
        vals = np.array([row[c] for row in rows], dtype=float)
        # Population sd: these are the complete set of repeats, not a sample of them.
        stats[c] = (float(vals.mean()), float(vals.std()), float(vals.min()), float(vals.max()))
        print("{:<20} {:.6f} +/- {:.6f}   [min {:.6f}, max {:.6f}]".format(c, *stats[c]))

    print()
    print("--- determinism check (vs {}) ---".format(runs[0][0]))
    statuses, all_ok = determinism_check(runs, logit_mats, preds)
    for label, logit_status, pred_status in statuses:
        print("{:<28} logits: {:<14} predictions: {}".format(
            label[:27], logit_status, pred_status))
    print()
    if len(runs) < 2:
        print("Only one repeat - determinism not exercised.")
        all_ok = True
    elif all_ok:
        print(
            "PASS: all {} repeats produced bit-identical logits and predictions. "
            "Same seed -> same result; every metric above has zero variance.".format(len(runs))
        )
    else:
        print(
            "FAIL: repeats diverged despite the identical seed. This is a real "
            "determinism bug -- inspect seeding, dataloader shuffling/worker seeds, "
            "and any non-deterministic CUDA kernels."
        )

    for row in rows:
        tsv_rows.append([logits_name, row["run"]] + ["{:.6f}".format(row[c]) for c in metric_cols])
    for agg, idx in (("mean", 0), ("sd", 1)):
        tsv_rows.append([logits_name, agg] + ["{:.6f}".format(stats[c][idx]) for c in metric_cols])
    return True, all_ok, metric_cols


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate same-seed d1 training repeats: top-1/3/5 accuracy, "
        "precision, recall, weighted-F1, mean +/- sd, and a determinism check. "
        "Reports the held-out test (final epoch) and, for continuity, the "
        "selection-biased best epoch."
    )
    parser.add_argument(
        "--glob",
        dest="patterns",
        action="append",
        required=True,
        help="glob for repeat result directories (repeatable). Quote it so the "
        "shell does not expand it, e.g. --glob 'data/reproduce_results/d1_rep*_278400'",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the per-repeat tables as TSV here (default: no file, stdout only)",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="expected number of repeats; warn if fewer were found (e.g. --expect 10)",
    )
    args = parser.parse_args(argv)

    tsv_rows, metric_cols, found_any, all_ok = [], None, False, True
    for logits_name, _key, description in EVALUATIONS:
        found, ok, cols = report_one(
            args.patterns, logits_name, description, args.expect, tsv_rows
        )
        found_any = found_any or found
        all_ok = all_ok and ok
        if cols is not None:
            metric_cols = cols

    if not found_any:
        parser.error(
            "no result directories matched, or none contained any of: {}".format(
                ", ".join(n for n, _, _ in EVALUATIONS)
            )
        )

    print()
    print("=" * 78)
    print("REPORT THE FIRST BLOCK (best epoch). Publish best_ckpt.pthx as d1.pthx so")
    print("d2-d6 run inference with the same model these numbers describe.")
    print("=" * 78)

    if args.out and metric_cols:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("evaluation\trun\t" + "\t".join(metric_cols) + "\n")
            for r in tsv_rows:
                fh.write("\t".join(r) + "\n")
        print("\nwrote {}".format(args.out))

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

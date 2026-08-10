"""Compare two runs of the same experiment executed in DIFFERENT environments.

Within one environment, repeats of a seeded muat run are expected to be identical --
that is what aggregate_repeats.py checks. Across environments they are NOT: CPU and GPU
float32 kernels differ in reduction order, so the logits differ in the last few decimal
places even when the computation is nominally the same. Demanding bit-equality there
would be claiming something no framework delivers.

This script therefore scores a pair of runs against an explicit tolerance, fixed before
the runs were executed (Sheet4 of example_files/local_checkpoint_reports_v2.xlsx):

  1. at least (n - 1) of n top-1 predictions identical, i.e. delta Top-1 acc <= 1/n
  2. max |delta logit| below --max-logit-delta (default 1e-3)
  3. every disagreeing sample is a NEAR-TIE: its top-2 logit margin in the reference run
     is smaller than the observed max |delta logit|

Criterion 3 is the one that carries the scientific claim. A flipped prediction on a
sample the reference model was confident about is a portability defect; a flip on a
sample whose top two classes were separated by less than the numerical noise is not.

Samples are matched on the aliquot UUID parsed out of the ``sample`` column, because the
two runs read their inputs from different paths (bundle vs preprocessed directory) and
the raw path strings will not match.

pkg_reproduce has no ``__init__.py``, so run it as a file path, not with ``-m``:

    python muat/pkg_reproduce/compare_environments.py \
        --ref  data/reproduce_results/d1_rep01_<id>/best_val_first_logits.tsv \
        --test data/reproduce_results/d2_rep01_<id>/best_val_first_logits.tsv \
        --ref-label 'd1 (Puhti GPU, bioconda)' --test-label 'd2 (Puhti CPU, docker)'

Exit status is 0 if the pair meets the tolerance, 1 if it does not.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# Import muat.metrics from the repo this file lives in, without needing muat installed.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from muat.metrics import load_logits_table  # noqa: E402

# PCAWG aliquot ids are UUIDs; the surrounding path and suffix vary between runs.
_UUID = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE
)


def sample_keys(path):
    """Aliquot id per row, in file order. Falls back to the basename stem."""
    df = pd.read_csv(path, sep='\t', usecols=lambda c: c == 'sample')
    if 'sample' not in df.columns:
        raise ValueError('%s: no "sample" column -- cannot align the two runs.' % path)
    keys = []
    for value in df['sample'].astype(str):
        m = _UUID.search(value)
        keys.append(m.group(0).lower() if m else os.path.basename(value).split('.')[0])
    return keys


def load(path):
    y_true, y_pred, class_names, logits = load_logits_table(path)
    keys = sample_keys(path)
    if len(keys) != len(y_true):
        raise ValueError('%s: %d sample ids for %d rows.' % (path, len(keys), len(y_true)))
    if len(set(keys)) != len(keys):
        raise ValueError('%s: duplicate sample ids -- cannot align unambiguously.' % path)
    return {
        'keys': keys,
        'y_true': dict(zip(keys, y_true)),
        'y_pred': dict(zip(keys, y_pred)),
        'logits': {k: logits[i] for i, k in enumerate(keys)},
        'class_names': class_names,
    }


def compare(ref, test, max_logit_delta):
    shared = [k for k in ref['keys'] if k in test['y_pred']]
    if not shared:
        raise ValueError('the two runs share no sample ids -- are they the same experiment?')

    if ref['class_names'] != test['class_names']:
        raise ValueError(
            'class columns differ between the two runs (%s vs %s) -- the models were '
            'trained on different label sets and are not comparable.'
            % (ref['class_names'], test['class_names'])
        )

    ref_mat = np.array([ref['logits'][k] for k in shared], dtype=float)
    test_mat = np.array([test['logits'][k] for k in shared], dtype=float)
    delta = np.abs(ref_mat - test_mat)

    # Top-2 margin in the REFERENCE run: how much numerical slack a sample had before
    # its prediction could legitimately flip.
    ordered = np.sort(ref_mat, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]

    max_delta = float(delta.max())
    # Judge each flip against the perturbation seen on ITS OWN row, not the global max --
    # otherwise one outlier sample would excuse every disagreement in the table.
    row_delta = delta.max(axis=1)

    disagreements = []
    for i, k in enumerate(shared):
        if ref['y_pred'][k] != test['y_pred'][k]:
            disagreements.append({
                'sample': k,
                'true': ref['y_true'][k],
                'ref_pred': ref['y_pred'][k],
                'test_pred': test['y_pred'][k],
                'ref_margin': float(margins[i]),
                'row_delta': float(row_delta[i]),
                'near_tie': float(margins[i]) < float(row_delta[i]),
            })

    n = len(shared)
    ref_acc = sum(ref['y_pred'][k] == ref['y_true'][k] for k in shared) / n
    test_acc = sum(test['y_pred'][k] == test['y_true'][k] for k in shared) / n

    return {
        'n': n,
        'n_ref_only': len(ref['keys']) - n,
        'n_test_only': len(test['keys']) - n,
        'agree': n - len(disagreements),
        'agreement_pct': 100.0 * (n - len(disagreements)) / n,
        'ref_acc': ref_acc,
        'test_acc': test_acc,
        'delta_top1': test_acc - ref_acc,
        'max_delta': max_delta,
        'mean_delta': float(delta.mean()),
        'bit_identical': max_delta == 0.0,
        'disagreements': disagreements,
        'all_near_ties': all(d['near_tie'] for d in disagreements),
        'pass_agreement': len(disagreements) <= 1,
        'pass_logit': max_delta < max_logit_delta,
    }


def report(res, ref_label, test_label, max_logit_delta):
    w = 78
    print('=' * w)
    print('Cross-environment comparison')
    print('  reference : %s' % ref_label)
    print('  test      : %s' % test_label)
    print('=' * w)
    if res['n_ref_only'] or res['n_test_only']:
        print('NOTE: %d sample(s) only in reference, %d only in test -- compared the %d shared.'
              % (res['n_ref_only'], res['n_test_only'], res['n']))
    print('samples compared        %d' % res['n'])
    print('predictions agreeing    %d / %d  (%.4f%%)'
          % (res['agree'], res['n'], res['agreement_pct']))
    print('Top-1 accuracy          reference %.6f   test %.6f   delta %+.6f'
          % (res['ref_acc'], res['test_acc'], res['delta_top1']))
    print('max |delta logit|       %.3e' % res['max_delta'])
    print('mean |delta logit|      %.3e' % res['mean_delta'])
    if res['bit_identical']:
        print('\nThe two runs are BIT-IDENTICAL. If they really did run in different '
              'environments,\nverify that -- it is a stronger result than expected and is worth '
              'confirming\nrather than reporting on trust.')

    if res['disagreements']:
        print('\ndisagreeing samples (%d):' % len(res['disagreements']))
        print('  %-38s %-18s %-18s %10s %10s  %s'
              % ('sample', 'reference pred', 'test pred', 'ref margin', 'row delta', 'near-tie'))
        for d in res['disagreements']:
            print('  %-38s %-18s %-18s %10.3e %10.3e  %s'
                  % (d['sample'], d['ref_pred'][:18], d['test_pred'][:18],
                     d['ref_margin'], d['row_delta'], 'yes' if d['near_tie'] else 'NO'))

    print('\n%s' % ('-' * w))
    checks = [
        ('at most 1 prediction differs', res['pass_agreement']),
        ('max |delta logit| < %.0e' % max_logit_delta, res['pass_logit']),
        ('all disagreements are near-ties', res['all_near_ties']),
    ]
    for label, ok in checks:
        print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    overall = all(ok for _, ok in checks)
    print('\n%s -- the two environments %s the agreed tolerance.'
          % ('PASS' if overall else 'FAIL',
             'agree within' if overall else 'DO NOT agree within'))
    print('=' * w)
    return overall


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True, help='reference run logits TSV')
    ap.add_argument('--test', required=True, help='test run logits TSV')
    ap.add_argument('--ref-label', default=None)
    ap.add_argument('--test-label', default=None)
    ap.add_argument('--max-logit-delta', type=float, default=1e-3,
                    help='tolerance on max |delta logit| (default: %(default)s)')
    ap.add_argument('--out', default=None, help='also write a one-row TSV summary here')
    args = ap.parse_args(argv)

    res = compare(load(args.ref), load(args.test), args.max_logit_delta)
    ok = report(res, args.ref_label or args.ref, args.test_label or args.test,
                args.max_logit_delta)

    if args.out:
        cols = ['ref', 'test', 'n', 'agree', 'agreement_pct', 'ref_acc', 'test_acc',
                'delta_top1', 'max_delta', 'mean_delta', 'all_near_ties', 'tolerance_met']
        row = [args.ref_label or args.ref, args.test_label or args.test, res['n'], res['agree'],
               '%.6f' % res['agreement_pct'], '%.6f' % res['ref_acc'], '%.6f' % res['test_acc'],
               '%+.6f' % res['delta_top1'], '%.6e' % res['max_delta'],
               '%.6e' % res['mean_delta'], res['all_near_ties'], ok]
        with open(args.out, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            fh.write('\t'.join(str(x) for x in row) + '\n')
        print('wrote %s' % args.out)

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

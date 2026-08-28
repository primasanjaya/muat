#!/usr/bin/env python
"""Package the checkpoints from an N-repeat reproducibility run for archival.

Produces two artifacts from a set of repeat result directories:

  <label>.zip   all N checkpoints (e.g. ckpt1.zip -> ckpt1_rep01.pthx ... ckpt1_rep10.pthx)
                plus MANIFEST.tsv and README.md. This is the determinism EVIDENCE bundle.
  <tag>.pthx    a single canonical checkpoint (repeat 1), which downstream inference
                recipes load as the `checkpoint` asset.

Both are published to Zenodo; the sha256 printed at the end goes into experiments.json.

A muat checkpoint (.pthx) is itself a zip of JSON configs + weight.pth. Two runs that
train identically still produce .pthx files with DIFFERENT outer md5, because the zip
container embeds a timestamp per member. Determinism must therefore be judged on the
INNER members, which is what this script hashes.

The output zip uses fixed member timestamps and a fixed compression level so that
re-running this script on the same inputs yields a byte-identical archive -- otherwise
the sha256 recorded in experiments.json would not be reproducible either.

Run by file path (pkg_reproduce has no __init__.py):

    python muat/pkg_reproduce/package_checkpoints.py \
        --glob 'data/reproduce_results/d1_rep*_280085' \
        --tag d1 --label ckpt1 --out-dir data/reproduce_release
"""

import argparse
import glob as globmod
import hashlib
import os
import sys
import zipfile

# Fixed epoch for every member written into the output archive, so the archive is
# reproducible. 1980-01-01 00:00:00 is the earliest timestamp the zip format allows.
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

CANONICAL_CKPT = 'best_ckpt.pthx'
# Inner members that must match across repeats for the run to count as deterministic.
# model_name/metadata are excluded: they carry run-local provenance, not model state.
COMPARED_MEMBERS = (
    'weight.pth',
    'model_config.json',
    'trainer_config.json',
    'dataloader_config.json',
    'motif_dict.json',
    'pos_dict.json',
    'ges_dict.json',
    'target_handler_1.json',
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inner_hashes(pthx_path):
    """sha256 of each compared member inside a .pthx, keyed by member name."""
    out = {}
    with zipfile.ZipFile(pthx_path) as z:
        names = set(z.namelist())
        for member in COMPARED_MEMBERS:
            if member in names:
                out[member] = sha256_bytes(z.read(member))
    return out


def discover(pattern, ckpt_name):
    dirs = sorted(d for d in globmod.glob(pattern) if os.path.isdir(d))
    if not dirs:
        raise SystemExit('no directories matched: %s' % pattern)
    found, missing = [], []
    for d in dirs:
        p = os.path.join(d, ckpt_name)
        (found if os.path.isfile(p) else missing).append(p if os.path.isfile(p) else d)
    if missing:
        raise SystemExit(
            'these result dirs have no %s (run incomplete?):\n  %s'
            % (ckpt_name, '\n  '.join(missing))
        )
    return found


def compare(paths):
    """Return (all_identical, per_member_identical, reference_hashes)."""
    ref = inner_hashes(paths[0])
    per_member = {m: True for m in ref}
    for p in paths[1:]:
        h = inner_hashes(p)
        for m in ref:
            if h.get(m) != ref[m]:
                per_member[m] = False
    return all(per_member.values()), per_member, ref


def build_readme(tag, label, paths, identical, ref, source_note, member_pattern):
    lines = [
        '# %s -- checkpoints from the %d-repeat %s reproducibility run' % (label, len(paths), tag),
        '',
        'Each `%s` is the best-epoch checkpoint of one independent repeat of' % member_pattern,
        'experiment `%s`. All repeats used the same seed, the same data split and the same' % tag,
        'hyperparameters, and were executed in the same environment; they differ only in',
        'being separate executions.',
        '',
        '## Determinism',
        '',
    ]
    if identical:
        lines += [
            'All %d repeats are IDENTICAL in every model-bearing member of the checkpoint,' % len(paths),
            'including the weight tensors:',
            '',
            '    weight.pth sha256 = %s' % ref.get('weight.pth', 'n/a'),
            '',
            'Note that the sha256 of the `.pthx` FILES differs between repeats even though the',
            'models are identical: `.pthx` is a zip container that stores a modification',
            'timestamp per member. Verify determinism on the inner members, not the container.',
        ]
    else:
        lines += [
            'WARNING: the repeats are NOT identical. See MANIFEST.tsv for which members differ.',
        ]
    lines += [
        '',
        '## Loading',
        '',
        'These are muat checkpoints: a zip of JSON configs plus `weight.pth`. Load them with',
        'muat (`muat predict from-checkpoint --ckpt-filepath <file>`), not with `torch.load`.',
        '',
        '## Provenance',
        '',
        source_note,
        '',
        'MANIFEST.tsv lists, per repeat, the source result directory, the sha256 of the',
        '.pthx container, and the sha256 of the inner weight.pth.',
        '',
    ]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--glob', required=True,
                    help="glob for the repeat result dirs, e.g. 'data/reproduce_results/d1_rep*_280085'")
    ap.add_argument('--tag', required=True, help='experiment tag, e.g. d1')
    ap.add_argument('--label', required=True, help='checkpoint label / archive name, e.g. ckpt1')
    ap.add_argument('--out-dir', required=True, help='directory to write the artifacts into')
    ap.add_argument('--ckpt-name', default=CANONICAL_CKPT,
                    help='which checkpoint file to take from each result dir (default: %(default)s)')
    ap.add_argument('--expect', type=int, default=None,
                    help='fail unless exactly this many repeats are found')
    ap.add_argument('--source-note', default='',
                    help='free-text provenance line recorded in the README (environment, node, install)')
    ap.add_argument('--allow-divergent', action='store_true',
                    help='package even if the repeats are not identical (default: refuse)')
    ap.add_argument('--member-prefix', default=None,
                    help="prefix for each archive member's filename (default: --label, "
                         "the original '<label>_repNN.pthx' convention)")
    ap.add_argument('--member-word', default='rep',
                    help="word between prefix and index, e.g. 'rep' (default) or 'run' "
                         "for '<prefix>_run3.pthx'-style naming")
    ap.add_argument('--no-zero-pad', action='store_true',
                    help="don't zero-pad the repeat index (run1, run2, ... instead of "
                         "run01, run02, ...)")
    args = ap.parse_args()

    paths = discover(args.glob, args.ckpt_name)
    if args.expect is not None and len(paths) != args.expect:
        raise SystemExit('expected %d repeats, found %d' % (args.expect, len(paths)))

    print('found %d checkpoints (%s):' % (len(paths), args.ckpt_name))
    for p in paths:
        print('   ', p)

    identical, per_member, ref = compare(paths)
    print('\ndeterminism across repeats (inner checkpoint members):')
    for member in COMPARED_MEMBERS:
        if member in per_member:
            print('   %-26s %s' % (member, 'identical' if per_member[member] else 'DIFFERS'))
    if not identical and not args.allow_divergent:
        raise SystemExit(
            '\nrepeats are NOT identical -- refusing to package. Investigate first, or pass '
            '--allow-divergent to package anyway.')

    os.makedirs(args.out_dir, exist_ok=True)

    # MANIFEST rows are computed before writing so the manifest can go inside the archive.
    member_prefix = args.member_prefix if args.member_prefix is not None else args.label
    idx_fmt = '%d' if args.no_zero_pad else '%02d'
    arcname_fmt = '%s_' + args.member_word + idx_fmt + '.pthx'
    rows = [('repeat', 'source_dir', 'member_in_archive', 'pthx_sha256', 'weight_pth_sha256')]
    members = []
    for idx, p in enumerate(paths, start=1):
        arcname = arcname_fmt % (member_prefix, idx)
        rows.append((str(idx), os.path.dirname(p), arcname,
                     sha256_file(p), inner_hashes(p).get('weight.pth', '')))
        members.append((arcname, p))
    manifest = '\n'.join('\t'.join(r) for r in rows) + '\n'

    source_note = args.source_note or 'Source result directories are listed in MANIFEST.tsv.'
    member_pattern = arcname_fmt.replace('%s', member_prefix).replace('%02d', 'NN').replace('%d', 'N')
    readme = build_readme(args.tag, args.label, paths, identical, ref, source_note, member_pattern)

    zip_path = os.path.join(args.out_dir, '%s.zip' % args.label)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arcname, src in members:
            info = zipfile.ZipInfo(arcname, date_time=FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(src, 'rb') as fh:
                z.writestr(info, fh.read())
        for arcname, text in (('MANIFEST.tsv', manifest), ('README.md', readme)):
            info = zipfile.ZipInfo(arcname, date_time=FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, text)

    # The canonical single checkpoint that inference recipes load. Copied verbatim from
    # repeat 1 -- byte-for-byte the file that produced the reported metrics.
    canonical_path = os.path.join(args.out_dir, '%s.pthx' % args.tag)
    with open(paths[0], 'rb') as src, open(canonical_path, 'wb') as dst:
        dst.write(src.read())

    with open(os.path.join(args.out_dir, '%s_MANIFEST.tsv' % args.label), 'w') as fh:
        fh.write(manifest)

    print('\nwrote:')
    for path in (zip_path, canonical_path):
        print('   %-52s %10.1f MB  sha256 %s'
              % (path, os.path.getsize(path) / 1e6, sha256_file(path)))

    print('\nexperiments.json asset fields:')
    print('   %s_checkpoint  ->  filename %s.pthx   sha256 %s'
          % (args.tag, args.tag, sha256_file(canonical_path)))
    print('   %s_all_repeats ->  filename %s.zip    sha256 %s'
          % (args.tag, args.label, sha256_file(zip_path)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

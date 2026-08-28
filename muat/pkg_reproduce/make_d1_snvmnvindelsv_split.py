"""Build the d1_snvmnvindelsv split: d1's own train/val partition, minus the samples
missing SV data, pointing prep_path at the new tokenized files this tag will produce.

Reuses class_name/class_index straight from d1's existing split TSVs -- deliberately
NOT re-derived from the PCAWG histology file, which disagrees with d1's own labels on
7 samples (a pre-existing minor metadata inconsistency, not something to fix here).

Usage:
    python muat/pkg_reproduce/make_d1_snvmnvindelsv_split.py \
        --out-dir muat/pkg_reproduce/splits \
        --prep-dir data/pcawg_full_mutations/preprocessed \
        --uuid-list-out data/pcawg_full_mutations/uuid_list.txt
"""
import argparse
import csv
import glob
import os


def load_split(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def main():
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=os.path.join(repo, 'muat', 'pkg_reproduce', 'splits'))
    ap.add_argument('--prep-dir', default=os.path.join(repo, 'data', 'pcawg_full_mutations', 'preprocessed'))
    ap.add_argument('--uuid-list-out', default=os.path.join(repo, 'data', 'pcawg_full_mutations', 'uuid_list.txt'))
    ap.add_argument('--sv-dir', default=os.path.join(
        repo, 'data', 'PCAWG', 'consensus_sv', 'icgc', 'open'))
    args = ap.parse_args()

    splits_dir = os.path.join(repo, 'muat', 'pkg_reproduce', 'splits')
    train = load_split(os.path.join(splits_dir, 'pcawg_orig_train.tsv'))
    val = load_split(os.path.join(splits_dir, 'pcawg_orig_val.tsv'))

    sv_uuids = set(
        os.path.basename(f).split('.')[0]
        for f in glob.glob(os.path.join(args.sv_dir, '*.bedpe.gz'))
    )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.uuid_list_out), exist_ok=True)

    all_uuids = []
    for name, rows in (('train', train), ('val', val)):
        kept = []
        dropped = []
        for row in rows:
            uuid = os.path.basename(row['prep_path']).split('.')[0]
            if uuid in sv_uuids:
                kept.append((uuid, row['class_name'], row['class_index']))
                all_uuids.append(uuid)
            else:
                dropped.append(uuid)

        out_path = os.path.join(args.out_dir, 'pcawg_orig_snvmnvindelsv_{}.tsv'.format(name))
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['prep_path', 'class_name', 'class_index'])
            for uuid, class_name, class_index in kept:
                prep_path = os.path.join(args.prep_dir, '{}.muat.tsv'.format(uuid))
                writer.writerow([prep_path, class_name, class_index])

        print('{}: kept {}, dropped {} (no SV data) -> {}'.format(
            name, len(kept), len(dropped), out_path))

    with open(args.uuid_list_out, 'w') as f:
        for uuid in all_uuids:
            f.write(uuid + '\n')
    print('wrote {} uuids -> {}'.format(len(all_uuids), args.uuid_list_out))


if __name__ == '__main__':
    main()

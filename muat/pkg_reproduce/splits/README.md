# Reproduce split definitions

These TSVs pin the exact train / validation / test sample sets for each reproducible
experiment. They are tiny, version-controlled, and shipped inside the package so a
`muat reproduce <tag>` run is bit-for-bit defined.

## Files (Group D, open-access PCAWG)

- `pcawg_open_labels.tsv` — frozen ground-truth label manifest: the **benchmark
  universe**. 1,812 open-access ICGC samples across the 17 MuAt tumour types
  (tumour types with >20 tumours in the full PCAWG cohort). This file *defines* the
  sample set, independent of any cluster-local path.
- `pcawg_orig_train.tsv` / `pcawg_orig_val.tsv` — training / internal-validation
  samples for `d1` (retraining), **1,449 / 363**. `d2`–`d6` (inference) evaluate on
  `pcawg_orig_val.tsv` as their test set.

The `pcawg_orig_*` files are **generated** from the original MuAt per-fold split
files (`example_files/local_{train,val}_split_muat1.tsv`) by `../make_d1_origsplit.py`
— they reproduce the original paper's partition, not a re-drawn split. Regenerate with:

```bash
python muat/pkg_reproduce/make_d1_origsplit.py
```

> **Note on the test set.** The original MuAt partition puts all 1,812 samples into
> train+val, so it has no clean held-out test. `d2`–`d6` therefore evaluate on the
> validation split (`pcawg_orig_val.tsv`); this is an **in-sample** evaluation, by
> design, to match the original paper. `make_splits.py` + `pcawg_open_labels.tsv`
> remain in the tree as a deterministic *clean* (leakage-free) re-split generator,
> but are not wired to any current recipe.

## Format

### `pcawg_open_labels.tsv`
Tab-separated, one sample per row:

```
sample          class_name        class_index
<aliquot_uuid>  <tumour_type>     <int 0..16>
```

`sample` is the PCAWG tumour aliquot UUID.

### `pcawg_orig_{train,val}.tsv`
MuAt's native split schema, exactly what the dataloader expects:

```
prep_path                class_name      class_index
<sample>.muat.tsv        <tumour_type>   <int>
```

`prep_path` is stored as a **basename only** (no directory) so the splits are
portable across machines. The bundle ships one friendly `<sample>.muat.tsv`
(plain TSV) per sample — built by `../make_bundle.py` from the internal
`*.token.gc.genic.exonic.cs.tsv.gz` files. At run time `muat reproduce` resolves
each basename to `<cache>/<bundle_extract_subdir>/<basename>` from the downloaded
data bundle (`pcawg_open_preprocessed`, extracts to `pcawg_open_muat/`).

## Provenance

The label manifest was harvested from the published per-fold split files
(`example_files/local_*_muat1.tsv`), which together cover 1,812 distinct samples /
17 classes. The class set follows the original MuAt paper (>20 tumours per type in
full PCAWG); the 17-class list is inherited from the paper, not re-thresholded on
the open subset. See `../experiments.md`.

# Public GDC cohort: native hg38 training and inference

A from-scratch training/inference demo on **public, open-access** GRCh38 data,
using the muat v0.1.23 CLI as released, no code changes required for the
pipeline itself. This exists to demonstrate muat's native-hg38 support (build a
dictionary from the data, tokenize, train, predict) on real, independently
downloadable data.

This recipe instead uses **GDC (Genomic Data Commons) open-access "Masked
Somatic Mutation" MAFs** — per-aliquot, consensus-caller somatic calls, no
dbGaP/DUA required, uniformly realigned to GRCh38 by GDC. Six TCGA cohorts
with visually distinct mutational signatures were picked for a clean
multi-class demo: `TCGA-SKCM` (melanoma, UV), `TCGA-LUSC` (lung squamous,
smoking), `TCGA-BRCA` (breast, APOBEC), `TCGA-COAD` (colorectal, MSI),
`TCGA-PRAD` (prostate, low burden), `TCGA-OV` (ovarian, HRD).

```bash
HG38=data/genome_reference/hg38.fa
OUT=data/hg38_public_demo
```

## Step 1 — fetch

```bash
python muat/pkg_reproduce/fetch_public_hg38_cohort.py \
  --projects TCGA-SKCM TCGA-LUSC TCGA-BRCA TCGA-COAD TCGA-PRAD TCGA-OV \
  --n-per-project 100 --out-dir $OUT
```

Writes `$OUT/gdc_manifest.json` (one entry per picked case) and downloads each
case's MAF to `$OUT/gdc_maf/<file_id>.maf.gz`. At most one aliquot per case is
picked, so there is no patient overlap across projects or later splits by
construction. This step is **not** required to be byte-reproducible (it's a
live API query); the split step below is.

## Step 2 — convert MAF to per-sample VCF

```bash
python muat/pkg_reproduce/convert_gdc_maf_to_vcf.py \
  --manifest $OUT/gdc_manifest.json --maf-dir $OUT/gdc_maf \
  --out-dir $OUT/vcf --labels-out $OUT/labels.json
```

## Step 3 — patient-level split

```bash
python muat/pkg_reproduce/make_public_hg38_splits.py \
  --labels $OUT/labels.json --tokenized-dir $OUT/preprocessed \
  --out-dir $OUT --seed 1337 --train-frac 0.7 --val-frac 0.15
```

Per-class stratified 70/15/15 split, pure-Python `random.Random` with a fixed
seed — deterministic given the same `labels.json`. Writes `$OUT/all_vcfs.txt`
(every kept sample, for the corpus-level dictionary step below) and
`$OUT/{train,val,test}_split.tsv` (`prep_path`/`class_name`/`class_index`,
`prep_path` pointing at the *tokenized* file the next step produces —
`--tokenized-dir` must match wherever step 4's `--tmp-dir` is).

## Step 4 — preprocess (annotate + build-dictionary + tokenize, one command)

```bash
muat preprocess --vcf --hg38 $HG38 \
  --build-dictionary --dictionary-which pos,motif,ges \
  --dictionary-suffix _pubhg38 --motif-labels inherit \
  --dictionary-out-dir $OUT/dicts/ \
  --input-list $OUT/all_vcfs.txt \
  --tmp-dir $OUT/preprocessed/
```

Run this as a **single, non-parallel call** over the full cohort (train, val, test together) — dictionary-building is corpus-level, every sample must be
seen before token ids are assigned, so splitting it across parallel jobs would
give each job its own vocabulary.

## Step 5 — train from scratch

```bash
muat train from-scratch \
  --mutation-type snv \
  --use-motif --use-position --use-ges \
  --train-split-filepath $OUT/train_split.tsv \
  --val-split-filepath $OUT/val_split.tsv \
  --motif-dictionary-filepath $OUT/dicts/dictMutation_pubhg38.tsv \
  --position-dictionary-filepath $OUT/dicts/dictChpos_pubhg38.tsv \
  --ges-dictionary-filepath $OUT/dicts/dictGES_pubhg38.tsv \
  --save-dir $OUT/model/ \
  --epoch 100
```

`--mutation-type snv`: this cohort is SNV-only by construction (step 2's scope limitation). 
All three `--*-dictionary-filepath` flags point at the `_pubhg38` dictionaries just
built; they get embedded in the resulting `.pthx` checkpoint, so inference
below doesn't need to pass them again.

## Step 6 — predict on the held-out test set

```bash
muat predict from-checkpoint \
  --ckpt-filepath $OUT/model/<best_checkpoint>.pthx \
  --input-list $OUT/test_vcfs.txt \
  --hg38 $HG38 \
  --result-dir $OUT/predict_test/
```

(`$OUT/test_vcfs.txt` is the raw-VCF-path equivalent of `test_split.tsv`'s
samples — extract it from `$OUT/labels.json`/`class_index.json` the same way
`all_vcfs.txt` was built in step 3.) The checkpoint carries its own
motif/position/ges dictionaries, so this re-annotates and re-tokenizes the raw
test VCFs internally with the training vocabulary. Score the result with
`muat.metrics` by adding a `target_name` column from `class_index.json`
first — `prediction_first_logits.tsv` has no ground truth on its own.

All outputs live under `data/hg38_public_demo/`

# iCAN WES: native hg38 training and inference

Training MuAt from scratch on the iCAN WES cohort, entirely in **native hg38**
coordinates (no liftover to hg19) — using the muat v0.1.23 CLI as released, no code
changes required for the pipeline itself.

Everything below assumes:

```bash
HG38=genome_reference/hg38.fa
OUT=data/ican
```

## Prerequisites

1. **iCAN WES VCFs**: PASS-filtered somatic calls, tumor/normal, GRCh38 coordinates.
2. **A labels table**: sample ID → cancer (sub)type.
3. **A patient-level train/val/test split** — the same patient must not appear in more
   than one of the three.
4. One combined list of every VCF that will ever be tokenized, **train + val + test
   together**:

```bash
mkdir -p $OUT
cat train_vcfs.txt val_vcfs.txt test_vcfs.txt > $OUT/ican_full_cohort_vcfs.txt
```

Dictionary-building is corpus-level — every sample must be seen before token ids are
assigned — so building from the full cohort up front, rather than train alone, avoids
val/test samples later hitting vocabulary the dictionary never saw.

## Preprocessing (annotate + build-dictionary + tokenize, one command)

The shipped position dictionary (`dictChpos.tsv`) was built from hg19 PCAWG bins and is
not valid for hg38. `--build-dictionary` derives `pos`/`motif`/`ges` directly from the
iCAN corpus instead — native hg38, no `--liftover` (that flag only exists to reuse the
hg19-vocabulary pretrained checkpoints):

```bash
muat preprocess --vcf --hg38 $HG38 \
  --build-dictionary --dictionary-which pos,motif,ges \
  --dictionary-suffix _ican --motif-labels inherit \
  --dictionary-out-dir $OUT/dicts/ \
  --input-list $OUT/ican_full_cohort_vcfs.txt \
  --tmp-dir $OUT/preprocessed/
```

This one call annotates every VCF, derives `dictChpos_ican.tsv` / `dictMutation_ican.tsv`
/ `dictGES_ican.tsv` from that annotated corpus, then tokenizes the same corpus with the
dictionaries it just built — so the tokens and the dictionaries cannot disagree. It prints
the three resolved `--*-dictionary-filepath` flags at the end; you need all three at
train time, or the defaults silently reintroduce the shipped (hg19) ones.

Two things worth knowing about what it does under the hood:
- `pos`/`ges` are fully corpus-derived — they cover everything in `$OUT/preprocessed/`.
  `motif` uses `--motif-labels inherit` (the default, passed explicitly for clarity): a
  motif's `mut_type` label comes from the *shipped* PCAWG dictionary, and any iCAN motif
  not found there is dropped rather than given an invented label. This only affects
  MNV/indel motifs in practice — the 96 trinucleotide SNV substitution types are
  exhaustive, so no new SNV motif can exist. Use `--motif-labels hybrid` instead if you'd
  rather derive labels for new motifs from ref/alt allele length than drop them; check
  `$OUT/dicts/dictMutation_ican.provenance.txt` either way.
- Any row whose motif, position, or ges fails to map is written to a sibling
  `<sample>.preperror.tsv` instead of `<sample>.muat.tsv` — automatic, not opt-in.
  `<sample>.muat.tsv` only ever contains fully-mapped rows. Check the printed vocabulary
  coverage summary and `$OUT/preprocessed/unmapped_vocab.tsv`: since the dictionaries were
  built from this exact cohort, `position` and `ges` should read `all mapped`; a non-zero
  count there means a sample was missing from the cohort list above, which is a data
  problem to fix, not something to wave off. Non-zero `motif` counts are expected (see
  above).

## Split TSV

Three columns, `prep_path` / `class_name` / `class_index`, pointing at the `.muat.tsv`
files above (format: `example_files/train_split_example.tsv`). Only train/val patients go
here — hold the test set out for inference.

```
prep_path	class_name	class_index
data/ican/preprocessed/<sample1>.muat.tsv	<CancerType>	0
data/ican/preprocessed/<sample2>.muat.tsv	<CancerType>	1
...
```

Write `$OUT/train_split.tsv` and `$OUT/val_split.tsv` this way.

## Train from scratch — snv+mnv+indel, motif + position + ges

```bash
muat train from-scratch \
  --mutation-type snv+mnv+indel \
  --use-motif --use-position --use-ges \
  --train-split-filepath $OUT/train_split.tsv \
  --val-split-filepath $OUT/val_split.tsv \
  --motif-dictionary-filepath $OUT/dicts/dictMutation_ican.tsv \
  --position-dictionary-filepath $OUT/dicts/dictChpos_ican.tsv \
  --ges-dictionary-filepath $OUT/dicts/dictGES_ican.tsv \
  --save-dir $OUT/model/ \
  --epoch 100
```

- `--mutation-type snv+mnv+indel` matches the ceiling muat's own pretrained WES benchmark
  uses — WES doesn't reliably call structural variants or MEI, and the `Neg`
  negative-sampling category needs whole-genome background WES doesn't have.
- All three `--*-dictionary-filepath` flags are passed explicitly, pointing at the
  `_ican` dictionaries just built — omitting any one of them falls back to the shipped
  hg19 dictionary for that kind. They get embedded inside the resulting `.pthx`
  checkpoint, so inference below doesn't need to pass them again.

## Results

For inference on the held-out test set:

```bash
muat predict from-checkpoint \
  --ckpt-filepath $OUT/model/<best_checkpoint>.pthx \
  --input-list test_vcfs.txt \
  --hg38 $HG38 \
  --result-dir $OUT/predict_test/
```

The checkpoint carries its own motif/position/ges dictionaries, so this re-annotates and
re-tokenizes the raw test VCFs internally with the same vocabulary used in training.
Output lands in `$OUT/predict_test/*/prediction_first_logits.tsv` (per-sample predicted
class + logits; no ground truth unless you add a `target_name` column yourself before
scoring with `muat.metrics`).
# muat Experiment Tracking

## Group A — Reproduce Sanjaya et al. 2023 (Controlled-Access Data)

| Tag | Purpose | Mode | Cohort | Split | Input Type | Checkpoint in | Checkpoint produced | Environment | Performance metrics | Reference |
|-----|---------|------|--------|-------|-----------|--------------|-------------------|-------------|--------------------:|-----------|
| a1 | Reproduce Sanjaya et al. 2023 results using muat on controlled-access data | Train | PCAWG-controlled access | Train | SNV+MNV+indels+SV+MEI+pos+ges | None | a1 | CSC Puhti GPU | | Sanjaya et al. 2023 |
| a2 | *(same as above)* | Inference | PCAWG-controlled access | Test | *(same)* | a1 | None | CSC Puhti CPU bioconda | | Sanjaya et al. 2023 |
| a3 | | Inference | PCAWG-controlled access | Test | | a1 | None | CSC Puhti CPU docker | | Sanjaya et al. 2023 |
| a4 | | Inference | PCAWG-controlled access | Test | | a1 | None | GEL bioconda | | Sanjaya et al. 2023 |
| a5 | | Inference | PCAWG-controlled access | Test | | a1 | None | GEL docker | | Sanjaya et al. 2023 |

---

## Group B — Cross-Cohort Portability: PCAWG → GEL, Fine-Tuning

| Tag | Purpose | Mode | Cohort | Split | Input Type | Checkpoint in | Checkpoint produced | Environment | Performance metrics | Reference |
|-----|---------|------|--------|-------|-----------|--------------|-------------------|-------------|--------------------:|-----------|
| b1 | Cross-cohort portability PCAWG→GEL, fine-tuning | Inference | GEL | Full set | SNV+MNV+indels+SV+MEI+pos+ges | a1 | None | GEL bioconda | | Sanjaya & Pitkänen 2026 |
| b2 | | Fine-tuning | GEL | Train | | a1 | b2 | GEL bioconda | | Sanjaya & Pitkänen 2026 |
| b3 | | Inference | GEL | Test | | b2 | None | GEL bioconda | | Sanjaya & Pitkänen 2026 |

---

## Group C — GEL Fine-Tuned Model Back on PCAWG

| Tag | Purpose | Mode | Cohort | Split | Input Type | Checkpoint in | Checkpoint produced | Environment | Performance metrics | Reference |
|-----|---------|------|--------|-------|-----------|--------------|-------------------|-------------|--------------------:|-----------|
| c1 | GEL fine-tuned model back on PCAWG | Inference | PCAWG-controlled access | Test | SNV+MNV+indels+SV+MEI+pos+ges | b2 | None | GEL bioconda | | Sanjaya & Pitkänen 2026 |

---

## Group D — Reproducibility Across Environments (Open-Access Data)

A 2×2 matrix — {train, inference} × {same, different environment} — plus an hg38 pair.
Every tag uses **seed 1337**: attributing a difference to the environment requires the
seed to be held fixed, and varying both would confound them.

| Tag | Purpose | Mode | Cohort | Split | Input Type | Checkpoint in | Checkpoint produced | Environment | Performance metrics | Reference |
|-----|---------|------|--------|-------|-----------|--------------|-------------------|-------------|--------------------:|-----------|
| d1 | Training-run reproducibility, same environment | Train (10× same seed) | PCAWG-open access (hg19) | 80:20 — 1449 train / 363 test | SNV+MNV+pos+ges | None | ckpt1 | CSC Puhti GPU, bioconda | see report workbook | This work |
| d2 | Training-run reproducibility, different environment | Train (10× same seed) | PCAWG-open access (hg19) | 80:20 — 1449 train / 363 test | SNV+MNV+pos+ges | None | ckpt2 | CSC Puhti CPU, docker | see report workbook | This work |
| d3 | Inference-run reproducibility, same environment | Inference (10× same seed) | PCAWG-open access (hg19) | Test (363) | SNV+MNV+pos+ges | ckpt1 | None | CSC Puhti GPU, bioconda | see report workbook | This work |
| d4 | Inference-run reproducibility, different environment | Inference (10× same seed) | PCAWG-open access (hg19) | Test (363) | SNV+MNV+pos+ges | ckpt1 | None | CSC Puhti CPU, docker | see report workbook | This work |
| d5 | Training with GRCh38 — *not runnable yet* | Train (10× same seed) | PCAWG-open access (hg19 calls lifted to hg38) | 80:20 — 1449 train / 363 test | SNV+MNV+pos+ges | None | ckpt3 | CSC Puhti GPU, bioconda | — | This work |
| d6 | Inference with GRCh38 — *not runnable yet* | Inference (10× same seed) | PCAWG-open access (hg19 calls lifted to hg38) | Test (363) | SNV+MNV+pos+ges | ckpt3 | None | CSC Puhti GPU, bioconda | — | This work |

**What "reproducible" means here, precisely.** Two different claims, which must not be
conflated:

- *Within* an environment, the repeats of one tag are expected to be **identical** —
  same logits, same predictions, same weight tensors.
- *Across* environments (d1 vs d2, d3 vs d4), they will **not** be bit-identical. CPU and
  GPU float32 kernels reduce in different orders, so bit-equality across devices is not
  achievable and is not claimed. Those pairs are scored against a tolerance fixed before
  the runs, using `muat/pkg_reproduce/compare_environments.py`.

Results, per-repeat figures and the cross-environment comparison are recorded in
`example_files/local_checkpoint_reports_v2.xlsx`, which is generated from the run
directories by `muat/pkg_reproduce/make_report_workbook.py`.

**d5/d6 are not runnable yet.** No hg38 bundle exists, and the shipped position dictionary
(`muat/extfile/dictChpos.tsv`) is hg19-derived; see the `_hg38_note` on tag d5 in
`experiments.json`. Note also that PCAWG open-access variants were called against hg19, so
this arm measures the effect of a liftover round-trip rather than performance on natively
hg38-called data.

---

## Reproducing these experiments

Each tag above is runnable through the `muat reproduce` CLI, which pins the data
splits, hyperparameters, checkpoint and random seed for that experiment (recipes
live in `muat/pkg_reproduce/experiments.json`).

Because compute nodes are often offline, asset download is **separated from the run**:

```bash
# 1. On a node WITH internet (e.g. an HPC login node), stage assets into a shared cache:
muat fetch d3 --cache-dir /path/to/shared/cache

# 2. In the (possibly offline) compute job, run purely from the cache:
muat reproduce d3 --cache-dir /path/to/shared/cache --result-dir ./results/d3
```

- `--cache-dir` defaults to `$MUAT_CACHE`, then `~/.cache/muat`. On HPC point it at a
  **shared** path visible to both login and compute nodes.
- By default Group D uses the **preprocessed** open-access bundle (fast, offline-clean).
  Pass `--from-raw` to download raw PCAWG data and run preprocessing in-node, which
  additionally exercises the preprocessing pipeline across environments.
- The Docker image runs `muat fetch` at build time, so `muat reproduce` works offline
  inside the container with no extra step.
- Groups A–C use controlled-access (PCAWG-controlled / GEL) data and are **not**
  externally downloadable; their recipes resolve to user-provided paths.

Useful flags: `muat reproduce --list` (show all tags), `muat reproduce <tag> --dry-run`
(print the resolved command without running).

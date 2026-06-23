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

| Tag | Purpose | Mode | Cohort | Split | Input Type | Checkpoint in | Checkpoint produced | Environment | Performance metrics | Reference |
|-----|---------|------|--------|-------|-----------|--------------|-------------------|-------------|--------------------:|-----------|
| d1 | Reproducibility across environments using open-access data | Train | PCAWG-open access | Train | SNV+MNV+pos+ges | None | d1 | CSC Puhti GPU | | This work |
| d2 | | Inference | PCAWG-open access | Test | | d1 | None | CSC Puhti CPU bioconda | | This work |
| d3 | | Inference | PCAWG-open access | Test | | d1 | None | CSC Puhti CPU docker | | This work |
| d4 | | Inference | PCAWG-open access | Test | | d1 | None | iCAN bioconda | | This work |
| d5 | | Inference | PCAWG-open access | Test | | d1 | None | iCAN docker | | This work |
| d6 | Run-level reproducibility | Inference (10× same seed) | PCAWG-open access | Test | | d1 | None | CSC Puhti CPU bioconda | | This work |

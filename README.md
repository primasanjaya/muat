# MuAt — Mutation Attention

[![Latest Release](https://img.shields.io/github/v/release/primasanjaya/muat)](https://github.com/primasanjaya/muat/releases)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/muat)](https://anaconda.org/bioconda/muat)
[![License](https://img.shields.io/github/license/primasanjaya/muat)](https://github.com/primasanjaya/muat/blob/main/LICENSE)
[![Conda Downloads](https://img.shields.io/conda/dn/bioconda/muat)](https://anaconda.org/bioconda/muat)
[![Issues](https://img.shields.io/github/issues/primasanjaya/muat)](https://github.com/primasanjaya/muat/issues)
[![Stars](https://img.shields.io/github/stars/primasanjaya/muat?style=social)](https://github.com/primasanjaya/muat)

**MuAt predicts the tumour type of a sample from its somatic mutations.**

Give it a VCF, MAF or TSV of somatic calls. MuAt represents each mutation by its
sequence motif, genomic position, and genic/exonic/strand context, then a transformer
reads the whole mutation set and outputs a probability per tumour type. Pretrained
whole-genome and whole-exome checkpoints download automatically, so prediction needs no
training. You can also train your own model.

```bash
mamba create -n muat-env -c conda-forge -c bioconda muat && mamba activate muat-env
muat predict pretrained wgs --mutation-type 'snv+mnv' \
  --input-filepath sample.muat.tsv --result-dir results
```

## Requirements

| | |
|---|---|
| **Operating system** | Linux, CPU & GPU (tested on CentOS 7); macOS, CPU only; Windows via WSL2 only |
| **Python** | 3.9 (>= 3.8 supported), with `conda` or `mamba` installed |
| **PyTorch** | 2.5.1 tested; installed automatically as a dependency |
| **CUDA** (GPU only) | 12.0 tested (driver 530.30.02). Must be **≤** your NVIDIA driver's maximum CUDA |
| **Hardware** | Runs on CPU; a CUDA-capable GPU is optional, and recommended for training |
| **Memory / runtime** | <!-- TODO: benchmark inference on a typical WGS genome --> |

Native Windows is not supported: the `bedtools`/`htslib`/`bcftools`/`bedops` dependencies
have no `win-64` conda builds. Use WSL2, where the Windows-side NVIDIA driver provides
GPU access — do not install a driver inside WSL.

## Installation

MuAt uses the GPU automatically when one is available and falls back to CPU otherwise.
The commands are identical either way, so choose an install path by convenience.

### CPU — bioconda

```bash
mamba create -n muat-env -c conda-forge -c bioconda muat
mamba activate muat-env
muat -h
```

`conda create` also works, just slower. A plain install resolves the CPU build of PyTorch.

### GPU — bioconda

Run `nvidia-smi` and read **CUDA Version** in the top-right: that is the maximum your
driver supports. Request that value or lower.

```bash
# replace 12.1 with your driver's maximum
mamba create -n muat-gpu -c conda-forge -c bioconda muat pytorch-gpu "cuda-version=12.1"
mamba activate muat-gpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A CUDA build and `True` mean the GPU is active.

<details>
<summary><b>GPU install troubleshooting</b> — silent CPU fallback, login nodes, older Linux</summary>

- **`cuda-version` must be ≤ your driver's maximum**, or the solver silently installs the
  CPU build with no error. Installing plain `muat` without `pytorch-gpu` also gives CPU.
- **Installing on a login node with no GPU** (common on HPC) fails with
  `__cuda ... is missing on the system`, because conda detects CUDA from the driver.
  Prefix the command with `CONDA_OVERRIDE_CUDA=12.1`.
- **On older Linux** (CentOS 7, glibc 2.17) the newest PyTorch builds are unavailable;
  conda will resolve an older compatible one.

</details>

### Containers

**Prebuilt, CPU** — bioconda publishes an image for every release. Pick the current tag
from [quay.io/biocontainers/muat](https://quay.io/repository/biocontainers/muat?tab=tags):

```bash
TAG=0.1.20--pyh106432d_0

docker pull quay.io/biocontainers/muat:$TAG
docker run  quay.io/biocontainers/muat:$TAG muat -h
```

On HPC, skip Docker — Galaxy mirrors the same images as ready-made Singularity images
(~720 MB), so one download is all you need:

```bash
wget https://depot.galaxyproject.org/singularity/muat:$TAG -O muat.sif
apptainer exec muat.sif muat -h

# or build the .sif from the registry, no Docker daemon involved
apptainer build muat.sif docker://quay.io/biocontainers/muat:$TAG
```

**Build locally** — required for GPU, since BioContainers images are built on CPU
builders and always contain the CPU build of PyTorch:

```bash
git clone https://github.com/primasanjaya/muat.git && cd muat

./build_docker.sh                       # GPU, auto-detects the driver's CUDA
CUDA_VERSION=12.1 ./build_docker.sh     # GPU, pinned explicitly
./build_docker.sh cpu                   # slim CPU-only image
```

The GPU image also runs on CPU-only hosts, so one image covers both.

**Running** — pass `--gpus all` (Docker) or `--nv` (Apptainer) to expose the GPU, and
mount your data with `-v /data:/data` or `--bind /data`:

```bash
docker run --gpus all muat:v0.1.21 predict pretrained wgs ...
apptainer run  --nv muat.sif predict pretrained wgs ...
```

> ⚠️ Whether you name `muat` in the command depends on the image. The locally-built image
> sets `muat` as its entrypoint, so you **omit** it. BioContainers images use a
> conda-activator entrypoint, so you **include** it. `apptainer exec` ignores entrypoints
> entirely, so include it there too.

> **Note:** an image runs wherever you `docker run` it, not only on the build host. If you
> build on a login node but run on GPU nodes, pin `CUDA_VERSION` to the lowest CUDA across
> your run nodes rather than relying on auto-detect.

## Quick Test

Two tests, cheapest first. Both use files that ship in `example_files/`. Run them from the
repo root, or substitute absolute paths.

### 1. Predict a preprocessed sample — no reference genome needed

`0a6be23a-....muat.tsv` is already preprocessed (4,219 SNVs), so MuAt skips preprocessing
and no `--hg19`/`--hg38` is required. The only download is the benchmark checkpoint:

```bash
muat predict pretrained wgs --mutation-type 'snv+mnv' \
  --input-filepath example_files/0a6be23a-d5a0-4e95-ada2-a61b2b5d9485.muat.tsv \
  --result-dir results
```

### 2. Predict a raw VCF — exercises preprocessing too

This needs the hg19 FASTA, roughly 3 GB to download and 3 GB again once unpacked:

```bash
mkdir -p genome_reference
curl -L -o genome_reference/hg19.fa.gz \
  https://ftp.sanger.ac.uk/pub/project/PanCancer/genome.fa.gz
gunzip genome_reference/hg19.fa.gz

muat predict pretrained wgs --mutation-type 'snv+mnv' \
  --hg19 genome_reference/hg19.fa \
  --input-filepath example_files/0a6be23a-d5a0-4e95-ada2-a61b2b5d9485.consensus.20160830.somatic.snv_mnv.vcf.gz \
  --result-dir results
```

## Usage

### Inputs and models

MuAt infers what to do from the file suffix:

| Input | Behaviour |
|---|---|
| `.vcf{,.gz}`, `.maf{,.gz}`, `.tsv` | preprocessed first; **requires** `--hg19` or `--hg38` |
| `.muat.tsv{,.gz}` | used as-is; the reference flag must be **omitted** |

All inputs in a single call must be the same kind — mixed batches are rejected.

`predict` and `predict-ensemble` each take the model from one of two sources:

| Source | Meaning |
|---|---|
| `pretrained {wgs,wes}` | benchmark checkpoint, downloaded automatically from HuggingFace |
| `from-checkpoint` | your own `.pthx` files; the assay is inferred from each checkpoint |

### Predict

```bash
# hg38 instead of hg19
muat predict pretrained wgs --mutation-type 'snv+mnv' \
  --hg38 /path/to/genome_reference/hg38.fa \
  --input-filepath sample.vcf.gz --result-dir results

# your own checkpoint
muat predict from-checkpoint --ckpt-filepath my_model.pthx \
  --hg19 /path/to/genome_reference/hg19.fa \
  --input-filepath sample.vcf.gz --result-dir results
```

For preprocessed `.muat.tsv` input, see [Quick Test 1](#1-predict-a-preprocessed-sample--no-reference-genome-needed);
to produce such files yourself, see the
[preprocessing guide](https://github.com/primasanjaya/muat/blob/main/documentation/README_preprocessing.md).

### Ensemble prediction

Logits are averaged across the fold checkpoints.

```bash
# benchmark ensemble, downloaded automatically
muat predict-ensemble pretrained wgs --mutation-type 'snv+mnv' \
  --hg19 /path/to/genome_reference/hg19.fa \
  --input-filepath sample.vcf.gz --result-dir results

# your own folds — one .pthx per fold
muat predict-ensemble from-checkpoint \
  --ckpt-filepath fold0.pthx fold1.pthx fold2.pthx \
  --hg19 /path/to/genome_reference/hg19.fa \
  --input-list inputs.txt --result-dir results
```

### Reproduce a published experiment

Experiments from the paper are pinned by tag in [experiments.md](experiments.md). Asset
download is separate from the run, so compute nodes without internet still work:

```bash
muat reproduce --list             # available tags
muat reproduce d1 --dry-run       # show the resolved recipe, no compute
muat fetch d1                     # on a machine WITH internet: stage assets into the cache
muat reproduce d1                 # runs offline from the cache
```

### Training

See the [training guide](https://github.com/primasanjaya/muat/blob/main/documentation/README_MuAtTraining.md),
or [README_PCAWG.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_PCAWG.md)
for full PCAWG training.

## Additional Resources

- [**Download PCAWG**](https://github.com/primasanjaya/muat/blob/main/documentation/README_download.md) — obtaining the PCAWG dataset
- [**Preprocessing**](https://github.com/primasanjaya/muat/blob/main/documentation/README_preprocessing.md) — turning raw calls into `.muat.tsv`
- [**Training**](https://github.com/primasanjaya/muat/blob/main/documentation/README_MuAtTraining.md) — general training instructions
- [**PCAWG training**](https://github.com/primasanjaya/muat/blob/main/documentation/README_PCAWG.md) — full training on PCAWG
- [**Genomics England**](https://github.com/primasanjaya/muat/blob/main/documentation/README_GEL.md) — training and prediction on the GEL dataset
- [**experiments.md**](experiments.md) — pinned experiment recipes for `muat reproduce`

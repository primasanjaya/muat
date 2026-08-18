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

## Requirements

| | |
|---|---|
| **Operating system** | Linux, CPU & GPU (verified on CentOS 7 / glibc 2.17 and Debian 12); macOS, CPU only; Windows via WSL2 only |
| **Python** | 3.9–3.14. conda resolves one for you — you do not need to pick a version |
| **Conda** | `conda` or `mamba`. Any current version; see the note on very old `mamba` below |
| **PyTorch** | Installed automatically. Verified with 2.13.0 (CPU) and 2.5.1 (CUDA 12.0) |
| **CUDA** (GPU only) | Must be **≤** your NVIDIA driver's maximum CUDA. 12.0/12.1 verified on driver 530.30.02 |
| **Hardware** | Runs on CPU; a CUDA-capable GPU is optional, and recommended for training |
| **Memory** | ≈1.2 GB RAM to predict; ≈2.8 GB GPU memory to train — a 4 GB card suffices (default hyperparameters)|
| **Disk** | ≈2.9 GB for the CPU environment, ≈8.5 GB for the GPU environment |

### Compute cost

Measured on the runs behind this release, not estimated. Training figures are the mean of ten
identically seeded repeats on one node.

| Task | Hardware | Wall-clock | Peak memory |
|---|---|---|---|
| Predict 1 preprocessed WGS sample | CPU, 4 threads | ≈14 s | ≈1.2 GB RAM |
| Train 100 epochs, 1449 samples | Tesla P100-16GB | ≈1 h 29 min (range 1:28–1:32) | ≈2.8 GB GPU |
| Train 100 epochs, 1449 samples | CPU, 4 threads | ≈25× slower per epoch — use a GPU | not yet measured |

Prediction is cheap enough to need no GPU: the 14 s above is dominated by interpreter start-up
and loading the checkpoint, not by inference. Training on CPU works and is bit-reproducible, but
it is the slow path — the GPU figure is per repeat, so a ten-repeat reproducibility run is ~15 h
on one P100.

Native Windows is not supported: the `bedtools`/`htslib`/`bcftools`/`bedops` dependencies
have no `win-64` conda builds. Use WSL2, where the Windows-side NVIDIA driver provides
GPU access — do not install a driver inside WSL.

## Installation

MuAt uses the GPU automatically when one is available and falls back to CPU otherwise.
The only choice below is whether to pull in the CUDA build of PyTorch.

### CPU — bioconda

```bash
mamba create -n muat-env -c conda-forge -c bioconda muat
mamba activate muat-env
muat -h
```

`mamba create` is a faster drop-in for `conda create`. A plain install resolves the CPU build
of PyTorch, and needs no CUDA, driver or GPU.

### GPU — bioconda

Run `nvidia-smi` and read **CUDA Version** in the top-right: that is the maximum your
driver supports. Request that value or lower.

```bash
# replace 12.1 with your driver's maximum
mamba create -n muat-gpu -c conda-forge -c bioconda muat pytorch-gpu "cuda-version=12.1"
mamba activate muat-gpu
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A CUDA build and `True` mean the GPU is active. Installing on a machine that has no GPU — a
login node, say — is fine and prints `False`; the environment is still correct and still runs
on CPU. See the troubleshooting note if the *solve itself* fails there.

### Verify the installation
```bash
usage: muat [-h]
            {download,preprocess,predict,train,predict-ensemble,reproduce,fetch} ...

Mutation Attention Tool

positional arguments:
  {download,preprocess,predict,train,predict-ensemble,reproduce,fetch}
                        Available commands
    download            Download the dataset or a reference genome.
    preprocess          Preprocess the dataset (raw -> annotated ->
                        tokenized).
    predict             Predict samples with a single model.
    train               Train the MuAt model.
    predict-ensemble    Run ensemble prediction (averages logits across fold
                        checkpoints).
    reproduce           Reproduce a pinned experiment (see experiments.md) by
                        tag, offline-safe.
    fetch               Download a reproduce tags assets into the cache (run
                        where there is internet).

options:
  -h, --help            show this help message and exit
```

This works for either environment and is the fastest end-to-end check: it downloads the
benchmark checkpoint and classifies a preprocessed sample that ships with the repo. No
reference genome is involved.

```bash
muat predict pretrained wgs --mutation-type 'snv+mnv' \
  --input-filepath example_files/0a6be23a-d5a0-4e95-ada2-a61b2b5d9485.muat.tsv \
  --result-dir results
```

Expected last lines — the prediction is deterministic, so the tumour type should match exactly:

```
0a6be23a-d5a0-4e95-ada2-a61b2b5d9485 is predicted to be Prost-AdenoCA
Results have been saved in results/
```

<details>
<summary><b>Install troubleshooting</b> — <code>install</code> vs <code>create</code>, silent CPU fallback, login nodes, older Linux, pinning</summary>

- **`Environment must first be created` / `Permission denied` / `No prefix found at ...`** —
  use `create`, or `install -n <env>`, rather than a bare `install`. A bare `conda install` /
  `mamba install` acts on the *currently active* environment, which on shared and HPC systems
  is normally a site-wide `base` you cannot write to. `conda info --base` shows which conda
  installation you are actually driving, and `conda env list` where new environments would be
  written; on clusters these often point at a shared, read-only installation rather than yours.
- **Always name `pytorch-gpu` for a GPU install.** A plain `muat` install resolves the **CPU**
  build of PyTorch — verified even on a machine with a working GPU and driver, because the
  solver prefers the CPU build regardless of hardware. This fails silently: you get a working
  environment that is simply ~50× slower.
- **Keep `cuda-version` ≤ your driver's maximum.** Above it, conda either refuses to solve or
  resolves an older CUDA build than you asked for, so check `torch.version.cuda` after
  installing rather than assuming you got the version you named.
- **Solving on a login node with no GPU** (common on HPC) fails with
  `__cuda ... is missing on the system`, because conda derives CUDA from the driver rather
  than from your request. Prefix the command with `CONDA_OVERRIDE_CUDA=12.1`, matching the
  `cuda-version` you asked for.
- **On older Linux** (CentOS 7, glibc 2.17) the newest **CUDA** builds of PyTorch are
  unavailable — they require `__glibc >=2.28` — so conda resolves 2.5.1 instead. **CPU**
  builds only need `__glibc >=2.17` and are unaffected. This is automatic; nothing to do.
- **A very old `mamba` (< 1.0, e.g. 0.11 from a 2021 site install)** cannot read current
  repodata and takes different `create` arguments. Check with `mamba --version`; if it is
  ancient, install [Miniforge](https://github.com/conda-forge/miniforge) into your home
  directory and use that instead.
- **Pinning for reproducibility.** The commands above deliberately float to the newest
  compatible stack. To fix it, name the versions:
  ```bash
  conda create -n muat-env -c conda-forge -c bioconda \
    muat=0.1.22 python=3.9 pytorch=2.5.1
  ```
  `muat-env.yml` in the repo is a fully version-locked **dependency** environment (it is what
  the Docker image builds from). It deliberately does not list `muat` itself, so install MuAt
  into it afterwards:
  ```bash
  conda env create -f muat-env.yml          # creates muat-env, dependencies only
  conda activate muat-env
  conda install -n muat-env -c conda-forge -c bioconda --no-deps muat=0.1.22
  ```

</details>

### Containers

**Prebuilt, CPU** — bioconda publishes an image for every release. Pick the current tag
from [quay.io/biocontainers/muat](https://quay.io/repository/biocontainers/muat?tab=tags):

```bash
TAG=0.1.22--pyh106432d_0

docker pull quay.io/biocontainers/muat:$TAG
docker run  quay.io/biocontainers/muat:$TAG muat -h
```

On HPC, skip Docker — Galaxy mirrors the same images as ready-made Singularity images
(~720 MB), so one download is all you need:

```bash
wget https://depot.galaxyproject.org/singularity/muat:$TAG -O muat.sif
apptainer exec muat.sif muat -h

# or pull the .sif straight from the registry, no Docker daemon involved
apptainer pull muat.sif docker://quay.io/biocontainers/muat:$TAG
```

> **If your cluster has `singularity` rather than `apptainer`**, substitute it verbatim —
> Apptainer is Singularity's renamed successor and these sub-commands are identical
> (`singularity pull …`, `singularity exec …`). Prefer `pull` over `build` for a registry
> image: it needs no root and no `--fakeroot`.

> **Note:** the Galaxy `.sif` mirror lags a release by a day or two, so `wget` may return 404
> for a brand-new tag while quay.io already has it. In that case use the `apptainer pull
> docker://...` form above, which fetches straight from quay.io, or pick an older tag.

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
docker run --gpus all muat:v0.1.22 predict pretrained wgs ...
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
muat download --reference --hg19 --download-dir genome_reference
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

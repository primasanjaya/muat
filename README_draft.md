# Mutation Attention


[![Latest Release](https://img.shields.io/github/v/release/primasanjaya/muat)](https://github.com/primasanjaya/muat/releases)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/muat)](https://anaconda.org/bioconda/muat)
[![License](https://img.shields.io/github/license/primasanjaya/muat)](https://github.com/primasanjaya/muat/blob/main/LICENSE)
[![Conda Downloads](https://img.shields.io/conda/dn/bioconda/muat)](https://anaconda.org/bioconda/muat)
[![Issues](https://img.shields.io/github/issues/primasanjaya/muat)](https://github.com/primasanjaya/muat/issues)
[![Stars](https://img.shields.io/github/stars/primasanjaya/muat?style=social)](https://github.com/primasanjaya/muat)

Conda package for Mutation Attention deep learning tool for tumour type and subtype classification

## Requirements

| | |
|---|---|
| **Operating system** | Linux (CPU & GPU); macOS (CPU only). Windows via WSL2 (CPU & GPU)|
| **Python** | 3.9 (>= 3.8 supported), conda or mamba installed |
| **CUDA (GPU only)** | <!-- TODO: confirm on target nodes, e.g. 11.8 --> with a matching NVIDIA driver |
| **Hardware** | Runs on CPU; a CUDA-capable GPU is optional and recommended for training |
| **Memory / runtime** | <!-- TODO: benchmark inference on a typical WGS genome --> |

## Quick Start

Install muat and all its dependencies in a single command (for CPU).

```bash
conda create -n muat-env -c conda-forge -c bioconda muat -y
conda activate muat-env
```
> 💡 If you have `mamba`, you can use `mamba create` instead of `conda create` for faster installation.

Verify the installation:
```bash
muat -h
```
You will see:
```
Mutation Attention Tool

positional arguments:
  {download,preprocess,predict,predict-ensemble,train}
                        Available commands
    download            Download the dataset.
    preprocess          Preprocess the dataset.
    predict             Predict samples with a single model.
    predict-ensemble    Run ensemble prediction (averages logits across fold checkpoints).
    train               Train the MuAt model.
```

## Quick Test

Run a prediction on the vcf using bundled checkpoint:

```bash
oneliner exact execution
```
> 💡 Use absolute paths (not relative paths) to ensure successful execution.

See [Usage](#usage) below for hg38, preprocessed inputs, custom checkpoints, and ensemble prediction.

## GPU Support 

To run on an NVIDIA GPU, install a CUDA build of PyTorch (`pytorch-gpu`). First
check the maximum CUDA your driver supports: run `nvidia-smi` and read the
"CUDA Version" in the top-right. Then request that value (or lower) as
`cuda-version`:

```bash
# replace 12.1 with the CUDA version your driver supports (e.g. 11.8)
conda create -n muat-gpu -c conda-forge -c bioconda muat pytorch-gpu "cuda-version=12.1" -y
conda activate muat-gpu
```

Verify the GPU is visible — you should see a CUDA build and `True`:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> ⚠️ The `cuda-version` you request must be **≤** your driver's max CUDA, or conda
> silently installs the CPU build. A plain `conda create ... muat` without
> `pytorch-gpu` also resolves to CPU.

## Docker / container installation
Build an image from source with the helper script. It auto-detects the GPU
driver and bakes in a compatible CUDA build, falling back to a CPU-only image
when no usable GPU is found:
```bash
./build_docker.sh                       # auto-detect GPU (CPU-only if none found)
./build_docker.sh cpu                   # force a slim CPU-only image
CUDA_VERSION=12.1 ./build_docker.sh     # force a specific CUDA build
```
The GPU image also runs on CPU-only hosts. Expose the GPU at run time:
```bash
docker run --gpus all muat:v0.1.20 muat predict ...     # Docker, using the GPU
docker run            muat:v0.1.20 muat predict ...      # same image, CPU only
```
On HPC/SPE systems without Docker, use Apptainer/Singularity (`--nv` exposes the GPU):
```bash
apptainer build muat.sif docker-daemon://muat:v0.1.20
apptainer run --nv muat.sif muat predict ...
```
A prebuilt CPU image is also available from [BioContainers](https://biocontainers.pro/tools/muat).

> **Note:** the image runs wherever you `docker run` it, not only on the build
> host. If you build on a login node but run on GPU nodes, pin `CUDA_VERSION` to
> the lowest CUDA across your run nodes rather than relying on auto-detect.

## Usage

Input mode is inferred from the file suffix:
- Raw inputs (`.vcf{,.gz}`, `.maf{,.gz}`, `.tsv`) are preprocessed first and require `--hg19` or `--hg38`.
- Preprocessed inputs (`.muat.tsv{,.gz}`) are used as-is; the reference flag must be omitted.
- All inputs in a single call must be the same kind (mixed batches are rejected).

### Predict a VCF written with hg38
```bash
muat predict pretrained wgs --mutation-type 'snv+mnv' --hg38 '/path/to/genome_reference/hg38.fa' --input-filepath 'path/to/sample.vcf.gz' --result-dir 'path/to/result_dir/'
```

### Predict preprocessed samples
Use the `.muat.tsv` (or `.muat.tsv.gz`) output of `muat preprocess` directly — no reference flag needed; the suffix tells muat to skip preprocessing. See the [preprocessing steps](https://github.com/primasanjaya/muat/blob/main/documentation/README_preprocessing.md).
```bash
muat predict pretrained wgs --mutation-type 'snv+mnv' --input-filepath 'path/to/sample.muat.tsv' --result-dir 'path/to/result_dir/'
```

### Predict with your own checkpoint
```bash
muat predict from-checkpoint --ckpt-filepath 'path/to/my_model.pthx' --hg19 '/path/to/genome_reference/hg19.fa' --input-filepath 'path/to/sample.vcf.gz' --result-dir 'path/to/result_dir/'
```

### Ensemble prediction (benchmark models)
Predict using the benchmark ensemble (auto-downloaded from HuggingFace):
```bash
muat predict-ensemble pretrained wgs --mutation-type 'snv+mnv' --hg19 '/path/to/genome_reference/hg19.fa' --input-filepath 'path/to/sample.vcf.gz' --result-dir 'path/to/result_dir/'
```

### Ensemble prediction (your own checkpoints)
Pass one `.pthx` per fold; logits are averaged across them. The assay (wgs/wes) is inferred from each checkpoint.
```bash
muat predict-ensemble from-checkpoint --ckpt-filepath 'path/fold0.pthx' 'path/fold1.pthx' 'path/fold2.pthx' --hg19 '/path/to/genome_reference/hg19.fa' --input-list 'path/to/inputs.txt' --result-dir 'path/to/result_dir/'
```

## Additional Resources
- **Download PCAWG:** Read [README_download.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_download.md) for details on downloading PCAWG Dataset.
- **Preprocessing:** Read [README_preprocessing.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_preprocessing.md) for details on preprocessing.
- **General Training:** Read [README_MuAtTraining.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_MuAtTraining.md) for general training instructions.
- **Full Training of PCAWG Dataset:** Read [README_PCAWG.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_PCAWG.md) for full training instructions on the PCAWG dataset.
- **Training and Predicting Genomics England Dataset:** Read [README_GEL.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_GEL.md) for complete training and prediction instructions on the Genomics England dataset.

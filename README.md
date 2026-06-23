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
| **Operating system** | Linux (tested on <!-- TODO: confirm distro/version, e.g. CentOS 7 / Rocky 8 -->); macOS supported for CPU-only use |
| **Python** | 3.9 (>= 3.8 supported) |
| **PyTorch** | <!-- TODO: confirm tested version, e.g. 2.x --> (installed automatically as a dependency) |
| **CUDA (GPU only)** | <!-- TODO: confirm on target nodes, e.g. 11.8 -->, with a matching NVIDIA driver |
| **Hardware** | Runs on CPU; a CUDA-capable GPU is optional and recommended for training |
| **Memory / runtime** | <!-- TODO: benchmark inference on a typical WGS genome --> |

muat runs on both CPU and GPU. The GPU is used automatically when one is
available (`torch.cuda.is_available()`); otherwise muat falls back to CPU and
the commands are identical either way.

## Quick Start

1. **Clone the muat Repository**
   ```bash
   git clone https://github.com/primasanjaya/muat.git
   ```

2. **Navigate to the muat Directory**.
   ```bash
   cd muat
   ```

3. **Create the Conda Environment**.<br>
   To create the conda environment, run:
   ```bash
   conda env create -f muat-env.yml
   ```

4. **Activate the Conda Environment**.<br>
   After creating the environment, activate it with:
   ```bash
   conda activate muat-env
   ```

5. **Install muat**<br>
   Install muat via bioconda channel
   ```bash
   conda install bioconda::muat
   ```

6. **Verify the Installation**<br>
   To test if the installation was successful, run:
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

### GPU support
muat uses the GPU automatically when one is available and falls back to CPU
otherwise — the commands are identical either way. After installing, check
which device you got:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
A CUDA build and `True` mean the GPU is active. If you see a CPU build or
`False` on a GPU machine, conda resolved the CPU build — force the CUDA build,
capping the version to what `nvidia-smi` reports your driver supports:
```bash
conda install -c conda-forge -c bioconda muat "pytorch=*=cuda*" "cuda-version<=11.8"
```

Both `predict` and `predict-ensemble` accept two sources:
- `pretrained {wgs,wes}` — auto-downloads the benchmark checkpoint(s) from HuggingFace.
- `from-checkpoint` — uses your own `.pthx` files; the assay is inferred from each checkpoint.

Input mode is inferred from file suffix:
- Raw inputs (`.vcf{,.gz}`, `.maf{,.gz}`, `.tsv`) are preprocessed first and require `--hg19` or `--hg38`.
- Preprocessed inputs (`.muat.tsv{,.gz}`) are used as-is; the reference flag must be omitted.
- All inputs in a single call must be the same kind (mixed batches are rejected).

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

## Quick Test
The example of SNV,MNV vcf file is in `example_files/0a6be23a-d5a0-4e95-ada2-a61b2b5d9485.consensus.20160830.somatic.snv_mnv.vcf.gz`.<br>
This file was written with hg19. To run prediction on this file, execute:

💡 **Tips**: use absolute paths (not relative paths) to ensure successful execution.

**Run the prediction (exactly using this command)**

```bash
(muat-env)$ muat predict pretrained wgs --mutation-type 'snv+mnv' --hg19 genome_reference/hg19.fa --input-filepath 'example_files/0a6be23a-d5a0-4e95-ada2-a61b2b5d9485.consensus.20160830.somatic.snv_mnv.vcf.gz' --result-dir results
```


### For VCF Files Written with hg38
To predict using VCF files written with hg38, run:
```bash
(muat-env)$ muat predict pretrained wgs --mutation-type 'snv+mnv' --hg38 '/path/to/genome_reference/hg38.fa' --input-filepath 'path/to/sample.vcf.gz' --result-dir 'path/to/result_dir/'
```

### Predicting preprocessed data samples (read preprocessing steps [here](https://github.com/primasanjaya/muat/blob/main/documentation/README_preprocessing.md))
Use the `.muat.tsv` (or `.muat.tsv.gz`) output of `muat preprocess` directly — no reference flag needed; the suffix tells muat to skip preprocessing.
```bash
(muat-env)$ muat predict pretrained wgs --mutation-type 'snv+mnv' --input-filepath 'path/to/sample.muat.tsv' --result-dir 'path/to/result_dir/'
```

### Predicting with your own checkpoint
```bash
(muat-env)$ muat predict from-checkpoint --ckpt-filepath 'path/to/my_model.pthx' --hg19 '/path/to/genome_reference/hg19.fa' --input-filepath 'path/to/sample.vcf.gz' --result-dir 'path/to/result_dir/'
```

## Run MuAt benchmark ensemble models
Example cli to predict samples using the benchmark ensemble (auto-downloaded from HuggingFace):
```bash
(muat-env)$ muat predict-ensemble pretrained wgs --mutation-type 'snv+mnv' --hg19 '/path/to/genome_reference/hg19.fa' --input-filepath 'path/to/sample.vcf.gz' --result-dir 'path/to/result_dir/'
```

## Run ensemble prediction with your own checkpoints
Pass one `.pthx` per fold; logits are averaged across them. The assay (wgs/wes) is inferred from each checkpoint.
```bash
(muat-env)$ muat predict-ensemble from-checkpoint --ckpt-filepath 'path/fold0.pthx' 'path/fold1.pthx' 'path/fold2.pthx' --hg19 '/path/to/genome_reference/hg19.fa' --input-list 'path/to/inputs.txt' --result-dir 'path/to/result_dir/'
```

## Additional Resources
- **Download PCAWG:** Read [README_download.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_download.md) for details on downloading PCAWG Dataset.
- **Preprocessing:** Read [README_preprocessing.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_preprocessing.md) for details on preprocessing.
- **General Training:** Read [README_MuAtTraining.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_MuAtTraining.md) for general training instructions.
- **Full Training of PCAWG Dataset:** Read [README_PCAWG.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_PCAWG.md) for full training instructions on the PCAWG dataset.
- **Training and Predicting Genomics England Dataset:** Read [README_GEL.md](https://github.com/primasanjaya/muat/blob/main/documentation/README_GEL.md) for complete training and prediction instructions on the Genomics England dataset.
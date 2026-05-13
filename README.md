# Mutation Attention


[![Latest Release](https://img.shields.io/github/v/release/primasanjaya/muat)](https://github.com/primasanjaya/muat/releases)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/muat)](https://anaconda.org/bioconda/muat)
[![License](https://img.shields.io/github/license/primasanjaya/muat)](https://github.com/primasanjaya/muat/blob/main/LICENSE)
[![Conda Downloads](https://img.shields.io/conda/dn/bioconda/muat)](https://anaconda.org/bioconda/muat)
[![Issues](https://img.shields.io/github/issues/primasanjaya/muat)](https://github.com/primasanjaya/muat/issues)
[![Stars](https://img.shields.io/github/stars/primasanjaya/muat?style=social)](https://github.com/primasanjaya/muat)



Conda package for Mutation Attention deep learning tool for tumour type and subtype classification

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

Both `predict` and `predict-ensemble` accept two sources:
- `pretrained {wgs,wes}` — auto-downloads the benchmark checkpoint(s) from HuggingFace.
- `from-checkpoint` — uses your own `.pthx` files; the assay is inferred from each checkpoint.

Input mode is inferred from file suffix:
- Raw inputs (`.vcf{,.gz}`, `.maf{,.gz}`, `.tsv`) are preprocessed first and require `--hg19` or `--hg38`.
- Preprocessed inputs (`.muat.tsv{,.gz}`) are used as-is; the reference flag must be omitted.
- All inputs in a single call must be the same kind (mixed batches are rejected).

## Docker container installation
You can build docker container from source by running `build_docker.sh` <br>
or you can access the prebuild one from [https://biocontainers.pro/tools/muat](https://biocontainers.pro/tools/muat)

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
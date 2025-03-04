# Mutation Attention

[![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/version.svg)](https://anaconda.org/bioconda/muat) [![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/latest_release_date.svg)](https://anaconda.org/bioconda/muat) [![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/platforms.svg)](https://anaconda.org/bioconda/muat) [![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/license.svg)](https://anaconda.org/bioconda/muat) [![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/downloads.svg)](https://anaconda.org/bioconda/muat)

Conda package for Mutation Attention deep learning tool for tumour type and subtype classification

## Quick Start

1. **Clone the muat Repository**
   ```bash
   git clone https://github.com/yourusername/muat.git
   ```

2. **Navigate to the muat Directory**
   ```bash
   cd muat
   ```

3. **Create the Conda Environment**
   To create the conda environment, run:
   ```bash
   conda env create -f muat-env.yml
   ```

4. **Activate the Conda Environment**
   After creating the environment, activate it with:
   ```bash
   conda activate muat-env
   ```

5. **Verify the Installation**
   To test if the installation was successful, run:
   ```bash
   muat -h
   ```

## Download PCAWG Dataset
A Python code example for downloading the PCAWG dataset can be found in `tests/download_pcawg.py`. To download the dataset, execute:
```bash
(muat-env)$ muat download --pcawg
```

## Predict PCAWG Samples (.vcf or .vcf.gz) Using Pretrained Models

### For VCF Files Written with hg19
To predict using VCF files written with hg19, run:
```bash
(muat-env)$ muat predict --wgs --hg19 --mutation-type 'snv' --input-filepath 'sample1_hg19.vcf.gz' 'sample2_hg19.vcf.gz' --result-dir 'path/to/result_dir/'
```

### For VCF Files Written with hg38
To predict using VCF files written with hg38, run:
```bash
(muat-env)$ muat predict --wgs --hg38 --mutation-type 'snv' --input-filepath 'sample_hg38.vcf' --result-dir 'path/to/result_dir/'
```

## Additional Resources
- **Preprocessing:** Read [README_preprocessing.md](README_preprocessing.md) for details on preprocessing.
- **General Training:** Read [README_MuAtTraining.md](README_MuAtTraining.md) for general training instructions.
- **Full Training of PCAWG Dataset:** Read [README_PCAWG.md](README_PCAWG.md) for full training instructions on the PCAWG dataset.
- **Full Training of Genomics England Dataset:** Read [README_GEL.md](README_GEL.md) for full training instructions on the Genomics England dataset.
# Mutation Attention

[![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/version.svg)](https://anaconda.org/bioconda/muat) [![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/latest_release_date.svg)](https://anaconda.org/bioconda/muat) [![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/platforms.svg)](https://anaconda.org/bioconda/muat) [![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/license.svg)](https://anaconda.org/bioconda/muat) [![Anaconda-Server Badge](https://anaconda.org/bioconda/muat/badges/downloads.svg)](https://anaconda.org/bioconda/muat)

Conda package for Mutation Attention deep learning tool for tumour type and subtype classification

## Quick Start

  * Clone muat repository
  * Go to muat repository
  * Create conda environment

```
conda env create -f muat-env.yml
```

## Installation

```bash
conda install bioconda::muat
```

## Download PCAWG dataset
an end-to-end code example is in tests/download_pcawg.py
```python
from muat.download import download_icgc_object_storage as download
import tarfile

files_to_download = ['PCAWG/consensus_snv_indel/README.md',
        'PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public.tgz',
        'PCAWG/consensus_sv/README.md',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.icgc.public.tgz',
        'PCAWG/consensus_sv/final_consensus_sv_bedpe_passonly.tcga.public.tgz',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.v1.4.2016-09-14.tsv',
        'PCAWG/data_releases/latest/release_may2016.v1.4.tsv',
        'PCAWG/data_releases/latest/pcawg_sample_sheet.2016-08-12.tsv',
        'PCAWG/clinical_and_histology/pcawg_specimen_histology_August2016_v9.xlsx']

download_data_path = '/path/to/save/downloaded_data/'
#download data
download(data_path=download_data_path, files_to_download=files_to_download)

tgz_file_path = muat_dir + '/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public.tgz'
# Extract the .tgz file
with tarfile.open(tgz_file_path, 'r:gz') as tar:
    tar.extractall(path=muat_dir + '/data/PCAWG/consensus_snv_indel/final_consensus_snv_indel_passonly_icgc.public/')  # Specify the directory to extract to
    print("Extraction complete.")
```

## Predict PCAWG sample (.vcf or .vcf.gz) using pretrained models
* Prepare tsv file with column 'vcf_hg19_path' containing absolute path of the vcf files which will be predicted by MuAt,
<br>look at the example file in ./example_files/predict_vcf_hg19_example.tsv
* Download the human genome reference which matches the vcf file, e.g. PCAWG used hg19.
```python
from muat.download import download_reference
genome_reference_path = '/dirpath/to/save/genome_reference/' #dirpath to store genome reference
download_reference(genome_reference_path=genome_reference_path)
```
* Choose your pretrained checkpoints, the best checkpoints stored in the package installation path `/pkg_ckpt/pcawg_wgs/`.

```python
from pkg_resources import resource_filename
checkpoint_path = resource_filename('muat','pkg_ckpt/pcawg_wgs/snv+mnv/pcawg-wgs-snv+mnv-MuAtMotifPositionGESF.pthx')
print(checkpoint_path) #use this absolute path for the prediction
```

* Run the inference from terminal
```bash
(muat-env)$  muat --predict-vcf-hg19 --hg19-filepath '/path/to/genome_reference.fa' --load-ckpt-filepath '/path/to/checkpoint.pthx' --vcf-hg19-filepath 'path/to/vcf_hg19.tsv' --result-dir 'path/to/output_result/'
```

## Preprocessing
Read README_preprocessing.md

## General Training
Read README_MuAtTraining.md

## Full Training of PCAWG dataset
Read README_PCAWG.md

## Full Training of Genomics England dataset
Read README_GEL.md
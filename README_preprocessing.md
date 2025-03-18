# Preprocessing

## Assumption
You should have the VCF files downloaded from PCAWG / ICGC or your own data.

## Overview
The preprocessing steps for MuAt input consist of two main stages:

1. **Getting Motif, Genic, Exonic, and Strand Annotation**
   - Output extension: `.gc.genic.exonic.cs.tsv.gz`
   
2. **Tokenizing the Three Information Types for MuAt Input**
   - Output extension: c

## Step-by-Step Instructions

### Step 1: Prepare the TSV File
- Create a TSV file with a column named `vcf_hg19_path` that contains the absolute paths of the VCF files to be predicted by MuAt.
- For an example, refer to the file located at: `./example_files/predict_vcf_hg19_example.tsv`.

### Step 2: Download the Human Genome Reference
- Download the human genome reference that matches your VCF file. For example, PCAWG used hg19.
  
```python
from muat.download import download_reference

genome_reference_path = '/dirpath/to/save/genome_reference/'  # Directory path to store the genome reference
download_reference(genome_reference_path=genome_reference_path)
```

### Step 3: Getting Motif, Genic, Exonic, and Strand Annotations

#### For VCF Files Written with hg19
```bash
(muat-env)$ muat --preprocessing-vcf-hg19 --vcf-hg19-filepath 'path/to/vcf_hg19.tsv' --hg19-filepath '/path/to/genome_reference.fa' --tmp-dir 'path/to/preprocessed_data/'
```

#### For VCF Files Written with hg38
```bash
(muat-env)$ muat --preprocessing-vcf-hg38 --vcf-hg38-filepath 'path/to/vcf_hg38.tsv' --hg38-filepath '/path/to/genome_reference.fa' --tmp-dir 'path/to/preprocessed_data/'
```

### Step 4: Tokenizing Motif, Genomic Position, and Genic Exonic Strand Annotation
- For tokenization, you will need a dictionary of motif, position, and gene vocabularies. You can either create your own or use the default ones stored in:
  - `muat/extfile/dictMutation.tsv` , `muat/extfile/dictChpos.tsv`, `muat/extfile/GES.tsv`
  
```bash
(muat-env)$ muat --tokenizing --tmp-dir 'path/to/preprocessed_data/'
```

## Final Output
After completing all the steps, you will receive the preprocessed and tokenized files in the specified `tmp_dir` with the extension `.token.gc.genic.exonic.cs.tsv.gz`. 

**Note:** These `.token.gc.genic.exonic.cs.tsv.gz` files will be used as input for MuAt, so all training, validation, and test splits will be based on these files.

## Additional Information
To learn how to train MuAt models, please refer to `README_PCAWG.md`.
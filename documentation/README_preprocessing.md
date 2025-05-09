# Preprocessing

## Assumption
You should have the VCF files downloaded from PCAWG / ICGC or your own data.

## Overview
The preprocessing steps for MuAt input consist of two main stages:

1. **Getting Motif, Genic, Exonic, and Strand Annotation**
   - Output extension: `.gc.genic.exonic.cs.tsv.gz`
   
2. **Tokenizing the Three Information Types for MuAt Input**
   - Output extension: `.muat.tsv`

#### For VCF Files Written with hg19
```bash
(muat-env)$ muat preprocessing --vcf --hg19 'path/to/hg19.fa' -input-filepath 'path/to/sample.vcf.gz' 
```

#### For VCF Files Written with hg38
```bash
(muat-env)$ muat preprocessing --vcf --hg38 'path/to/hg38.fa' -input-filepath 'path/to/sample.vcf.gz' 
```

## Final Output
After completing all the steps, you will receive the preprocessed and tokenized files in the specified `tmp_dir` with the extension `.muat.tsv`. 

**Note:** These `.muat.tsv` files will be used as input for MuAt, so all training, validation, and test splits will be based on these files.

## Additional Information
To learn how to train MuAt models, please refer to `README_PCAWG.md`.
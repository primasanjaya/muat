# Predicting Genomics England samples
## Using MuAt2 pretrained model 

All muat2 checkpoints have been stored in in `/re_gecip/machine_learning/muat/muat2_checkpoint/`.
To run MuAt2 prediction, run the command below pointing to the pretrained checkpoint you want to use:
```bash
(muat-env)$ muat predict --wgs --hg38 --mutation-type 'snv' --input-filepath 'sample1_hg19.vcf.gz' 'sample2_hg19.vcf.gz' --result-dir 'path/to/result_dir/' 
```


## Using MuAt pretrained (PCAWG)
for hg19:
```bash
(muat-env)$ muat predict --wgs --hg19 --mutation-type 'snv' --input-filepath 'sample1_hg19.vcf.gz' 'sample2_hg19.vcf.gz' --result-dir 'path/to/result_dir/'
```
or for for hg38 sample:
```bash
(muat-env)$ muat predict --wgs --hg38 --mutation-type 'snv' --input-filepath 'sample_hg38.vcf' --result-dir 'path/to/result_dir/'
```

# Training Genomics England 

Read somagg documentation [here](https://re-docs.genomicsengland.co.uk/somAgg/)

## Preprocessing somAgg
```bash
(muat-env)$ muat preprocessing --somagg --hg38 --input-filepath '/path/to/somagg/chunks.vcf.gz' --hg19-filepath '/path/to/hg19.fa' --hg38-filepath '/path/to/hg38.fa'
```

## Create Training-Val data split
1. **Create tsv files containing prep_path, class_name, subclass_name,class_index,subclass_index**.
The example files are in example_files/train_split_2labels_example.tsv, example_files/val_split_2labels_example.tsv

2. **Run training**
```bash
(muat-env)$ muat 
```

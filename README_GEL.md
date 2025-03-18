# Predicting Genomics England Samples
## Quick VCF hg38 Inference
To run MuAt2 prediction, execute the command below, pointing to the pretrained checkpoint stored in `/re_gecip/machine_learning/muat/muat2_checkpoint/`.

```bash
(muat-env)$ muat predict wgs --hg38 '/re_gecip/machine_learning/muat/genome_ref/ref38_fa' --ckpt-filepath 'path/to/checkpoint.pthx' --input-filepath 'sample1.vcf.gz' --result-dir 'path/to/result_dir/' 
```

## Quick VCF hg19 Inference
```bash
(muat-env)$ muat predict wgs --hg19 '/re_gecip/machine_learning/muat/genome_ref/ref' --ckpt-filepath 'path/to/checkpoint.pthx' --input-filepath 'sample1.vcf.gz' --result-dir 'path/to/result_dir/' 
```

## Quick Inference from Preprocessed Data (.token.gc.genic.exonic.cs.tsv.gz)
```bash
(muat-env)$ muat predict wgs --no-preprocessing --ckpt-filepath 'path/to/checkpoint.pthx' --input-filepath 'sample1.token.gc.genic.exonic.cs.tsv.gz' --result-dir 'path/to/result_dir/' 
```

# Preprocessing somAgg 
The documentation for somAgg can be found [here](https://re-docs.genomicsengland.co.uk/somAgg/).

1. **Run Preprocessing on somAgg Files**<br>
```bash
muat preprocessing --somagg --hg38 'path/to/ref_38.fa' --input-filepath 'path/to/somagg/chunks.vcf.gz' --tmp-dir 'path/to/preprocessed_data/'
```
After successfully running this command, you will receive folders named by the platekey sample. Each folder will contain variants from the somAgg chunk input.<br>
Once you have executed the above command for all chunks, the next step is to combine all chunks to obtain all variants across genomes per platekey (sample).<br>

2. **Combine All Chunk Files Inside the Platekey Folder**<br>
```python
python -c "from muat.preprocessing import combine_somagg_chunks_to_platekey; combine_somagg_chunks_to_platekey(sample_folder='path/to/preprocessed_data/platekey/', tmp_dir='path/to/preprocessed_data/')"
```
This will generate a file named `platekey_sample.token.gc.genic.exonic.cs.tsv.gz`.

# Training MuAt2 Using the MuAt PCAWG-Pretrained Model
To run MuAt2 training, you need to specify which MuAt pretrained checkpoint you want to use. <br>
The MuAt checkpoint is located in `/re_gecip/machine_learning/muat/muat_checkpoint/`.<br>

1. **Prepare Train-Test Split**<br>
The example file for the train-test split is located at `example_files/train_split_2labels_example.tsv`. <br>
You can fill in the tumor type information obtained from LabKey cancer_analysis (i.e 'Disease Type' column) to the example train split file, and 'Disease Sub Type' in the 'subclass_name' column.<br>
For example:

| prep_path                                  | class_name | subclass_name | class_index | subclass_index |
| :----------------------------------------- | :---------:| :------------:| :---------: | --------------:|
| platekey1.token.gc.genic.exonic.cs.tsv.gz  |   BREAST   | DUCTAL        | 1           |  13            |
| platekey2.token.gc.genic.exonic.cs.tsv.gz  |   LUNG     | ADENOCARCINOMA| 2           |  10            |

For MuAt2e models where the prediction heads are combined type/typesubtype you can put it like below:

| prep_path                                  | class_name |    subclass_name    | class_index | subclass_index |
| :----------------------------------------- | :---------:| :------------------:| :---------: | --------------:|
| platekey1.token.gc.genic.exonic.cs.tsv.gz  |   BREAST   | BREAST DUCTAL       | 1           |  76            |
| platekey2.token.gc.genic.exonic.cs.tsv.gz  |   LUNG     | LUNG ADENOCARCINOMA | 2           |  15            |


2. **Run MuAt2 Using the MuAt1 Pretrained Model Finetuned on GEL**<br>
```bash
muat train wgs 
```
# How to train MuAt models

## Assumption
You should have the preprocessed data (.token.gc.genic.exonic.cs.tsv.gz) to train MuAt. Read the preprocessing steps [here](README_preprocessing.md).

## Overview
To train the MuAt model, you need to specify the train/test split in the TSV file. <br>
The example files are in `example_files/{train,val}_split_example.tsv`. <br>

MuAt can be trained with subclasses (cases for MuAt2 with tumor types and subtypes).
To do this, you need to specify the subclass name and subclass index as examples in `example_files/{train,val}_split_2labels_example.tsv`. <br>

You can train MuAt from scratch or from pretrained checkpoints.
## From scratch
```bash
muat train from-scratch --mutation-type 'snv' --use-motif --use-position --use-ges --train-split-filepath 'path/to/example_files/train_split_example.tsv' --val-split-filepath 'path/to/example_files/val_split_example.tsv' --save-dir 'path/to/save/ckpt/' --epoch 100 --learning-rate 0.001 --batch-size 4 --n-layer 1 --n-head 1 --n-emb 128 --mutation-sampling-size 5000
```

## From checkpoint (example for MuAt2 fine-tuning on GEL data)
```bash
muat train from-checkpoint --ckpt-filepath 'path/to/ckpt/model.pthx' --mutation-type 'snv+mnv' --train-split-filepath 'path/to/example_files/train_split_2labels_example.tsv' --val-split-filepath 'path/to/example_files/val_split_example.tsv' --save-dir 'path/to/save/ckpt/' --epoch 5
```


# MBJ-DA

[SIGIR-AP 2023] Python implementation for "Multi-Behavior Job Recommendation with Dynamic Availability"

## Introduction

## Usage

### Requirements

- [pyenv](https://github.com/pyenv/pyenv)
- [Poetry](https://github.com/python-poetry/poetry)
- You need to install python (>=3.8 and <3.11) via pyenv in advance.

### Setup

```sh
$ poetry env use 3.8.10 # please specify your python version
$ poetry install
```

### Training

```sh
$ poetry run python -m MBJ-DA.train
```

You can see the usage by the following command.

```sh
$ poetry run python -m MBJ-DA.train -h
usage: train.py [-h] [--seed SEED] [--dataset [DATASET]] [--data_path [DATA_PATH]] [--dim DIM] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--da_size DA_SIZE]
                [--neg_size NEG_SIZE] [--lr LR] [--patience PATIENCE] [--Ks [KS]] [--model_path [MODEL_PATH]]

Run MBJ-DA.

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random seed.
  --dataset [DATASET]   Choose a dataset from {toy}.
  --data_path [DATA_PATH]
                        Input data path.
  --dim DIM             Number of dimension.
  --epoch EPOCH         Number of epoch.
  --batch_size BATCH_SIZE
                        Batch size.
  --da_size DA_SIZE     Dynamic availabiliry size.
  --neg_size NEG_SIZE   Negative sampling size.
  --lr LR               Learning rate.
  --patience PATIENCE   Number of epoch for early stopping.
  --Ks [KS]             Calculate metric@K when evaluating.
  --model_path [MODEL_PATH]
                        Model path for evaluation.
```

### Evaluation

```sh
$ poetry run python -m MBJ-DA.test --model_path trained_model/toy_lr0.005_dim32/best.pth # please specify your model path
```

## Dataset

Due to privacy and business restrictions, we cannot release our dataset right now.
Instead of our dataset, there is a toy dataset for checking our code functionality.

You can adapt our code for your own dataset with the following dataset format.

### Dataset format

To use our code, the following two types of data are required.

TODO

## Citation

If you make use of this code or our algorithm, please cite the following paper.
After our paper is published officially, we'll replace the following citation as official one.

```txt
@inproceedings{saito2023,
	author={Saito, Yosuke and Sugiyama, Kazunari},
	booktitle={Proceedings of the 1st ACM SIGIR-AP Conference on Research and Development in Information Retrieval},
	title={Multi-Behavior Job Recommendation with Dynamic Availability},
	year={2023}
}
```

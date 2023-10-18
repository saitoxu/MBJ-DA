# MBJ-DA

[SIGIR-AP 2023] Python implementation for "Multi-Behavior Job Recommendation with Dynamic Availability"

## Introduction

In recent years, we can see a lot of job postings on the Internet, providing us with more diverse job opportunities.
As a result, it is getting more and more difficult for job seekers to find job postings relevant to their preferences.
Consequently, job recommendations play an important role to reduce the burden of job searching.
Generally, job postings have a publication period, for example, 30 days.
Then they have been expired since the positions were occupied.
As a result, job seekers may be frustrated when they experience such situations as they cannot apply for the positions.
This indicates that job seekers may have strong preferences for job postings even if their application behaviors cannot be observed.
This kind of gap has not been investigated in the line of Multi-Behavior Recommendation.
Therefore, in this work, we propose a new job recommendation model, called Multi-Behavior Job Recommendation with Dynamic Availability (MBJ-DA), which takes into account: (1) auxiliary behaviors other than an application behavior and (2) the influence of dynamic availability of job postings.
MBJ-DA enables a more accurate estimation of each user’s actual preferences by explicitly distinguishing the noise potentially inherent in auxiliary behaviors.
Furthermore, by explicitly considering the influence of the dynamic availability of job postings, MBJ-DA can mitigate biases resulting from the influence and estimate each user’s actual preferences more accurately.
Experimental results on our dataset constructed from an actual job search website show that MBJ-DA outperforms several state-of-the-arts in terms of MRR and nDCG.

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

To use our code, you need the following four types of data.

#### 1. jobs.txt

This is data for the job posting start time and end time.
The format is below.
`<start_ts>` and `<end_ts>` must be integers.

```txt
<job_id> <start_ts> <end_ts>
```

#### 2. train.txt

This is interaction data for each user.
Each interaction is represented in the format `<job_id>:<behavior_id>:<interaction_ts>`.
`<behavior_id>` must be an integer.

```txt
<user_id> <job_id>:<behavior_id>:<interaction_ts> ...
```

#### 3. val.txt

This is the interaction data used for validation.
The format is the same as `train.txt`, but the interaction data only includes a single entry related to the target behavior.

#### 4. test.txt

This is the data used for evaluation.
It has the same format as val.txt.

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

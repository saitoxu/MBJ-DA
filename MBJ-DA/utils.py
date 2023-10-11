import os
from typing import List
import random
import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.best_value = .0
        self.count = 0
        self.patience = patience

    def __call__(self, value: float):
        should_save, should_stop = False, False
        if value > self.best_value:
            self.best_value = value
            self.count = 0
            should_save = True
        else:
            self.count += 1
        if self.count >= self.patience:
            should_stop = True
        return should_save, should_stop


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class NegativeSampler():
    def __init__(self, path: str, job_size: int) -> None:
        self.observed = {}
        self.all_jobs = set(range(job_size))
        datasets = ['train.txt', 'val.txt', 'test.txt']
        for dataset in datasets:
            with open(f'{path}/{dataset}') as f:
                for line in f:
                    user_id, *interactions = line.split(' ')
                    user_id = int(user_id)
                    for interaction in interactions:
                        job_id, _, _ = interaction.split(':')
                        job_id = int(job_id)
                        if user_id not in self.observed:
                            self.observed[user_id] = set()
                        self.observed[user_id].add(job_id)

    def sample(self, user_id: int, size: int) -> List[int]:
        observed = self.observed[user_id]
        result = random.sample(list(self.all_jobs - observed), size)
        return result

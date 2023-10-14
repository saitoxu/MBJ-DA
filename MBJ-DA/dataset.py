from torch.utils.data import Dataset
import pandas as pd
from typing import List


class BaseTrainDataset(Dataset):
    def _calc_stats(self, data_path: str) -> int:
        user_size = 0
        job_size = 0
        behavior_size = 0
        with open(f'{data_path}/jobs.txt') as f:
            for line in f:
                job_id = int(line.split(' ')[0])
                job_size = max(job_size, job_id + 1)
        with open(f'{data_path}/train.txt') as f:
            for line in f:
                user_id, *interactions = line.split(' ')
                user_size = max(user_size, int(user_id) + 1)
                for interaction in interactions:
                    job_id, behavior_id, _ = interaction.split(':')
                    job_id = int(job_id)
                    behavior_id = int(behavior_id)
                    behavior_size = max(behavior_size, int(behavior_id) + 1)
        return user_size, job_size, behavior_size

    def _load_train_data(self, data_path: str, user_size: int, job_size: int, behavior_size: int):
        auxiliary_data, target_data = [], []

        job_availability = []
        with open(f'{data_path}/jobs.txt') as f:
            for line in f:
                _, start, end = map(lambda x: float(x), line.split(' '))
                job_availability.append([int(start), int(end)])
        job_availability = pd.DataFrame(job_availability, columns=['start', 'end'])

        auxiliary_mtxs = []
        for _ in range(behavior_size-1):
            auxiliary_mtx = [[0 for _ in range(job_size)] for _ in range(user_size)]
            auxiliary_mtxs.append(auxiliary_mtx)
        target_mtx = [[0 for _ in range(job_size)] for _ in range(user_size)]
        da_mtx = [[0 for _ in range(job_size)] for _ in range(user_size)]
        target_behavior_id = behavior_size-1
        with open(f'{data_path}/train.txt') as f:
            for line in f:
                user_id, *raw_interactions = list(line.split(' '))
                user_id = int(user_id)
                user_last_ts = -1
                interactions = []
                for interaction in raw_interactions:
                    job_id, behavior_id, ts = interaction.split(':')
                    job_id = int(job_id)
                    behavior_id = int(behavior_id)
                    ts = int(ts)
                    interactions.append((job_id, behavior_id, ts))
                    user_last_ts = max(user_last_ts, ts)
                    if behavior_id == target_behavior_id:
                        target_mtx[user_id][job_id] = 1
                    else:
                        auxiliary_mtxs[behavior_id][user_id][job_id] = 1
                for interaction in interactions:
                    job_id, behavior_id, _ = interaction
                    if behavior_id == target_behavior_id:
                        continue
                    if job_availability['end'][job_id] < user_last_ts:
                        da_mtx[user_id][job_id] = 1
        auxiliary_data = []
        for i, mtx in enumerate(auxiliary_mtxs):
            auxiliary_data += self._convert_auxiliary_mtx_to_list(mtx, i)
        target_data = self._convert_target_mtx_to_list(target_mtx, da_mtx)

        return auxiliary_data, target_data

    def _convert_auxiliary_mtx_to_list(self, mtx: List[List[int]], behavior_id: int):
        data = []
        for i, row in enumerate(mtx):
            for j, cell in enumerate(row):
                if cell == 1:
                    data.append([i, j, behavior_id])
        return data

    def _convert_target_mtx_to_list(self, mtx: List[List[int]], da_mtx: List[List[int]]):
        data = []
        for i, row in enumerate(mtx):
            total = sum(da_mtx[i])
            for j, cell in enumerate(row):
                if cell == 1:
                    data.append([i, j, total])
        return data

    def _load_test_data(self, data_path: str, file: str) -> None:
        data = []
        with open(f'{data_path}/{file}') as f:
            for line in f:
                user_id, interaction = list(line.split(' '))
                user_id = int(user_id)
                job_id, _, _ = interaction.split(':')
                job_id = int(job_id)
                data.append([user_id, job_id])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AuxiliaryTrainDataset(BaseTrainDataset):
    def __init__(self, data_path: str):
        self.user_size, self.job_size, self.behavior_size = self._calc_stats(data_path)
        self.data, _ = self._load_train_data(data_path, self.user_size, self.job_size, self.behavior_size)


class TargetTrainDataset(BaseTrainDataset):
    def __init__(self, data_path: str):
        self.user_size, self.job_size, self.behavior_size = self._calc_stats(data_path)
        _, self.data = self._load_train_data(data_path, self.user_size, self.job_size, self.behavior_size)


class ValDataset(BaseTrainDataset):
    def __init__(self, data_path: str):
        self.user_size, self.job_size, self.behavior_size = self._calc_stats(data_path)
        self.data = self._load_test_data(data_path, 'val.txt')


class TestDataset(BaseTrainDataset):
    def __init__(self, data_path: str):
        self.user_size, self.job_size, self.behavior_size = self._calc_stats(data_path)
        self.data = self._load_test_data(data_path, 'test.txt')

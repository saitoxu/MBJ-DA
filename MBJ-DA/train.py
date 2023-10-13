import os
import torch
from torch.utils.data import DataLoader
from time import time
from functools import reduce

from .parser import parse_args
from .dataset import AuxiliaryTrainDataset, TargetTrainDataset, ValDataset
from .utils import EarlyStopping, seed_everything, NegativeSampler
from .metrics import evaluate
from .logger import getLogger
from .model import Model


def train(auxiliary_dataloader, target_dataloader, model, optimizer, sampler, neg_size, device, logger):
    model.train()
    dataloaders = [auxiliary_dataloader, target_dataloader]

    size = sum(list(map(lambda x: len(x.dataset), dataloaders)))
    iterators = list(map(lambda x: iter(x), dataloaders))
    losses = [[] for _ in range(len(iterators))]
    batch = 0
    current = 0
    batch_continue = True

    while batch_continue:
        continue_flags = [True for _ in range(len(iterators))]
        for i, iterator in enumerate(iterators):
            try:
                behavior_ids = None
                das = None
                if i == 0:
                    users, pos_jobs, behavior_ids = next(iterator)
                    users, pos_jobs, behavior_ids = users.to(device), pos_jobs.to(device), behavior_ids.to(device)
                else:
                    users, pos_jobs, das = next(iterator)
                    users, pos_jobs, das = users.to(device), pos_jobs.to(device), das.to(device)

                neg_job_id_lists = []
                for u in users:
                    neg_job_id_lists.append(sampler.sample(int(u), neg_size))
                neg_job_id_lists = torch.tensor(neg_job_id_lists).to(device)

                loss = model(i, users, pos_jobs, behavior_ids, das, neg_job_id_lists)
                losses[i] = [loss.item()] + losses[i]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current += len(users)
            except StopIteration:
                continue_flags[i] = False

        if batch % 50 == 0:
            cnt = min(list(map(lambda x: len(x), losses)))
            _loss = list(reduce(lambda x, y: x + y, list(map(lambda x: x[:cnt], losses))))
            _loss = sum(_loss) / (cnt*len(losses))
            logger.debug(f"Loss: {_loss:>12f} [{current:>8d}/{size:>8d}]")
        batch_continue = any(continue_flags)
        batch += 1


if __name__ == '__main__':
    start = int(time())
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    seed_everything(args.seed)

    logger = getLogger(name=__name__, path=args.save_path)

    for key, value in vars(args).items():
        logger.debug(f'{key}: {value}')

    torch.backends.cudnn.benchmark = True

    data_path = f"{args.data_path}/{args.dataset}"
    num_workers = 2 if os.cpu_count() > 1 else 0

    auxiliary_train_data = AuxiliaryTrainDataset(data_path=data_path)
    auxiliary_train_dataloader = DataLoader(auxiliary_train_data, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    target_train_data = TargetTrainDataset(data_path=data_path)
    target_train_dataloader = DataLoader(target_train_data, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)

    val_data = ValDataset(data_path=data_path)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    user_size = auxiliary_train_data.user_size
    job_size = auxiliary_train_data.job_size
    behavior_size = auxiliary_train_data.behavior_size
    weight_decay = 0.0001
    model = Model(
        user_size=user_size,
        job_size=job_size,
        behavior_size=behavior_size,
        dim=args.dim,
        da_size=args.da_size,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    early_stop = EarlyStopping(args.patience)
    negative_sampler = NegativeSampler(data_path, job_size)

    Ks = eval(args.Ks)
    epoch = args.epoch
    test_interval = 5
    for t in range(args.epoch):
        logger.debug(f"Epoch {t+1}")
        logger.debug('-'*32)
        train(auxiliary_train_dataloader, target_train_dataloader, model, optimizer, negative_sampler, args.neg_size, device, logger)
        torch.save(model, args.save_path + 'latest.pth')
        if (t+1) % test_interval == 0:
            mrr, _, _ = evaluate(val_dataloader, model, Ks, negative_sampler, device, logger.debug)
            # early stopping
            should_save, should_stop = early_stop(mrr)
            if should_save:
                torch.save(model, args.save_path + 'best.pth')
            if should_stop:
                epoch = t + 1
                logger.debug('Early stopping.')
                break
    end = int(time())
    logger.debug('Done!')

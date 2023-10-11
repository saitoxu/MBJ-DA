import torch
import numpy as np


def evaluate(dataloader, model, Ks, sampler, device, log=print):
    model.eval()
    preds = []
    with torch.no_grad():
        for users, pos_jobs in dataloader:
            for idx in range(len(users)):
                user = int(users[idx])
                correct_job = int(pos_jobs[idx])
                unobserved_jobs = sampler.sample(user, 99)
                candidate_jobs = torch.tensor([correct_job] + unobserved_jobs).to(device)
                pred = model.predict(user=user, candidates=candidate_jobs)
                preds.append((user, pred.cpu()))
    mrr, hrs, ndcgs = _calc_metrics(preds, Ks)
    _display_metrics(mrr, hrs, ndcgs, log)
    return mrr, hrs, ndcgs


def _calc_metrics(preds, Ks):
    ranks = []
    for _, pred in preds:
        _, indices = torch.sort(pred, descending=True)
        rank = indices.argmin() + 1
        ranks.append(rank.item())
    return _mrr(ranks), _hit_ratios(ranks, Ks), _ndcgs(ranks, Ks)


def _mrr(ranks):
    return np.array(list(map(lambda x: 1 / x, ranks))).sum() / len(ranks)


def _hit_ratios(ranks, Ks):
    results = []
    for k in Ks:
        hr = len(list(filter(lambda x: x <= k, ranks))) / len(ranks)
        results.append(hr)
    return results


def _ndcgs(ranks, Ks):
    results = []
    for k in Ks:
        ndcg = np.array(list(map(lambda x: 1 / np.log2(x + 1), list(filter(lambda x: x <= k, ranks))))).sum() / len(ranks)
        results.append(ndcg)
    return results


def _display_metrics(mrr, hrs, ndcgs, log):
    rounded_hrs = list(map(lambda x: float(f'{x:>7f}'), hrs))
    rounded_ndcgs = list(map(lambda x: float(f'{x:>7f}'), ndcgs))
    log(f'MRR:\t{mrr:>7f}')
    log(f'HRs:\t{rounded_hrs}')
    log(f'nDCGs:\t{rounded_ndcgs}')

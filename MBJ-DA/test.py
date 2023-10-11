import torch
from torch.utils.data import DataLoader

from .parser import parse_args
from .dataset import TestDataset
from .utils import seed_everything, NegativeSampler
from .metrics import evaluate

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    args = parse_args()
    seed_everything(args.seed)

    data_path = f'{args.data_path}/{args.dataset}'

    test_data = TestDataset(data_path=data_path)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    negative_sampler = NegativeSampler(data_path, test_data.job_size)
    model = torch.load(args.model_path)
    Ks = eval(args.Ks)
    evaluate(test_dataloader, model, Ks, negative_sampler, device)

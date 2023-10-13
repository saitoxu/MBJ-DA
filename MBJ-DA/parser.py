import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run MBJ-DA.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--dataset', nargs='?', default='toy', help='Choose a dataset from {toy}.')
    parser.add_argument('--data_path', nargs='?', default='datasets', help='Input data path.')
    parser.add_argument('--dim', type=int, default=32, help='Number of dimension.')
    parser.add_argument('--epoch', type=int, default=300, help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--da_size', type=int, default=10, help='Dynamic availabiliry size.')
    parser.add_argument('--neg_size', type=int, default=2, help='Negative sampling size.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=5, help='Number of epoch for early stopping.')
    parser.add_argument('--Ks', nargs='?', default='[5,10,20]', help='Calculate metric@K when evaluating.')
    parser.add_argument('--model_path', nargs='?', default='', help='Model path for evaluation.')

    args = parser.parse_args()

    save_dir = f'{args.dataset}_lr{args.lr}_dim{args.dim}'
    save_path = f'trained_model/{save_dir}/'
    args.save_path = save_path

    return args

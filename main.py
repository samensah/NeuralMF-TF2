import numpy as np
import argparse
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from solver import Solver as Solver


def main(args):
    seed = args.seed
    np.random.seed(seed)
    np.set_printoptions(precision=4)

    print()
    print('[ARGUMENTS]')
    print(args)
    print()
    
    net = Solver(vars(args))

    if args.mode == 'train' : net.train()
    elif args.mode == 'test' : net.test(save_ckpt=True)
    else : return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RS experiment')
    parser.add_argument('--epoch', default = 100, type=int, help='epoch size')
    parser.add_argument('--log_file', default = 'log.txt', type=str, help='file to save loss info')
    parser.add_argument('--seed', default = 1, type=int, help='random seed')
    parser.add_argument('--mode',default='train', type=str, help='train or eval')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory path')
    parser.add_argument('--load_ckpt',default='', type=str, help='checkpoint name')

    # filmTrust data
    parser.add_argument('--lr', default = 1e-4, type=float, help='learning rate')
    parser.add_argument('--emb_dim', type=int, default=40, help='embedding size of item and user.')
    parser.add_argument('--batch_size', default = 24, type=int, help='batch size')
    parser.add_argument('--dataset', default='filmTrust_data', type=str, help='dataset name')
    parser.add_argument('--K', default = 100, type=int, help='dimension of hidden layer')
    parser.add_argument('--mlp_layers', default=1, type=int, help='number of mlp layer')

    args = parser.parse_args()
    main(args)

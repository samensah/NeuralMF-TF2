import h5py
import scipy.io as io
import numpy as np


def chunk_into_batches(data, batch_size):
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

def return_data(opt):
    train_data = io.loadmat(opt['dataset']+'/train.mat')['train']
    test_data = io.loadmat(opt['dataset']+'/test.mat')['test']

    train_num = len(train_data)
    test_num = len(test_data)
    print('size of train and test: {}, {}'.format(len(train_data), len(test_data)))
    user_num = len(set(train_data[:,0]) | set(test_data[:,0]))
    item_num = len(set(train_data[:,1]) | set(test_data[:,1]))
    label_num = len(set(train_data[:,2]) | set(test_data[:,2]))
    print('user num, item num and label num: {}, {}'.format(user_num, item_num, label_num))

    # user_id, item_id, rating
    # chunk into batches
    train_data, test_data = chunk_into_batches(train_data, opt['batch_size']), chunk_into_batches(test_data, opt['batch_size'])

    return train_data, test_data, user_num, item_num, train_num, test_num


if __name__ == "__main__":
    opt = dict()
    opt['dataset'] = 'filmTrust_data'
    opt['batch_size'] = 100
    return_data(opt)

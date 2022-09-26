#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader

import pl_models
from data.dataset import get_vocab_dataset

ABS_DIR = os.path.split(os.path.abspath(__file__))[0]
OUT_DIR = os.path.join(ABS_DIR, 'enc_data')
NP_FILE = os.path.join(OUT_DIR, 'enc_vectors.npy')
MEAN_VAR = os.path.join(OUT_DIR, 'enc_mean_var.npy')
MAX_MIN = os.path.join(OUT_DIR, 'enc_max_min.npy')
FIGURE = os.path.join(OUT_DIR, 'enc_visual.png')


def parse():
    parser = argparse.ArgumentParser(description='generate encode space')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path of input data')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='path of vocabulary')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='path of checkpoint of model')
    parser.add_argument('--model', type=str, choices=['vae_dgmg', 'dgmg'],
                        required=True, help='type of model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of mini-batch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of cpus to use')
    parser.add_argument('--save_vectors', action='store_true',
                        help='store all the vectors')
    parser.add_argument('--output_dir', type=str, default=OUT_DIR,
                        help=f'path to store output files, default {OUT_DIR}')
    return parser.parse_args()


def draw2D(np_array):
    '''use pca to downgrade the vectors to 2d,
       np_array: [num_samples, encode_dim]'''
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(np_array)

    fake = np.random.rand(10, np_array.shape[-1])
    fake = pca.transform(fake)

    real_color = 'blue'
    fake_color = 'grey'
    fig = plt.figure()
    ax = plt.subplot()
    ax.scatter(new_data[:, 0], new_data[:, 1], c=real_color, alpha=0.6)
    ax.scatter(fake[:, 0], fake[:, 1], c=fake_color, alpha=0.6)
    plt.savefig(FIGURE, dpi=200)
    print(f'2D illustration of encoding space is saved to {FIGURE}.\n'
          f'Tip: {real_color} marks real vectors and {fake_color} marks random vectors.')


def mean_var(np_array):
    '''calculate the mean and standard variance of each dimension
       np_array: [num_samples, feature_dim]'''
    mean = np.mean(np_array, axis=0)
    var = np.std(np_array, axis=0)
    mean_var = np.stack([mean, var], axis=1)
    np.save(MEAN_VAR, mean_var)
    print('Mean and standard variance of each dimension have been '
          f'saved to {MEAN_VAR} with array shape {mean_var.shape}')


def max_min(np_array):
    '''calculate the max and min of each dimension
       np_array: [num_samples, feature_dim]'''
    _max = np.max(np_array, axis=0)
    _min = np.min(np_array, axis=0)
    max_min = np.stack([_max, _min], axis=1)
    np.save(MAX_MIN, max_min)
    print('Max and min of each dimension have been '
          f'saved to {MAX_MIN} with array shape {max_min.shape}')


MEAN, VAR = None, None
def gauss_sample(n=1):
    '''get n samples from encoding space using gaussian distribution'''
    assert n > 0
    global MEAN, VAR
    if MEAN is None or VAR is None:
        stat = np.load(MEAN_VAR)
        MEAN, VAR = stat[:, 0], stat[:, 1]
    epsilon = np.random.randn(n, MEAN.shape[0])
    return VAR * epsilon + MEAN


MAX, MIN = None, None
def uni_sample(n=1):
    '''get n samples from encoding space using uniform distribution'''
    assert n > 0
    global MAX, MIN
    if MAX is None or MIN is None:
        stat = np.load(MAX_MIN)
        MAX, MIN = stat[:, 0], stat[:, 1]
    uni = np.random.uniform(low=0.25, high=0.75, size=(n, MAX.shape[0]))
    return uni * (MAX - MIN) + MIN 


def sample(n=1, method='uniform'):
    sample_funcs = {
        'uniform': uni_sample,
        'gaussian': gauss_sample
    }
    return sample_funcs[method](n)


def main(args):
    vocab, train_set = get_vocab_dataset(args.vocab_path, args.data_path)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers)
    model = str2model(args.model).load_from_checkpoint(args.ckpt)
    model.eval()
    res = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            res.append(model.get_z(batch))
        res = torch.cat(res, dim=0).detach().cpu().numpy()
    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_vectors:
        np.save(NP_FILE, res)
        print(f'All encoded vectors are saved to {NP_FILE}')
    draw2D(res)
    mean_var(res)
    max_min(res)


def str2model(s):
    if s == 'vae_dgmg':
        return pl_models.vae_dgmg_model.VAEDGMGModel
    elif s == 'dgmg':
        return pl_models.dgmg_model.DGMGModel
    else:
        raise NotImplementedError(f'{s} model not implemented!')


if __name__ == '__main__':
    main(parse())
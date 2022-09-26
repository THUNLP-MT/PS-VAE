#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import argparse
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')
from evaluation.utils import get_normalized_property_scores, restore_property_scores, PROPS
from utils.chem_utils import smiles2molecule
from utils.logger import print_log
sys.path.remove('..')


ABS_DIR = os.path.split(os.path.abspath(__file__))[0]
DEFAULT_OUTPUT_DIR = os.path.join(ABS_DIR, 'prop_figs')


def parse():
    parser = argparse.ArgumentParser(description='Statistic tool for properties')
    parser.add_argument('--data', type=str, required=True,
                        help='Path of data')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Path to save statistic figures and log')
    parser.add_argument('--cpus', type=int, default=os.cpu_count(),
                        help='Number of cpu cores to parallel calculation')
    return parser.parse_args()


def draw_hist(data, title, xlabel, ylabel='Frequency', png=None):
    plt.clf()
    plt.hist(data, bins=40, density=False,
             facecolor="blue", edgecolor="black", alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if png is None:
        png = title + '.png'
    plt.savefig(png)


def get_props(smiles, addition):
    all_props = []
    for s in tqdm(smiles):
        s = s.strip('\n')
        mol = smiles2molecule(s)
        props = get_normalized_property_scores(mol)
        props = restore_property_scores(props)
        add_prop = [func(mol) for func in addition]
        all_props.append(props + add_prop)
    return all_props


def anum(mol):
    return mol.GetNumAtoms()


def main(args):
    '''data should consists of one smiles per line'''
    with open(args.data, 'r') as fin:
        smiles = fin.readlines()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    addition_props = [('atom number', anum)]
    all_props = []
    pool = mp.Pool(args.cpus)
    size = int(len(smiles) / args.cpus)
    progs = [pool.apply_async(get_props,
                             args=(smiles[i:min(len(smiles), i+size)],
                                   [func for _, func in addition_props])) \
             for i in range(0, len(smiles), size)]
    for p in progs:
        all_props.extend(p.get())
    all_props = np.array(all_props)
    logs = ''
    for i, pname in enumerate(PROPS + [pn for pn, _ in addition_props]):
        p = all_props[:, i]
        _mean = round(np.mean(p), 2)
        _max = round(np.max(p), 2)
        _min = round(np.min(p), 2)
        logs += f'{pname}: mean {_mean} max {_max} min {_min}\n'
        draw_hist(p, pname, pname, png=os.path.join(args.output_dir, pname + '.png'))
    print(logs)
    with open(os.path.join(args.output_dir, 'log'), 'w') as fout:
        fout.writelines(logs)


if __name__ == '__main__':
    main(parse())
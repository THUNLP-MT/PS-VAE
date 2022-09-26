#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import argparse

import numpy as np

from logger import print_log

def parse():
    parser = argparse.ArgumentParser(description='split train/dev/test')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to origin dataset')
    parser.add_argument('--valid_ratio', type=float, required=True,
                        help='Ratio of validation set')
    parser.add_argument('--test_ratio', type=float, required=True,
                        help='Ratio of test set')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=6)
    return parser.parse_args()


def main(args):
    suffix = args.data.split('.')[-1]
    with open(args.data, 'r') as fin:
        lines = fin.readlines()
    valid_ratio, test_ratio = args.valid_ratio, args.test_ratio
    train_ratio = 1 - valid_ratio - test_ratio
    valid_num = int(valid_ratio * len(lines))
    test_num = int(test_ratio * len(lines))
    train_num = len(lines) - valid_num - test_num
    split_pos = [0, train_num, train_num + valid_num, len(lines)]
    np.random.seed(args.seed)
    np.random.shuffle(lines)
    for i, name in enumerate(['train', 'valid', 'test']):
        out_dir = os.path.join(args.output_dir, name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, name + '.' + suffix)
        with open(out_path, 'w') as fout:
            data = lines[split_pos[i]:split_pos[i+1]]
            print_log(f'{name} set: {len(data)}')
            fout.writelines(data)


if __name__ == '__main__':
    main(parse())
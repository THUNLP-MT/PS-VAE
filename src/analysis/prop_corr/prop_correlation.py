#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
from copy import copy
from tqdm import tqdm
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
np.random.seed(2021)

from data.mol_bpe import Tokenizer
from evaluation.utils import eval_funcs_dict
from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='property and substructure correlation')
    parser.add_argument('--normal_smi', type=str, required=True, help='Path to normal smiles')
    parser.add_argument('--zinc_smi', type=str, required=True, help='Path to zinc smiles')
    parser.add_argument('--prop_smi', type=str, required=True, help='Path to property smiles')
    parser.add_argument('--prop', type=str, choices=['qed', 'sa', 'logp'], required=True)
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('--out', type=str, required=True, help='Save figure')
    return parser.parse_args()


def turn(smi):
    return Chem.MolFromSmiles(smi)

def cal_piece_dist(smis, tokenizer, pool):
    patts = pool.map(turn, tokenizer.idx2piece)
    mols = pool.map(turn, smis)
    dist = defaultdict(int)
    for m in mols:
        for p, s in zip(patts, tokenizer.idx2piece):
            if p is None:  # <pad> and <s>
                continue
            if m.HasSubstructMatch(p):
                dist[s] += 1
    return dist


def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)
    

def main(args):
    tokenizer = Tokenizer(args.vocab)
    score_func = eval_funcs_dict()[args.prop]
    print_log('Tokenizing molecules')
    dataset = {'without target': args.normal_smi,
               'with target': args.prop_smi,
               'zinc': args.zinc_smi
            }
    props = {}
    with mp.Pool(8) as pool:
        for d in dataset:
            print_log(f'{d} molecules from {dataset[d]} tokenized')
            with open(dataset[d], 'r') as fin:
                dataset[d] = fin.read().strip().split('\n')
            dataset[d] = pool.map(turn, dataset[d])
            props[d] = pool.map(score_func, dataset[d])
        patts = pool.map(turn, tokenizer.idx2piece)
    
    table2 = {
        'piece': [],
        'corr': []
    }
    piece_dist = { key: defaultdict(int) for key in dataset}
    prop_array = np.concatenate([props[d] for d in dataset])
    selected_pieces = {}
    for p, s in tqdm(zip(patts, tokenizer.idx2piece), total=len(patts)):
        if p is None or p.GetNumAtoms() < 5:
            continue
        x, piece_num_atoms = [], p.GetNumAtoms()
        for _type in dataset:
            mol_set = dataset[_type]
            for mol in mol_set:
                if mol.HasSubstructMatch(p):
                    x.append(len(mol.GetSubstructMatches(p)))# * mol.GetNumAtoms() / piece_num_atoms)
                    selected_pieces[s] = True
                    piece_dist[_type][s] += x[-1]
                else:
                    x.append(0)

        if s in selected_pieces:
            corr = np.corrcoef(x, prop_array)[0][1]
            table2['piece'].append(s)
            table2['corr'].append(corr)

    # diffs = []
    # pieces = table2['piece']
    # for p in pieces:
    #     low, high = piece_dist['without target'][p], piece_dist['with target'][p]
    #     sig = 1
    #     if low > high:
    #         sig = -1
    #         low, high = high, low
    #     diffs.append(sig * high / (low + 1e-4))
    # order = sorted([i for i in range(len(pieces))], key=lambda i: diffs[i])
    order = sorted([i for i in range(len(table2['corr']))], key=lambda i: table2['corr'][i])
    table2['piece'] = [table2['piece'][i] for i in order]
    table2['corr'] = [table2['corr'][i] for i in order]

    # filtering and sampling
    selected_idx = []
    for i, p in enumerate(table2['piece']):
        ref_dataset = 'zinc'
        ref = piece_dist[ref_dataset][p] + 1
        skip = True
        for t in piece_dist:
            if t == ref_dataset:
                continue
            skip = skip and (piece_dist[t][p] == 0)
        if skip or piece_dist['without target'][p] / ref > 2: # both zero or outliers
            continue
        selected_idx.append(i)
    selected_idx = sorted(np.random.choice(selected_idx, size=100, replace=False))

    new_table = {
        'piece': [],
        'freq': [],
        'type': []
    }
    for i in selected_idx:
        p = table2['piece'][i]
        ref_dataset = 'zinc'
        ref = piece_dist[ref_dataset][p] + 1
        for t in piece_dist:
            if t == ref_dataset:
                continue
            new_table['piece'].append(p)
            new_table['freq'].append(piece_dist[t][p] / ref)
            new_table['type'].append(t)

    table2['piece'] = [table2['piece'][i] for i in selected_idx]
    table2['corr'] = [table2['corr'][i] for i in selected_idx]

    sns.set(font_scale = 4)
    fig, ax1 = plt.subplots(figsize=(80, 30))
    sns.lineplot(ax=ax1, data=table2, x='piece', y='corr', sort=False)
    plt.xticks(rotation=90, fontsize=20)
    ax1.set_xlabel('Graph Piece')
    ax1.set_ylabel('Pearson Correlation')
    ax2 = ax1.twinx()
    # sns.histplot(ax=ax2, data=table, x='piece', hue='type', stat='probability')
    sns.histplot(ax=ax2, data=new_table, x='piece', weights='freq', hue='type', stat='probability')
    move_legend(ax2, 'upper left')
    ax2.set_ylabel('Frequency')
    fig.tight_layout()
    fig.savefig(args.out)

    
if __name__ == '__main__':
    main(parse())
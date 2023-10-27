#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from tqdm import tqdm
from random import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from data.mol_bpe import Tokenizer
from utils import chem_utils
from utils.nn_utils import to_one_hot
import evaluation


class BPEMolDataset(Dataset):
    def __init__(self, fname, tokenizer):
        super(BPEMolDataset, self).__init__()
        self.root_path, self.file_path = os.path.split(fname)
        self.save_path = os.path.join(self.root_path, 'processed_data.pkl')
        self.tokenizer = tokenizer
        try:
            self.data = torch.load(self.save_path)
        except FileNotFoundError:
            self.data = self.process()
    
    @staticmethod
    def process_step1(mol, tokenizer):
        '''to raw representation (save storage space)'''
        # add nodes
        x = [tokenizer.chem_vocab.atom_to_idx(mol.GetAtomWithIdx(i).GetSymbol())
             for i in range(mol.GetNumAtoms())] # [num_nodes]
        # edge index and edge attr
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            attr = tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())
            edge_index.append([begin, end])  # do not repeat for storage
            edge_attr.append(attr)
        # add property scores
        properties = evaluation.utils.get_normalized_property_scores(mol)
        # piece breakdown
        pieces, groups = tokenizer(mol, return_idx=True)
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'props': properties,
            'pieces': pieces,
            'groups': groups
        }

    @staticmethod
    def process_step2(data, tokenizer):
        '''to adjacent matrix representation'''
        x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
        ad_mat = [[tokenizer.chem_vocab.bond_to_idx(None) for _ in x] for _ in x]
        for i in range(len(edge_attr)):
            begin, end = edge_index[i]
            ad_mat[begin][end] = ad_mat[end][begin] = edge_attr[i]
        x_pieces, x_pos, pieces = [0 for _ in x], [0 for _ in x], data['pieces']
        edge_select = [[1 for _ in x] for _ in x]
        for pos, group in enumerate(data['groups']):
            group_len = len(group)
            for i in range(group_len):
                for j in range(i, group_len):
                    m, n = group[i], group[j]
                    edge_select[m][n] = edge_select[n][m] = 0   # self-loop is also excluded
            for aid in group:
                x_pieces[aid] = pieces[pos]
                x_pos[aid] = pos  # because the first group is <s>, therefore starting pid is already 1

        return {
            'x': x,
            'ad_mat': ad_mat,
            'props': data['props'],
            'pieces': pieces,
            'x_pieces': x_pieces,
            'x_pos': x_pos,
            'edge_select': edge_select
        }
    
    @staticmethod
    def process_step3(data_list, tokenizer, device='cpu'):
        '''collate data from step2'''
        # pad atom
        # [batch_size, N]
        xs, lens, x_pieces, x_pos = [], [], [], []
        for data in data_list:
            x = torch.tensor(data['x'], device=device)
            xs.append(x)
            lens.append(len(x))
            x_pieces.append(torch.tensor(data['x_pieces'], device=device))
            x_pos.append(torch.tensor(data['x_pos'], device=device))
        xs = pad_sequence(xs, batch_first=True, padding_value=tokenizer.atom_pad_idx())
        x_pieces = pad_sequence(x_pieces, batch_first=True, padding_value=tokenizer.pad_idx())
        x_pos = pad_sequence(x_pos, batch_first=True, padding_value=0)  # 0 is padding for position embedding
        atom_mask = torch.zeros(xs.shape[0], xs.shape[1] + 1, dtype=torch.long, device=device)
        atom_mask[torch.arange(xs.shape[0], device=device), lens] = 1
        atom_mask = atom_mask.cumsum(dim=1)[:, :-1]  # remove the superfluous column

        batch_size, node_num = xs.shape[0], xs.shape[1]
        edge_index, edge_attr, golden_edge, props = [], [], [], []
        in_piece_edge_idx = []
        edge_select = torch.zeros(batch_size, node_num, node_num, device=device) # [batch_size, N, N]
        none_bond = tokenizer.chem_vocab.bond_to_idx(None)
        for i, data in enumerate(data_list):
            ad_mat = data['ad_mat']
            offset = node_num * i
            props.append(data['props'])
            for m, row in enumerate(data['edge_select']):
                for n, val in enumerate(row):  # this is ad mat, 0-1 and 1-0 will both be added
                    edge_select[i][m][n] = val
                    attr = ad_mat[m][n]
                    begin, end = m + offset, n + offset
                    if attr != none_bond:
                        edge_index.append([begin, end])
                        # edge_index.append([end, begin])
                        # edge_attr.append(attr)
                        edge_attr.append(attr)
                        if val == 0:  # to select in-piece bonds for decoder
                            in_piece_edge_idx.append(len(edge_index) - 1)
                            # in_piece_edge_idx.append(len(edge_index) - 2)
                    if val == 1:
                        # balance none bond and normal bond (if not, pos/neg is about 0.022)
                        if attr != none_bond or random() < 0.022:
                            golden_edge.append(attr)
                        else:
                            edge_select[i][m][n] = 0
                        
        # pad pieces
        pieces = pad_sequence([torch.tensor(data['pieces'], dtype=torch.long, device=device) for data in data_list],
                              batch_first=True, padding_value=tokenizer.pad_idx())
        edge_attr = torch.tensor(edge_attr, dtype=torch.long, device=device)
        if len(edge_index):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous() # [E, 2]
        else:
            edge_index = torch.Tensor(2, 0, device=device).long()
        return {
            # 'x': F.one_hot(xs, num_classes=tokenizer.chem_vocab.num_atom_type()),    # [batch_size, N, node_dim]
            'x': xs,    # [batch_size, N]
            'x_pieces': x_pieces,   # [batch_size, N]
            'x_pos': x_pos,  # [batch_size, N]
            'atom_mask': atom_mask.bool(),  # [batch_size, N], mask paddings
            'pieces': pieces,   # [batch_size, seq_len]
            'edge_index': edge_index,
            'edge_attr': F.one_hot(edge_attr, num_classes=tokenizer.chem_vocab.num_bond_type()),  # [E, edge_dim]
            'edge_select': edge_select.bool(),  # [batch_size, N, N]
            'golden_edge': torch.tensor(golden_edge, dtype=torch.long),  # [E]
            'in_piece_edge_idx': in_piece_edge_idx, # [E']
            'props': torch.tensor(props)        # [batch_size, num_props]
        }

    def process(self):
        # load smiles
        file_path = os.path.join(self.root_path, self.file_path)
        with open(file_path, 'r') as fin:
            lines = fin.readlines()
        smiles = [s.strip('\n') for s in lines]
        # turn smiles into data type of PyG
        data_list = []
        for s in tqdm(smiles):
            mol = chem_utils.smiles2molecule(s, kekulize=True)
            if mol is None:
                continue
            data_list.append(self.process_step1(mol, self.tokenizer))
        torch.save(data_list, self.save_path)
        return data_list

    def __getitem__(self, idx):
        data = self.data[idx]
        return self.process_step2(data, self.tokenizer)
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, data_list):
        return self.process_step3(data_list, self.tokenizer)


def get_dataloader(fname, tokenizer, batch_size, shuffle=False, num_workers=4):
    dataset = BPEMolDataset(fname, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=dataset.collate_fn, num_workers=num_workers)


if __name__ == '__main__':
    import sys
    # dataset = BPEMolDataset(sys.argv[1], sys.argv[2])
    # data = dataset[0]
    # for key in data.keys:
    #     print(key, data[key])
    tokenizer = Tokenizer(sys.argv[2])
    dataloader = get_dataloader(sys.argv[1], tokenizer, 1, shuffle=True)
    neg, pos = 0, 0
    for batch in tqdm(dataloader):
        # print(batch['x'].shape[1])
        # print(batch['edge_index'].shape[1])
        # for key in batch:
        #     print(f'{key}: {batch[key]}')
        # break
        golden = batch['golden_edge']
        x_pieces = batch['x_pieces']
        edge_index = batch['edge_index']
        aid2pid = { i: x_pieces[0][i].item() for i in range(x_pieces.shape[1]) }
        print(edge_index.shape)
        for e in edge_index.t():
            begin, end = e
            begin, end = begin.item(), end.item()
            begin_pid, end_pid = aid2pid[begin], aid2pid[end]
            if begin_pid == end_pid:
                continue
            print(begin_pid, end_pid)
        break
        for e in golden:
            if e == tokenizer.chem_vocab.bond_to_idx(None):
                neg += 1
            else:
                pos += 1
    # print(f'pos {pos}, neg {neg}, pos / neg {pos / neg}')

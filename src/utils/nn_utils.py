#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from evaluation.utils import map_prop_to_idx
from data.mol_bpe import PIECE_CONNECT_NUM


def to_one_hot(idx, size):
    one_hot = [0 for _ in range(size)]
    one_hot[idx] = 1
    return one_hot


def multi_one_hot_embed(x, one_hot_sizes):
    # x is [data_num, features]
    res = []
    for i, size in enumerate(one_hot_sizes):
        if size == 1: # do not embed
            res.append(x[:, i].unsqueeze(-1))
        else:
            res.append(F.one_hot(x[:, i], num_classes=size))
    if len(res) == 1:
        return res[0]
    return torch.cat(res, dim=-1)


def pyg_batch_to_tensors(batch, vocab):
    # pad and turn Batch structure to tensors
    # data.x: [n, node_dim], data.edge_index: [2, e], data.edge_attr: [e, edge_dim]
    # data.edge_sets: [n, e, 2], (<front, attr>)
    data_list = batch.to_data_list()
    node_num = max([len(data.x) for data in data_list])
    edge_num = max([max([len(es) for es in data.edge_sets]) for data in data_list])
    x = []
    edge_index, edge_attr, edge_mask = [], [], []
    gold_node_type, gold_add_edge, gold_edge_dest = [], [], []
    add_edge_mask = []  # since bceloss has no ignore_idx option, mask is needed(1 for selected)
    attr_dim = data_list[0].edge_attr.shape[1]
    device = data_list[0].x.device
    na = len(vocab)  # number of atom types

    for i, data in enumerate(data_list):
        x.append(data.x[:, :na])  # only reserve atom type
        cur_edge_index, cur_edge_attr, cur_edge_mask = [], [], []
        offset = i * node_num
        add_end = False
        cur_gold_nt, cur_gold_ae, cur_gold_ed = [], [], []
        cur_ae_mask = []
        for n in range(node_num):
            add_edge_end = False
            if n < len(data.edge_sets):
                cur_gold_nt.append(torch.argmax(data.x[n][:na]).item())
                edge_set = data.edge_sets[n]
                for e in range(edge_num):
                    if e < len(edge_set):
                        cur_gold_ae.append(1)
                        cur_ae_mask.append(1)
                        after = n
                        front, attr = edge_set[e]
                        cur_gold_ed.append(front * vocab.num_bond_type() + attr.index(1))
                        after += offset
                        front += offset
                        cur_edge_mask.append(0)
                        cur_edge_mask.append(0)
                    else:
                        if not add_edge_end:
                            cur_gold_ae.append(0)
                            cur_ae_mask.append(1)
                            add_edge_end = True
                        cur_gold_ae.append(0)
                        cur_ae_mask.append(0)  # padding won't be selected
                        cur_gold_ed.append(vocab.bond_pad_idx())
                        after = -1
                        front, attr = -1, [0 for _ in range(attr_dim)]
                        cur_edge_mask.append(1)
                        cur_edge_mask.append(1)
                    cur_edge_index.append([after, front])
                    cur_edge_index.append([front, after])
                    cur_edge_attr.append(attr)
                    cur_edge_attr.append(attr)
            else:
                if not add_end:
                    cur_gold_nt.append(vocab.atom_end_idx())
                    add_end = True
                cur_gold_nt.append(vocab.atom_pad_idx())
                after = -1
                front, attr = -1, [0 for _ in range(attr_dim)]
                for e in range(edge_num):
                    cur_gold_ae.append(0)
                    cur_ae_mask.append(0)
                    cur_gold_ed.append(vocab.bond_pad_idx())
                    cur_edge_mask.append(1)
                    cur_edge_mask.append(1)
                    cur_edge_index.append([after, front])
                    cur_edge_index.append([front, after])
                    cur_edge_attr.append(attr)
                    cur_edge_attr.append(attr)
            if not add_edge_end:
                cur_gold_ae.append(0)
                cur_ae_mask.append(1)
        if not add_end:
            cur_gold_nt.append(vocab.atom_end_idx())
        edge_index.append(torch.tensor(cur_edge_index, device=device).view(-1, 2))
        edge_attr.append(torch.tensor(cur_edge_attr, device=device).view(-1, attr_dim))
        edge_mask.append(cur_edge_mask)
        gold_node_type.append(cur_gold_nt)
        gold_add_edge.append(cur_gold_ae)
        gold_edge_dest.append(cur_gold_ed)
        add_edge_mask.append(cur_ae_mask)
    x = pad_sequence(x, batch_first=True, padding_value=0) # [b, n, node_dim]
    edge_index = torch.stack(edge_index)  # [b, n*e, 2]
    edge_attr = torch.stack(edge_attr)    # [b, n*e, edge_dim]
    edge_mask = torch.tensor(edge_mask, device=device, requires_grad=False)  # [b, n*e]
    gold_node_type = torch.tensor(gold_node_type, device=device)  # [b, n+1]
    gold_add_edge = torch.tensor(gold_add_edge, dtype=torch.float, device=device) # [b, n*(e+1)]
    gold_edge_dest = torch.tensor(gold_edge_dest, device=device)  # [b, n*e]
    add_edge_mask = torch.tensor(add_edge_mask, device=device, dtype=torch.bool)  # [b, n*(e+1)]
    return x, edge_index, edge_attr, edge_mask,\
           (gold_node_type, gold_add_edge, gold_edge_dest, add_edge_mask)


class VAEEarlyStopping(EarlyStopping):
    def __init__(self, finish_anneal_step, **kwargs):
        self.finish_anneal_step = finish_anneal_step
        super(VAEEarlyStopping, self).__init__(**kwargs)
    
    def on_validation_end(self, trainer, pl_module):
        step = pl_module.global_step
        super(VAEEarlyStopping, self).on_validation_end(trainer, pl_module)
        if step < self.finish_anneal_step:
            self.best_score = torch.tensor(float('inf'))
            self.wait_count = 0
    

def common_config(args):
    config = {
        'lr': args.lr,
        'alpha': args.alpha,
        'beta': args.beta,
        'max_beta': args.max_beta,
        'step_beta': args.step_beta,
        'kl_warmup': args.kl_warmup,
        'kl_anneal_iter': args.kl_anneal_iter,
        'selected_properties': map_prop_to_idx(args.props)
    }
    return config


def predictor_config(args):
    config = {
        'predictor': {
            'dim_feature': args.latent_dim,
            'dim_hidden': args.predictor_hidden_dim,
            'num_property': len(args.props)
        },
    }
    return config


def encoder_config(args, vocab):
    dim_out = args.graph_embedding_dim  # will be turned into latent dim by decoder
    dim_in = vocab.num_atom_type()
    if hasattr(vocab, 'num_formal_charge'):
        dim_in += vocab.num_formal_charge()
    config = {
        'encoder': {
            'dim_in': dim_in,
            'num_edge_type': vocab.num_bond_type(),
            'dim_hidden': args.node_hidden_dim,
            'dim_out': dim_out
        },
    }
    return config


def ps_vae_config(args, tokenizer):
    chem_vocab = tokenizer.chem_vocab
    atom_dim = args.atom_embedding_dim + \
               args.piece_embedding_dim + \
               args.pos_embedding_dim
    config = {
        'encoder': {
            'dim_in': atom_dim,
            'num_edge_type': chem_vocab.num_bond_type(),
            'dim_hidden': args.node_hidden_dim,
            'dim_out': args.graph_embedding_dim
        },
        'vae_piece_decoder': {
            'atom_embedding_dim': args.atom_embedding_dim,
            'piece_embedding_dim': args.piece_embedding_dim,
            'max_pos': args.max_pos,
            'pos_embedding_dim': args.pos_embedding_dim,
            'piece_hidden_dim': args.piece_hidden_dim,
            'node_hidden_dim': args.node_hidden_dim,
            'num_edge_type': chem_vocab.num_bond_type(),
            'cond_dim': args.graph_embedding_dim,
            'latent_dim': args.latent_dim,
            'tokenizer': tokenizer,
            't': 4
        }
    }
    return config
#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pl_models import PSVAEModel
from data import bpe_dataset
from data.mol_bpe import Tokenizer
from utils.logger import print_log
from utils.nn_utils import VAEEarlyStopping
from utils.nn_utils import common_config, predictor_config, encoder_config
from utils.nn_utils import ps_vae_config


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     pl.utilities.seed.seed_everything(seed=seed)

setup_seed(2021)


def train(model, train_loader, valid_loader, test_loader, args):
    checkpoint_callback = ModelCheckpoint(
                            monitor=args.monitor,
                          )
    print_log('Using vae kl warmup early stopping strategy')
    anneal_step = args.kl_warmup + (args.max_beta // args.step_beta - 1) * args.kl_anneal_iter
    early_stop_callback = VAEEarlyStopping(
                            finish_anneal_step=anneal_step,
                            monitor=args.monitor,
                            patience=args.patience
                          )
    trainer_config = {
        'gpus': args.gpus,
        'max_epochs': args.epochs,
        'default_root_dir': args.save_dir,
        'callbacks': [checkpoint_callback, early_stop_callback],
        'gradient_clip_val': args.grad_clip
    }
    if len(args.gpus.split(',')) > 1:
        trainer_config['accelerator'] = 'dp'
    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model, train_loader, valid_loader)
    # test
    trainer.test(model, test_dataloaders=test_loader)


def parse():
    """parse command"""
    parser = argparse.ArgumentParser(description='training overall model for molecule generation')
    parser.add_argument('--train_set', type=str, required=True, help='path of training dataset')
    parser.add_argument('--valid_set', type=str, required=True, help='path of validation dataset')
    parser.add_argument('--test_set', type=str, required=True, help='path of test dataset')
    parser.add_argument('--vocab', type=str, required=True, help='path of vocabulary (.pkl) or bpe vocab(.txt)')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--save_dir', type=str, required=True, help='path to store the model')
    parser.add_argument('--batch_size', type=int, default=64, help='size of mini-batch')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, required=True,
                        help='balancing reconstruct loss and predictor loss')
    # vae training
    parser.add_argument('--beta', type=float, default=0.001,
                        help='balancing kl loss and other loss')
    parser.add_argument('--step_beta', type=float, default=0.0005,
                        help='value of beta increasing step')
    parser.add_argument('--max_beta', type=float, default=0.005)
    parser.add_argument('--kl_warmup', type=int, default=2000,
                        help='Within these steps beta is set to 0')
    parser.add_argument('--kl_anneal_iter', type=int, default=1000)

    parser.add_argument('--num_workers', type=int, default=4, help='number of cpus to load data')
    parser.add_argument('--gpus', default=None, help='gpus to use')
    parser.add_argument('--epochs', type=int, default=20, help='max epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping number of epochs')
    parser.add_argument('--grad_clip', type=float, default=0,
                        help='clip large gradient to prevent gradient boom')
    parser.add_argument('--monitor', type=str, default='val_loss',
                        help='Value to monitor in early stopping')

    # model parameters
    parser.add_argument('--props', type=str, nargs='+', choices=['qed', 'sa', 'logp', 'gsk3b', 'jnk3'],
                        default=['qed', 'logp'], help='properties to predict')
    parser.add_argument('--predictor_hidden_dim', type=int, default=200,
                        help='hidden dim of predictor (MLP)')
    parser.add_argument('--node_hidden_dim', type=int, default=100,
                        help='dim of node hidden embedding in encoder and decoder')
    parser.add_argument('--graph_embedding_dim', type=int, default=200,
                        help='dim of graph embedding by encoder and also condition for ae decoders')
    parser.add_argument('--latent_dim', type=int, default=56,
                        help='dim of latent z for vae decoders')
    # ps-vae decoder only
    parser.add_argument('--max_pos', type=int, default=50,
                        help='Max number of pieces')
    parser.add_argument('--atom_embedding_dim', type=int, default=50,
                        help='Embedding dim for a single atom')
    parser.add_argument('--piece_embedding_dim', type=int, default=100,
                        help='Embedding dim for piece')
    parser.add_argument('--pos_embedding_dim', type=int, default=50,
                        help='Position embedding of piece')
    parser.add_argument('--piece_hidden_dim', type=int, default=200,
                        help='Hidden dim for rnn used in piece generation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    print_log(args)
    print_log('loading data ...')
    tokenizer = Tokenizer(args.vocab)
    vocab = tokenizer.chem_vocab
    train_loader = bpe_dataset.get_dataloader(args.train_set, tokenizer, batch_size=args.batch_size,
                                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = bpe_dataset.get_dataloader(args.valid_set, tokenizer, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
    test_loader = bpe_dataset.get_dataloader(args.test_set, tokenizer, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)


    # config and create model
    print('creating model ...')
    config = {**common_config(args), **encoder_config(args, vocab), **predictor_config(args)}
    # config of encoder is also updated
    config.update(ps_vae_config(args, tokenizer))
    model = PSVAEModel(config, tokenizer)
    print_log(f'config: {config}')
    print(model)

    # train from start
    print('start training')
    train(model, train_loader, valid_loader, test_loader, args)

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import random
from tqdm import tqdm
import argparse
from functools import partial

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.logger import print_log

########### Import your packages below ##########
from guacamol.benchmark_suites import goal_directed_benchmark_suite

from predictor import Dataset, GuacaPredictor, HierG2GDataset, collate_fn, JTVAEDataset


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2021)


class TrainConfig:
    def __init__(self, save_dir, lr, max_epoch, metric_min_better=True, patience=3, grad_clip=None):
        self.save_dir = save_dir
        self.lr = lr
        self.max_epoch = max_epoch
        self.metric_min_better = metric_min_better
        self.patience = patience
        self.grad_clip = grad_clip

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config):
        self.model = model
        self.config = config
        self.optimizer = self.get_optimizer()
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {
                'scheduler': None,
                'frequency': None
            }
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # distributed training
        self.local_rank = -1

        # log
        self.model_dir = os.path.join(config.save_dir, 'checkpoint')
        self.writer = None  # initialize right before training

        # training process recording
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.patience = config.patience


    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        else:
            data = data.to(device)
        return data

    def _is_main_proc(self):
        return self.local_rank == 0 or self.local_rank == -1

    def _train_epoch(self, device):
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            loss = self.train_step(batch, self.global_step)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if hasattr(t_iter, 'set_postfix'):
                t_iter.set_postfix(loss=loss.item())
            self.global_step += 1
            if self.sched_freq == 'batch':
                self.scheduler.step()
        if self.sched_freq == 'epoch':
            self.scheduler.step()
    
    def _valid_epoch(self, device):
        metric_arr = []
        self.model.eval()
        with torch.no_grad():
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                metric = self.valid_step(batch, self.valid_global_step)
                metric_arr.append(metric.cpu().item())
                self.valid_global_step += 1
        self.model.train()
        # judge
        valid_metric = np.mean(metric_arr)
        if self._metric_better(valid_metric):
            self.patience = self.config.patience
            if self._is_main_proc():
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                module_to_save = self.model.module if self.local_rank == 0 else self.model
                torch.save(module_to_save, save_path)
        else:
            self.patience -= 1
        self.last_valid_metric = valid_metric
    
    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def train(self, device_ids, local_rank):
        # set local rank
        self.local_rank = local_rank
        # init writer
        if self._is_main_proc():
            self.writer = SummaryWriter(self.config.save_dir)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            with open(os.path.join(self.config.save_dir, 'train_config.json'), 'w') as fout:
                json.dump(self.config.__dict__, fout)
        # main device
        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device('cpu' if main_device_id == -1 else f'cuda:{main_device_id}')
        self.model.to(device)
        if local_rank != -1:
            print_log(f'Using data parallel, local rank {local_rank}, all {device_ids}')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank
            )
        else:
            print_log(f'training on {device_ids}')
        for _ in range(self.config.max_epoch):
            print_log(f'epoch{self.epoch} starts') if self._is_main_proc() else 1
            self._train_epoch(device)
            print_log(f'validating ...') if self._is_main_proc() else 1
            self._valid_epoch(device)
            self.epoch += 1
            if self.patience <= 0:
                break

    def log(self, name, value, step):
        if self._is_main_proc():
            self.writer.add_scalar(name, value, step)

    ########## Overload these functions below ##########
    # define optimizer
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        lam = lambda epoch: 1 / (epoch + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return {
            'scheduler': scheduler,
            'frequency': 'epoch' # or batch
        }

    # train step, note that batch should be dict/list/tuple or other objects with .to(device) attribute
    def train_step(self, batch, batch_idx):
        loss = self.model(batch['embeds'], batch['scores'])
        self.log('Loss/train', loss, batch_idx)
        return loss

    # validation step
    def valid_step(self, batch, batch_idx):
        loss = self.model(batch['embeds'], batch['scores'])
        self.log('Loss/validation', loss, batch_idx)
        return loss


def parse():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')

    # guacamol related
    parser.add_argument('--suite', type=str, default='v2', help='version of guacamol')

    # vae related
    parser.add_argument('--ckpt', type=str, required=True, help='path to trained vae')
    parser.add_argument('--model_type', type=str, default='vae', choices=['vae', 'hierg2g', 'jtvae'],
                        help='Type of model to load, vae for our implemenation')
    parser.add_argument('--vocab', type=str, default=None, help='When model type is hierg2g, the vocab path is needed')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=8)

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    return parser.parse_args()


def main(args):
    # load benchmarks
    benchmarks = goal_directed_benchmark_suite(version_name=args.suite)
    scoring_funcs = [benchmark.objective.score_list for benchmark in benchmarks]
    func_names = [benchmark.name for benchmark in benchmarks]
    ########### load your train / valid set ###########
    if args.model_type == 'vae':
        print('here')
        train_set = Dataset(args.ckpt, args.train_set, scoring_funcs)
        valid_set = Dataset(args.ckpt, args.valid_set, scoring_funcs)
    elif args.model_type == 'jtvae':
        train_set = JTVAEDataset(args.ckpt, args.train_set, scoring_funcs)
        valid_set = JTVAEDataset(args.ckpt, args.valid_set, scoring_funcs)
    elif args.model_type == 'hierg2g':
        device = torch.device('cpu') if args.gpus[0] == -1 else torch.device(f'cuda:{args.gpus[0]}')
        train_set = HierG2GDataset((args.vocab, args.ckpt), args.train_set, scoring_funcs,
                                   process_batch_size=args.batch_size, device=device,
                                   num_workers=args.num_workers)
        valid_set = HierG2GDataset((args.vocab, args.ckpt), args.valid_set, scoring_funcs,
                                   process_batch_size=args.batch_size, device=device,
                                   num_workers=args.num_workers)
    else:
        raise NotImplementedError('model type not recognized')

    ########## set your collate_fn ##########

    ########## define your model #########
    init_smiles = [train_set[i][0] for i in range(4)]
    init_smiles = ['CCN(C(=O)CO)C1CCN(Cc2ccc(-c3ccc(F)cc3)cc2)C(=O)C1=O',
                   'COP1(=S)NCC(c2cc(Cl)c(O)c(F)c2F)N1']
    model = GuacaPredictor(train_set.get_embed_dim(), func_names, scoring_funcs, init_smiles)

    if len(args.gpus) > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle)
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    config = TrainConfig(args.save_dir, args.lr, args.max_epoch, grad_clip=args.grad_clip)
    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = parse()
    main(args)

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import random
from tqdm import tqdm

import numpy as np
import torch
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.utils.helpers import setup_default_logger
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from rdkit import Chem

from pl_models import PSVAEModel


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class Generator(DistributionMatchingGenerator):
    def __init__(self, gpu: int, ckpt: str):
        super().__init__()
        
        if gpu == -1:
            loc = torch.device('cpu')
        else:
            loc = torch.device(f'cuda:{gpu}')
        model = PSVAEModel.load_from_checkpoint(ckpt)
        model.to(loc)
        self.model = model
        self.device = loc

    def generate(self, number_samples: int):
        zs = self.model.sample_z(number_samples, self.device)
        smis = []
        for z in tqdm(zs):
            mol = self.model.inference_single_z(z, max_atom_num=60, add_edge_th=0.5, temperature=0.8)
            if not isinstance(mol, str) and mol is not None:
                Chem.SanitizeMol(mol)
                smis.append(Chem.MolToSmiles(mol))
            else:
                smis.append(mol)
        return smis


class FakeGenerator(DistributionMatchingGenerator):
    def __init__(self, smiles):
        self.smiles = smiles
    
    def generate(self, number_samples: int):
        np.random.shuffle(self.smiles)
        return self.smiles[:number_samples]


def parse():
    parser = argparse.ArgumentParser(description='Distribution learning benchmark for vae models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--ckpt', type=str, required=True, help='Full path to SMILES RNN model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--dist_file', type=str, required=True, help='Distribution file (Training set)')
    parser.add_argument('--suite', default='v2')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')

    return parser.parse_args()


def main(args):
    setup_seed(args.seed)
    setup_default_logger()

    json_file_path = os.path.join(args.output_dir, 'distribution_learning_results.json')

    smiles_file = os.path.join(args.output_dir, 'generated.smi')
    if os.path.exists(smiles_file):
        with open(smiles_file, 'r') as fin:
            smiles = fin.read().strip().split('\n')
    else:
        generator = Generator(args.gpu, args.ckpt)
        smiles = generator.generate(10000)
        with open(smiles_file, 'w') as fout:
            fout.writelines(list(map(lambda line: line + '\n', smiles)))
    fake_generator = FakeGenerator(smiles)
    assess_distribution_learning(
        fake_generator,
        chembl_training_file=args.dist_file,
        json_output_file=json_file_path,
        benchmark_version=args.suite
    )


if __name__ == '__main__':
    main(parse())

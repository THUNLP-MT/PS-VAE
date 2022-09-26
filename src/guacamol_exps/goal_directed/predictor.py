#!/usr/bin/python
# -*- coding:utf-8 -*-
from functools import partial
from tqdm import tqdm
import multiprocessing as mp

from rdkit import Chem
import torch
import torch.nn as nn

from pl_models import PSVAEModel
from modules.predictor import Predictor
from utils.chem_utils import smiles2molecule

from time import time

class Dataset(torch.utils.data.Dataset):
    def __init__(self, model_ckpt, fpath, scoring_funcs,
                 process_batch_size=32, device=torch.device('cpu'),
                 num_workers=8):
        super().__init__()
        self.model_ckpt = model_ckpt
        self.device = device
        with open(fpath, 'r') as fin:
            self.smis = fin.read().strip().split('\n')
        self.embeds, self.scores = [], [[] for _ in scoring_funcs]

        if num_workers > 1:
            pool = mp.Pool(num_workers)
            spliced_smis, splice_size = [], (len(self.smis) + num_workers - 1) // num_workers
            for begin in range(0, len(self.smis), splice_size):
                end = min(begin + splice_size, len(self.smis))
                smis = self.smis[begin:end]
                spliced_smis.append(smis)

            # get embeds and scores
            res = pool.map(partial(self.get_embed_scores, scoring_funcs=scoring_funcs,
                                   batch_size=process_batch_size), tqdm(spliced_smis))
            pool.close()
        else:
            res = [self.get_embed_scores(self.smis, scoring_funcs, process_batch_size)]

        for embeds, scores in res:
            self.embeds.extend(embeds)
            for i, s in enumerate(scores):
                self.scores[i].extend(s)
        print(f'valid number of samples: {len(self.embeds)}')


    def load_model(self):
        model = PSVAEModel.load_from_checkpoint(self.model_ckpt)
        model.to(self.device)
        model.eval()
        return model

    def get_embed_scores(self, seq, scoring_funcs, batch_size):
        model = self.load_model()
        if isinstance(seq, str):
            seq = [seq]
        origin_seq = seq
        embeds, scores = [], [[] for _ in scoring_funcs]
        for begin in tqdm(range(0, len(origin_seq), batch_size)):
            end = min(begin + batch_size, len(origin_seq))
            seq = origin_seq[begin:end]
            good_seq = []
            with torch.no_grad():
                mols = [smiles2molecule(smi) for smi in seq]
                good_mols = []
                for i, mol in enumerate(mols):
                    if mol is not None:
                        good_mols.append(mol)
                        good_seq.append(seq[i])
                try:
                    zs = model.get_z_from_batch_mol(good_mols)
                except Exception as e:
                    print(e)
                    print('Error occurred, drop this batch')
                    continue
                embeds.extend(zs.cpu().numpy())
            for i, func in enumerate(scoring_funcs):
                scores[i].extend(func(good_seq))
        return embeds, scores

    def __len__(self):
        return len(self.embeds)
    
    def __getitem__(self, idx):
        scores = [mol_scores[idx] for mol_scores in self.scores]
        return self.smis[idx], self.embeds[idx], scores

    def get_embed_dim(self):
        return len(self.embeds[0])



class InferenceModel:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def emb_to_seq(self, embeddings):
        smis = []
        with torch.no_grad():
            embeddings = torch.tensor(embeddings, device=self.device)
            for z in embeddings:
                mol = self.model.inference_single_z(z, max_atom_num=60, add_edge_th=0.5, temperature=0.8)
                if not isinstance(mol, str) and mol is not None:
                    Chem.SanitizeMol(mol)
                    smis.append(Chem.MolToSmiles(mol))
                else:
                    smis.append(mol)
        return smis

    def seq_to_emb(self, seq):
        if isinstance(seq, str):
            seq = [seq]
        with torch.no_grad():
            mols = [smiles2molecule(smi) for smi in seq]
            ill_idxs, good_mols = [], []
            for i, mol in enumerate(mols):
                if mol is not None:
                    good_mols.append(mol)
                
                
            zs = self.model.get_z_from_batch_mol(mols)
            zs = zs.cpu().numpy()
        return zs


class GuacaPredictor(nn.Module):
    def __init__(self, embed_dim, func_names, scoring_funcs, init_smiles):
        super().__init__()
        self.predictor = Predictor(embed_dim, embed_dim // 2, len(scoring_funcs))
        self.pred_loss = nn.MSELoss()
        self.init_smiles = init_smiles
        self.func2idx, self.func2name = {}, {}
        for i, func in enumerate(scoring_funcs):
            key = self.get_func_key(func)
            self.func2idx[key] = i
            self.func2name[key] = func_names[i]
            print(key, func_names[i])
        assert len(self.func2idx) == len(scoring_funcs)

    def get_func_key(self, func):
        scores = func(self.init_smiles)
        key = '\t'.join([str(round(s, 4)) for s in scores])
        return key

    def forward(self, embeds, targets):
        scores = self.predictor(embeds)  # [batch_size, num_properties]
        loss = self.pred_loss(scores, targets)
        return loss

    def predict(self, embeds, scoring_func=None):
        squeeze = False
        if len(embeds.shape) == 1:
            embeds = embeds.unsqueeze(0)
            squeeze = True
        scores = self.predictor(embeds)
        if scoring_func is None:
            return scores
        key = self.get_func_key(scoring_func)
        res = scores[:, self.func2idx[key]]
        if squeeze:
            res = res.squeeze()
        return res
#!/usr/bin/python
# -*- coding:utf-8 -*-
# import numpy as np
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
# import sascorer
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
sys.path.remove(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

def smiles2molecule(smiles: str):
    '''turn smiles to molecule'''
    return Chem.MolFromSmiles(smiles)


def similarity(mol1, mol2):
    fps1 = AllChem.GetMorganFingerprint(mol1, 2)
    fps2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.TanimotoSimilarity(fps1, fps2)


def num_long_cycles(mol):
    """Calculate the number of long cycles.
    Args:
      mol: Molecule. A molecule.
    Returns:
      negative cycle length.
    """
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if not cycle_list:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return -cycle_length


def get_sa(molecule):
    '''return synthesis accessibility of given molecule.
       The value ranges from 1 to 10, the lower, the better (the easier to make)'''
    return sascorer.calculateScore(molecule)


def get_penalized_logp(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    
    # return log_p + SA + cycle_score
    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def get_qed(molecule):
    '''QED of given molecule. The value ranges from 0 to 1, the higher, the better'''
    return qed(molecule)


def eval_funcs_dict():
    eval_funcs = {
        'qed': get_qed,
        'sa': get_sa,
        'logp': get_penalized_logp
    }
    return eval_funcs


def get_normalized_property_scores(mol):
    # make every property approximately in range (0, 1), and all are the higher, the better
    qed = get_qed(mol)
    sa = get_sa(mol)
    logp = get_penalized_logp(mol)
    return [qed, 1 - sa / 10, (logp + 10) / 13]  # all are the higher, the better


def restore_property_scores(normed_props):
    return [normed_props[0], 10 * (1 - normed_props[1]),
            13 * normed_props[2] - 10]


PROP_TH = [0.6, 4.0, 0]
NORMALIZED_TH = None
PROPS = ['qed', 'sa', 'logp']
def map_prop_to_idx(props):
    global PROPS
    idxs = []
    p2i = {}
    for i, p in enumerate(PROPS):
        p2i[p] = i
    for p in props:
        if p in p2i:
            idxs.append(p2i[p])
        else:
            raise ValueError('Invalid property')
    return sorted(list(set(idxs)))


def overpass_th(prop_vals, prop_idx):
    ori_prop_vals = [0 for _ in PROPS]
    for i, val in zip(prop_idx, prop_vals):
        ori_prop_vals[i] = val
    ori_prop_vals = restore_property_scores(ori_prop_vals)
    for i in prop_idx:
        if ori_prop_vals[i] < PROP_TH[i]:
            return False
    return True


class TopStack:
    '''Only save the top-k results and the corresponding '''
    def __init__(self, k, cmp):
        # k: capacity, cmp: binary comparator indicating if x is prior to y
        self.k = k
        self.stack = []
        self.cmp = cmp

    def push(self, val, data=None):
        i = len(self.stack) - 1
        while i >= 0:
            if self.cmp(self.stack[i][0], val):
                break
            else:
                i -= 1
        i += 1
        self.stack.insert(i, (val, data))
        if len(self.stack) > self.k:
            self.stack.pop()
    
    def get_iter(self):
        return iter(self.stack)


if __name__ == '__main__':
    # args = parse()
    # init_stats(args.data, args.cpus)
    eg = 'CN(C)CC[C@@H](c1ccc(Br)cc1)c1ccccn1'
    m = smiles2molecule(eg)
    eval_funcs = eval_funcs_dict()
    for key in eval_funcs:
        f = eval_funcs[key]
        print(f'{key}: {f(m)}')
    print(f'normalized: {get_normalized_property_scores(m)}')
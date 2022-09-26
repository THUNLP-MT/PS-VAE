from collections import defaultdict
from queue import Queue
import heapq
from copy import copy

import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType
import torch
from torch_geometric.data import Data

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':3, 'O':2, 'P':5, 'S':6} #, 'Se':4, 'Si':4}
Bond_List = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]  # aromatic bonds are shits


class GeneralVocab:
    def __init__(self, atom_special=None, bond_special=None):
        # atom
        self.idx2atom = list(MAX_VALENCE.keys())
        if atom_special is None:
            atom_special = []
        self.idx2atom += atom_special
        self.atom2idx = { atom: i for i, atom in enumerate(self.idx2atom) }
        # bond
        self.idx2bond = copy(Bond_List)
        if bond_special is None:
            bond_special = []
        self.idx2bond += bond_special
        self.bond2idx = { bond: i for i, bond in enumerate(self.idx2bond) }
        
        self.atom_special = atom_special
        self.bond_special = bond_special
    
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]
    
    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def idx_to_bond(self, idx):
        return self.idx2bond[idx]
    
    def bond_to_idx(self, bond):
        return self.bond2idx[bond]
    
    def bond_idx_to_valence(self, idx):
        bond_enum = self.idx2bond[idx]
        if bond_enum == BondType.SINGLE:
            return 1
        elif bond_enum == BondType.DOUBLE:
            return 2
        elif bond_enum == BondType.TRIPLE:
            return 3
        else:   # invalid bond
            return -1
    
    def num_atom_type(self):
        return len(self.idx2atom)
    
    def num_bond_type(self):
        return len(self.idx2bond)


def smiles2molecule(smiles: str, kekulize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    if kekulize:
        Chem.Kekulize(mol)
    return mol


def molecule2smiles(mol):
    return Chem.MolToSmiles(mol)


def rec(mol):
    return smiles2molecule(molecule2smiles(mol), kekulize=False)


def data2molecule(vocab, data, sanitize=True):
    '''turn PyG data to molecule'''
    mol = Chem.RWMol()
    idx2atom = []
    for atom_idx in data.x:     # add atoms
        if not isinstance(atom_idx, int):  # one-hot form
            atom_idx = torch.argmax(atom_idx[:len(vocab)])
        atom = Chem.Atom(vocab.idx_to_atom(int(atom_idx)))
        idx2atom.append(mol.AddAtom(atom))
    edge_list = data.edge_index.t()  # [num, 2]
    edge_dict = {}
    for edge, attr in zip(edge_list, data.edge_attr):
        i1, i2 = edge
        i1, i2 = int(i1), int(i2)
        if i1 > i2:
            i1, i2 = i2, i1
        key = f'{i1},{i2}'
        if key in edge_dict:
            continue
        edge_dict[key] = True
        a1, a2 = idx2atom[i1], idx2atom[i2]
        if len(attr) > 1:
            attr = torch.argmax(attr)
        bond_type = vocab.get_bond_enum(int(attr))
        # if bond_type == BondType.AROMATIC:
        #     bond_type = BondType.DOUBLE
        mol.AddBond(a1, a2, bond_type)
    mol = mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def mol2file(mol, file_name, grid=False, molsPerRow=6):
    if isinstance(mol, list):
        if not grid:
            for i, m in enumerate(mol):
                Draw.MolToFile(m, f'{i}_{file_name}')
        else:
            img = Draw.MolsToGridImage(mol, molsPerRow=molsPerRow, subImgSize=(400, 400))
            with open(file_name, 'wb') as fig:
                img.save(fig)
            
    else:
        Draw.MolToFile(mol, file_name)


def get_submol(mol, idx2atom):
    sub_mol = Chem.RWMol()
    oid2nid = {}  # old id to new id
    for nid, oid in enumerate(idx2atom.keys()):
        atom = mol.GetAtomWithIdx(oid)
        new_atom = Chem.Atom(atom.GetSymbol())
        sub_mol.AddAtom(new_atom)
        oid2nid[oid] = nid
    for aid in idx2atom:
        atom = mol.GetAtomWithIdx(aid)
        for bond in atom.GetBonds():
            nei_id = bond.GetBeginAtomIdx()
            if nei_id == aid:
                nei_id = bond.GetEndAtomIdx()
            if nei_id in idx2atom and nei_id < aid:
                sub_mol.AddBond(oid2nid[aid], oid2nid[nei_id], bond.GetBondType())
            
    sub_mol = sub_mol.GetMol()
    return sub_mol
    # try:
    #     Chem.SanitizeMol(sub_mol)
    #     return sub_mol
    # except Exception:
    #     return None


def cnt_atom(smi):
    cnt = 0
    for c in smi:
        if c in MAX_VALENCE:
            cnt += 1
    return cnt


def get_base64(mol):
    return Chem.RDKFingerprint(mol).ToBase64()


def dfs_order(mol, root):
    '''return list of atoms in dfs order and idx2order mapping dict'''
    stack = [root]
    visited = {}
    order_list = []
    idx2order = {}
    visited[root.GetIdx()] = True
    while stack:
        atom = stack.pop()
        idx2order[atom.GetIdx()] = len(order_list)
        order_list.append(atom)
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx not in visited:
                stack.append(nei)
                visited[idx] = True
    return order_list, idx2order


def bfs_order(mol, root):
    queue = Queue()
    queue.put(root)
    visited = {}
    order_list = []
    idx2order = {}
    visited[root.GetIdx()] = True
    while not queue.empty():
        atom = queue.get()
        idx2order[atom.GetIdx()] = len(order_list)
        order_list.append(atom)
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx not in visited:
                queue.put(nei)
                visited[idx] = True
    return order_list, idx2order


def bfs_morgan_order(m):
    # get root and id2flow
    id2flow = {}
    for i in range(m.GetNumAtoms()):
        atom = m.GetAtomWithIdx(i)
        id2flow[i] = len(atom.GetBonds())
    k = len(set(id2flow.values()))
    while True:
        new_id2flow = {}
        for i in range(m.GetNumAtoms()):
            atom = m.GetAtomWithIdx(i)
            new_id2flow[i] = 0
            for bond in atom.GetBonds():
                nei_id = bond.GetBeginAtomIdx()
                if nei_id == i:
                    nei_id = bond.GetEndAtomIdx()
                new_id2flow[i] += id2flow[nei_id]
        new_k = len(set(new_id2flow.values()))
        if new_k <= k:
            break
        else:
            k, id2flow = new_k, new_id2flow
    for idx in id2flow:
        atom = m.GetAtomWithIdx(idx)
        atom.SetAtomMapNum(id2flow[idx])
    root_idx, _ = min(id2flow.items(), key=lambda x: x[1])
    root = m.GetAtomWithIdx(root_idx)
    # bfs order. Nodes with lower flow value has higher priority
    queue = Queue()
    queue.put(root)
    visited = {}
    order_list = []
    idx2order = {}
    visited[root.GetIdx()] = True
    while not queue.empty():
        atom = queue.get()
        idx2order[atom.GetIdx()] = len(order_list)
        order_list.append(atom)
        neis = []
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx not in visited:
                visited[idx] = True
                neis.append(idx)
        neis.sort(key=lambda i: id2flow[i])     # nodes with lower flow value has higher priority
        for idx in neis:
            queue.put(m.GetAtomWithIdx(idx))
    return order_list, idx2order


def bfs_morgan_order_extended(m):
    '''get the atom with the smallest connectivity value as the next atom (ignore radius limit of bfs)'''
    # get root and id2flow
    id2flow = {}
    for i in range(m.GetNumAtoms()):
        atom = m.GetAtomWithIdx(i)
        id2flow[i] = len(atom.GetBonds())
    k = len(set(id2flow.values()))
    while True:
        new_id2flow = {}
        for i in range(m.GetNumAtoms()):
            atom = m.GetAtomWithIdx(i)
            new_id2flow[i] = 0
            for bond in atom.GetBonds():
                nei_id = bond.GetBeginAtomIdx()
                if nei_id == i:
                    nei_id = bond.GetEndAtomIdx()
                new_id2flow[i] += id2flow[nei_id]
        new_k = len(set(new_id2flow.values()))
        if new_k <= k:
            break
        else:
            k, id2flow = new_k, new_id2flow
    for idx in id2flow:
        atom = m.GetAtomWithIdx(idx)
        atom.SetAtomMapNum(id2flow[idx])
    root_idx, _ = min(id2flow.items(), key=lambda x: x[1])
    root = m.GetAtomWithIdx(root_idx)
    # extended bfs order. Nodes with lower flow value has higher priority
    class PrioAtom:
        def __init__(self, flow, atom):
            self.flow = flow
            self.atom = atom
        
        def __lt__(self, other):
            return self.flow < other.flow
    heap = []
    heapq.heappush(heap, PrioAtom(id2flow[root_idx],root))
    visited = {}
    order_list = []
    idx2order = {}
    visited[root.GetIdx()] = True
    while len(heap):
        atom = heapq.heappop(heap).atom
        idx2order[atom.GetIdx()] = len(order_list)
        order_list.append(atom)
        neis = []
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx not in visited:
                visited[idx] = True
                heapq.heappush(heap, PrioAtom(id2flow[idx], m.GetAtomWithIdx(idx)))
    return order_list, idx2order


def bfs_order_by_admat(admat):
    root_idx = 0
    queue = Queue()
    queue.put(root_idx)
    visited = {}
    order_list = []
    idx2order = {}
    visited[root_idx] = True
    while not queue.empty():
        next_id = queue.get()
        idx2order[next_id] = len(order_list)
        order_list.append(next_id)
        neis = []
        for nei_id, has_edge in enumerate(admat[next_id]):
            if not has_edge:
                continue
            if nei_id not in visited:
                visited[nei_id] = True
                queue.put(nei_id)
    return order_list, idx2order


def bfs_morgan_order_extended_by_admat(admat):
    # get root and id2flow
    id2flow = {}
    for i, row in enumerate(admat):
        id2flow[i] = sum(row)
    k = len(set(id2flow.values()))
    while True:
        new_id2flow = {}
        for i, row in enumerate(admat):
            new_id2flow[i] = 0
            for j, has_edge in enumerate(row):
                if not has_edge:
                    continue
                new_id2flow[i] += id2flow[j]
        new_k = len(set(new_id2flow.values()))
        if new_k <= k:
            break
        else:
            k, id2flow = new_k, new_id2flow
    root_idx, _ = min(id2flow.items(), key=lambda x: x[1])
    # extended bfs order. Nodes with lower flow value has higher priority
    heap = []
    heapq.heappush(heap, (id2flow[root_idx],root_idx))
    visited = {}
    order_list = []
    idx2order = {}
    visited[root_idx] = True
    while len(heap):
        _, next_id = heapq.heappop(heap)
        idx2order[next_id] = len(order_list)
        order_list.append(next_id)
        neis = []
        for nei_id, has_edge in enumerate(admat[next_id]):
            if not has_edge:
                continue
            if nei_id not in visited:
                visited[nei_id] = True
                heapq.heappush(heap, (id2flow[nei_id], nei_id))
    return order_list, idx2order


def del_N_positive(mol):
    # may produce single atoms or unconnected blocks
    Chem.Kekulize(mol)
    remove_Ns = {}
    rw_mol = Chem.RWMol()
    remove_atoms = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        new_atom = Chem.Atom(atom.GetSymbol())
        if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1:
            if atom.GetNumExplicitHs() == 0:
                oxygen = []
                for nei in atom.GetNeighbors():
                    if nei.GetSymbol() == 'O':
                        oxygen.append(nei.GetIdx()) # reduce NO2 to NH2
                if len(oxygen) == 2:
                    remove_atoms.extend(oxygen)
                else:
                    remove_Ns[i] = False
        rw_mol.AddAtom(new_atom)
    
    # ring_info = mol.GetRingInfo()
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        nitro, other = None, None
        if begin in remove_Ns and not remove_Ns[begin]:
            nitro, other = begin, end
        elif end in remove_Ns and not remove_Ns[end]:
            nitro, other = end, begin
        # if nitro is not None and ring_info.NumAtomRings(other) > 0:
        if nitro is not None:
            remove_Ns[nitro] = True
            continue
        rw_mol.AddBond(begin, end, bond.GetBondType())

    # find max connect block
    aid2bid, blocks = {}, []
    for i in range(rw_mol.GetNumAtoms()):
        if i in aid2bid:
            continue
        aid2bid[i] = len(blocks)
        blocks.append([])
        blocks[-1].append(i)
        atom = rw_mol.GetAtomWithIdx(i)
        queue = Queue()
        queue.put(i)
        visited = {}
        visited[i] = True
        while not queue.empty():
            aid = queue.get()
            atom = rw_mol.GetAtomWithIdx(aid)
            for nei in atom.GetNeighbors():
                idx = nei.GetIdx()
                if idx not in visited:
                    queue.put(idx)
                    aid2bid[idx] = aid2bid[i]
                    blocks[-1].append(idx)
                    visited[idx] = True
    final_bid = max([i for i in range(len(blocks))], key=lambda i: len(blocks[i]))
    for aid in aid2bid:
        if aid2bid[aid] != final_bid:
            remove_atoms.append(aid)

    remove_atoms = sorted(remove_atoms, reverse=True)
    for i in remove_atoms:
        rw_mol.RemoveAtom(i)
    new_mol = rw_mol.GetMol()
    Chem.SanitizeMol(new_mol)
    return new_mol


def shortest_path_len(i, j, mol):
    queue = Queue()
    queue.put((mol.GetAtomWithIdx(i), 1))
    visited = {}
    visited[i] = True
    while not queue.empty():
        atom, dist = queue.get()
        neis = []
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx == j:
                return dist + 1
            if idx not in visited:
                visited[idx] = True
                neis.append(idx)
                queue.put((mol.GetAtomWithIdx(idx), dist + 1))
    return None


def cycle_check(i, j, mol):
    cycle_len = shortest_path_len(i, j, mol)
    return cycle_len is None or (cycle_len > 4 and cycle_len < 7)


def valence_check(aid1, aid2, edges1, edges2, new_edge, vocab, c1=0, c2=0):
    new_valence = vocab.bond_idx_to_valence(new_edge)
    if new_valence == -1:
        return False
    atom1 = vocab.idx_to_atom(aid1)
    atom2 = vocab.idx_to_atom(aid2)
    a1_val = sum(list(map(vocab.bond_idx_to_valence, edges1)))
    a2_val = sum(list(map(vocab.bond_idx_to_valence, edges2)))
    # special for S as S is likely to have either 2 or 6 valence
    if (atom1 == 'S' and a1_val == 2) or (atom2 == 'S' and a2_val == 2):
        return False
    return a1_val + new_valence + abs(c1) <= MAX_VALENCE[atom1] and \
           a2_val + new_valence + abs(c2) <= MAX_VALENCE[atom2]


def get_random_submol(mol):  # use bfs order and randomly drop 1-5 atoms
    root_idx = np.random.randint(0, mol.GetNumAtoms())
    root_atom = mol.GetAtomWithIdx(root_idx)
    order_list, idx2order = bfs_order(mol, root_atom)
    drop_num = np.random.randint(0, 5)
    rw_mol = Chem.RWMol()
    for atom in mol.GetAtoms():
        atom_sym = atom.GetSymbol()
        new_atom = Chem.Atom(atom_sym)
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        rw_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rw_mol.AddBond(begin, end, bond.GetBondType())
    if drop_num == 0:
        return rw_mol.GetMol()
    removed = [atom.GetIdx() for atom in order_list[-drop_num:]]
    removed = sorted(removed, reverse=True)
    for idx in removed:
        rw_mol.RemoveAtom(idx)
    return rw_mol.GetMol()
#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem

from modules.encoder import Encoder
from modules.common_nn import MLP
from utils.chem_utils import smiles2molecule, valence_check, cnt_atom, cycle_check, molecule2smiles, rec
from utils.chem_utils import del_N_positive
from utils.logger import print_log


class VAEPieceDecoder(nn.Module):
    '''use variational auto encoder'''
    def __init__(self, atom_embedding_dim, piece_embedding_dim, max_pos,
                 pos_embedding_dim, piece_hidden_dim,
                 node_hidden_dim, num_edge_type,
                 cond_dim, latent_dim, tokenizer, t=4):
        super(VAEPieceDecoder, self).__init__()
        self.tokenizer = tokenizer
        # piece predictor
        self.atom_embedding = nn.Embedding(tokenizer.num_atom_type(), atom_embedding_dim)
        self.piece_embedding = nn.Embedding(tokenizer.num_piece_type(), piece_embedding_dim)
        self.pos_embedding = nn.Embedding(max_pos, pos_embedding_dim)  # max position idx is 99, 0 is padding
        self.latent_to_rnn_hidden = nn.Linear(latent_dim, piece_hidden_dim)
        self.rnn = nn.GRU(piece_embedding_dim, piece_hidden_dim, batch_first=True)
        self.to_vocab = nn.Linear(piece_hidden_dim, tokenizer.num_piece_type())
        # graph embedding
        node_dim = atom_embedding_dim + piece_embedding_dim + pos_embedding_dim
        self.graph_embedding = Encoder(node_dim, num_edge_type, node_hidden_dim,
                                       dim_out=1, t=t)  # dim_out is of no use
        # edge link predictor
        mlp_in = node_hidden_dim * 2 + latent_dim
        self.edge_predictor = nn.Sequential(
            MLP(
                dim_in=mlp_in,
                dim_hidden=mlp_in // 2,
                dim_out=mlp_in,
                act_func=nn.ReLU,
                num_layers=3
            ),
            nn.Linear(mlp_in, num_edge_type)
        )
        # vae
        self.latent_dim = latent_dim
        self.W_mean = nn.Linear(cond_dim, latent_dim)
        self.W_log_var = nn.Linear(cond_dim, latent_dim)

        # loss
        self.piece_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx())
        self.edge_loss = nn.CrossEntropyLoss()

    def rsample(self, conds):
        batch_size = conds.shape[0]
        z_mean = self.W_mean(conds)
        z_log_var = -torch.abs(self.W_log_var(conds)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def embed_atom(self, atom_ids, piece_ids, pos_ids):
        atom_embed = self.atom_embedding(atom_ids)
        piece_embed = self.piece_embedding(piece_ids)
        pos_embed = self.pos_embedding(pos_ids)
        return torch.cat([atom_embed, piece_embed, pos_embed], dim=-1)

    def forward(self, x, x_pieces, x_pos, edge_index, edge_attr, pieces, conds, edge_select, golden_edge, return_accu=False):
        # x: [batch_size, N, dim_in]
        # edge_index and edge_attr do not include the edges to be predicted
        # pieces: [batch_size, seq_len] in idx form
        # conds: [batch_size, cond_dim]
        # edge_select: [batch_size, N, N] indicate which edges are to be predicted
        # golden_egdes: [E], only include edges to be predicted

        z_vecs, kl_loss = self.rsample(conds)  # [batch_size, latent_dim]
        # embed pieces
        gold_piece = pieces[:, 1:].flatten()
        pieces = self.piece_embedding(pieces) # [batch_size, seq_len, embedding_dim]
        # pieces prediction
        # [1 (num_layers), batch_size, rnn_hidden_size]
        init_hidden = self.latent_to_rnn_hidden(z_vecs).unsqueeze(0)
        # init_hidden = torch.zeros(self.rnn.num_layers, z_vecs.shape[0], self.rnn.hidden_size, device=z_vecs.device)
        pieces_in = pieces[:, :-1]
        rnn_in = pieces_in
        output, _ = self.rnn(rnn_in, init_hidden)  # [batch_size, seq_len, hidden_size]
        output = self.to_vocab(output)  # [batch_size, seq_len, num_piece_type]

        # graph embedding
        batch_size, node_num, node_dim = x.shape
        node_x = x.view(-1, node_dim)
        node_embedding, _ = self.graph_embedding.embed_node(node_x, edge_index, edge_attr)  # [batch_size * N, node_dim]
        node_embedding = node_embedding.view(batch_size, node_num, -1) # [batch_size, N, node_dim]
        # edge prediction
        # [batch_size, N, N, node_dim], the second N contains same node embedding
        src_embedding = torch.repeat_interleave(node_embedding, node_num, dim=1).view(batch_size, node_num, node_num, -1)
        src_embedding = src_embedding[edge_select] # to [E, node_dim]
        # [batch_size, N, N, node_dim], the first N contains same sets of node embedding
        dst_embedding = torch.repeat_interleave(node_embedding, node_num, dim=0).view(batch_size, node_num, node_num, -1)
        dst_embedding = dst_embedding[edge_select]
        # [batch_size, N, N, latent_dim]
        latent_repeat = torch.repeat_interleave(z_vecs, node_num ** 2, dim=0).view(batch_size, node_num, node_num, -1)
        latent_repeat = latent_repeat[edge_select]
        # [E, node_dim * 2 + latent_dim]
        # double direction is needed (a kind of data augmentation)
        edge_pred_in = torch.cat([src_embedding, dst_embedding, latent_repeat], dim=-1)
        pred_edge = self.edge_predictor(edge_pred_in) # [E, num_edge_type] 

        # loss
        pred_piece = output.view(-1, output.shape[-1])
        piece_loss = self.piece_loss(pred_piece, gold_piece)
        edge_loss = self.edge_loss(pred_edge, golden_edge)
        rec_loss = piece_loss + edge_loss
        loss_tuple = (rec_loss, piece_loss, edge_loss, kl_loss)

        # accu
        if return_accu:
            not_pad = (gold_piece != self.tokenizer.pad_idx())
            piece_accu = ((torch.argmax(pred_piece, dim=-1) == gold_piece) & not_pad).sum().item() / not_pad.sum().item()
            edge_accu = (torch.argmax(pred_edge, dim=-1) == golden_edge).sum().item() / len(golden_edge)
            return z_vecs, (loss_tuple, (piece_accu, edge_accu))

        return z_vecs, loss_tuple
    
    def inference(self, z, max_atom_num, add_edge_th, temperature):
        # z: [latent_dim]
        z = z.unsqueeze(0)  # [1, latent_dim]
        batch_size = z.shape[0] # 1
        # predict piece
        # [1, 1, embedding_dim]
        cur_piece = self.piece_embedding(torch.tensor([[self.tokenizer.end_idx()]], dtype=torch.long, device=z.device))
        hidden = self.latent_to_rnn_hidden(z).unsqueeze(0)  # unsqueeze for 1 layer of rnn
        # hidden = torch.zeros(self.rnn.num_layers, z.shape[0], self.rnn.hidden_size, device=z.device)
        piece_ids, cur_piece_id = [], None
        cur_atom_num = 0
        while cur_piece_id != self.tokenizer.end_idx():
            # rnn_in = torch.cat([cur_piece, z.unsqueeze(0)], dim=-1)
            rnn_in = cur_piece
            output, hidden = self.rnn(rnn_in, hidden)
            output = self.to_vocab(output)  # [1, 1, num_piece_type]
            output = output.squeeze()
            output[self.tokenizer.pad_idx()] = float('-inf')  # mask pad
            if len(piece_ids) == 0:
                output[self.tokenizer.end_idx()] = float('-inf')  # at least output one piece
            probs = torch.softmax(output / temperature, dim=-1)  # [num_piece_type]
            cur_piece_id = torch.multinomial(probs, num_samples=1)
            cur_piece = self.piece_embedding(cur_piece_id).unsqueeze(0)
            cur_piece_id = cur_piece_id.item()
            cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(cur_piece_id))
            piece_ids.append(cur_piece_id)
            if cur_atom_num > max_atom_num:  # the last one will still be dropped
                break
            if len(piece_ids) == self.pos_embedding.num_embeddings: # 0 is padding, the last one dropped
                break
        piece_ids = piece_ids[:-1]  # get rid of end token

        # graph embedding and start to construct a molecule
        x, edge_index, edge_attr, groups = [], [], [], []
        aid2gid = {} # map atom idx to group idx
        aid2bid = {} # map atom idx to connected block (bid)
        block_atom_cnt = []
        gen_mol = Chem.RWMol()  # generated mol
        edge_sets = []  # record each atom is connected to which kinds of bonds
        x_pieces, x_pos = [], []
        for pos, pid in enumerate(piece_ids):
            smi = self.tokenizer.idx_to_piece(pid)
            try:
                mol = smiles2molecule(smi, kekulize=True)
            except Exception:
                print(smi)
            offset = len(x)
            group, atom_num = [], mol.GetNumAtoms()
            for aid in range(atom_num):
                atom = mol.GetAtomWithIdx(aid)
                group.append(len(x))
                aid2gid[len(x)], aid2bid[len(x)] = len(groups), len(groups)
                x.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
                edge_sets.append([])
                x_pieces.append(pid)
                x_pos.append(pos + 1)  # position starts from 1

                gen_mol.AddAtom(Chem.Atom(atom.GetSymbol()))  # add atom to generated mol

            groups.append(group)
            block_atom_cnt.append(atom_num)
            for bond in mol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                attr = self.tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())
                begin, end = begin + offset, end + offset
                edge_index.append([begin, end])
                edge_index.append([end, begin])
                edge_attr.append(attr)
                edge_attr.append(attr)
                edge_sets[begin].append(attr)
                edge_sets[end].append(attr)

                gen_mol.AddBond(begin, end, bond.GetBondType())  # add bond to generated mol

        atoms, edges, edge_types = x, edge_index, edge_attr
        node_x = self.embed_atom(torch.tensor(x, dtype=torch.long, device=z.device),
                                 torch.tensor(x_pieces, dtype=torch.long, device=z.device),
                                 torch.tensor(x_pos, dtype=torch.long, device=z.device))
        if len(edge_index) == 0:
            edge_index = torch.randn(2, 0, device=z.device).long()
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=z.device).t().contiguous()
        edge_attr = F.one_hot(torch.tensor(edge_attr, dtype=torch.long, device=z.device),
                              num_classes=self.tokenizer.chem_vocab.num_bond_type())
        node_embedding, _ = self.graph_embedding.embed_node(node_x, edge_index, edge_attr)  # [n, node_embeding_dim]
        # construct edge select mat, only care about up triangle
        node_num = len(x)
        edge_select = torch.triu(torch.ones(node_num, node_num, dtype=torch.long, device=z.device))
        for group in groups:
            group_len = len(group)
            for i in range(group_len):
                for j in range(i, group_len):
                    m, n = group[i], group[j]
                    edge_select[m][n] = edge_select[n][m] = 0
        edge_select = edge_select.unsqueeze(0).bool()  # [1, node_num, node_num]
        # predict edge
        node_embedding = node_embedding.unsqueeze(0)  # [1, n, embedding_dim]
        src_embedding = torch.repeat_interleave(node_embedding, node_num, dim=1).view(batch_size, node_num, node_num, -1)
        src_embedding = src_embedding[edge_select]
        dst_embedding = torch.repeat_interleave(node_embedding, node_num, dim=0).view(batch_size, node_num, node_num, -1)
        dst_embedding = dst_embedding[edge_select]
        latent_repeat = torch.repeat_interleave(z, node_num ** 2, dim=0).view(batch_size, node_num, node_num, -1)
        latent_repeat = latent_repeat[edge_select]
        edge_pred_in = torch.cat([src_embedding, dst_embedding, latent_repeat], dim=-1)
        if edge_pred_in.shape[0]:   # maybe only one piece is generated -> no edges need to be predicted
            pred_edge = self.edge_predictor(edge_pred_in) # [E, num_edge_type]
            pred_edge = torch.softmax(pred_edge, dim=-1)
            # add edge to mol by confidence level
            pred_edge_index = torch.nonzero(edge_select.squeeze())  # [E, 2]
            none_bond = self.tokenizer.chem_vocab.bond_to_idx(None)
            confidence, edge_type = torch.max(pred_edge, dim=-1)  # [E], [E]
            possible_edge_idx = [i for i in range(len(pred_edge))
                                 if confidence[i] >= add_edge_th and edge_type[i] != none_bond]
            sorted_idx = sorted(possible_edge_idx, key=lambda i: confidence[i], reverse=True)

            # original edge generation algorithm: add the edges with higher confidence
            # sorted_idx = sorted(range(len(pred_edge)), key=lambda i: confidence[i], reverse=True)
            for i in sorted_idx:
                begin, end = pred_edge_index[i]
                begin, end = begin.item(), end.item()
                bond_type = edge_type[i]
                # the cycle check is very important (only generate cycles with 5 or 6 atoms)
                if valence_check(atoms[begin], atoms[end], edge_sets[begin],
                                 edge_sets[end], bond_type, self.tokenizer.chem_vocab) and \
                                     cycle_check(begin, end, gen_mol):
                    gen_mol.AddBond(begin, end, self.tokenizer.chem_vocab.idx_to_bond(bond_type))
                    edge_sets[begin].append(bond_type)
                    edge_sets[end].append(bond_type)
                    # update connected block
                    bid1, bid2 = aid2bid[begin], aid2bid[end]
                    if bid1 != bid2:
                        gid = aid2gid[begin]
                        for aid in aid2bid:  # redirect all atom in block1 to block2
                            if aid2bid[aid] == bid1:
                                aid2bid[aid] = bid2
                        block_atom_cnt[bid2] += block_atom_cnt[bid1]
                    
        # delete isolated parts
        # find connect block with max atom num
        bid = max(range(len(block_atom_cnt)), key=lambda i: block_atom_cnt[i])
        atoms_to_remove = sorted([aid for aid in aid2bid.keys() if aid2bid[aid] != bid], reverse=True)
        for aid in atoms_to_remove:
            gen_mol.RemoveAtom(aid)
        gen_mol = gen_mol.GetMol()
        Chem.SanitizeMol(gen_mol)
        Chem.Kekulize(gen_mol)
        return gen_mol

    def inference_constraint(self, z, max_atom_num, add_edge_th, temperature, constraint_mol):
        # z: [latent_dim]
        z = z.unsqueeze(0)  # [1, latent_dim]
        batch_size = z.shape[0] # 1
        # predict piece
        # [1, 1, embedding_dim]
        
        # constraint_mol = del_N_positive(constraint_mol)
        Chem.Kekulize(constraint_mol)
        init_pieces, init_groups = self.tokenizer.tokenize(constraint_mol, return_idx=True)
        piece_ids, cur_piece_id = list(init_pieces[1:-1]), None
        cur_atom_num = 0
        for pid in piece_ids:
            cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(pid))

        # max_atom_num = 1.2 * cur_atom_num

        cur_piece = self.piece_embedding(torch.tensor([[init_pieces[-2]]], dtype=torch.long, device=z.device))
        init_rnn_in = torch.tensor(init_pieces[:-2], device=z.device) # include start <eos>
        init_embeddings = self.piece_embedding(init_rnn_in).unsqueeze(0)

        init_pieces, init_groups = list(init_pieces[1:-1]), list(init_groups[1:-1])
        init_mol = constraint_mol
        
        hidden = self.latent_to_rnn_hidden(z).unsqueeze(0)  # unsqueeze for 1 layer of rnn
        _, hidden = self.rnn(init_embeddings, hidden)
        while cur_piece_id != self.tokenizer.end_idx():
            # rnn_in = torch.cat([cur_piece, z.unsqueeze(0)], dim=-1)
            rnn_in = cur_piece
            output, hidden = self.rnn(rnn_in, hidden)
            output = self.to_vocab(output)  # [1, 1, num_piece_type]
            output = output.squeeze()
            output[self.tokenizer.pad_idx()] = float('-inf')  # mask pad
            if len(piece_ids) == 0:
                output[self.tokenizer.end_idx()] = float('-inf')  # at least output one piece
            probs = torch.softmax(output / temperature, dim=-1)  # [num_piece_type]
            cur_piece_id = torch.multinomial(probs, num_samples=1)
            cur_piece = self.piece_embedding(cur_piece_id).unsqueeze(0)
            cur_piece_id = cur_piece_id.item()
            cur_atom_num += cnt_atom(self.tokenizer.idx_to_piece(cur_piece_id))
            piece_ids.append(cur_piece_id)
            if cur_atom_num > max_atom_num:  # the last one will still be dropped
                break
            if len(piece_ids) == self.pos_embedding.num_embeddings: # 0 is padding, the last one dropped
                break
        piece_ids = piece_ids[:-1]  # get rid of end token

        # graph embedding and start to construct a molecule
        x, edge_index, edge_attr, groups = [], [], [], []
        aid2gid = {} # map atom idx to group idx
        aid2bid = {} # map atom idx to connected block (bid)
        block_atom_cnt = []
        gen_mol = Chem.RWMol()  # generated mol
        edge_sets = []  # record each atom is connected to which kinds of bonds
        x_pieces, x_pos = [], []
        # add init mol
        inter_group_edges = []
        init_aid2aid = {}
        for pos, pid in enumerate(piece_ids):
            if pos == len(init_pieces):
                break
            group, atom_num = [], len(init_groups[pos])
            for aid in init_groups[pos]:
                atom = init_mol.GetAtomWithIdx(aid)
                group.append(len(x))
                init_aid2aid[aid] = len(x)
                aid2gid[len(x)], aid2bid[len(x)] = len(groups), len(groups)
                x.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
                edge_sets.append([])
                x_pieces.append(pid)
                x_pos.append(pos + 1)  # position starts from 1

                atom_sym = atom.GetSymbol()
                new_atom = Chem.Atom(atom_sym)
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                gen_mol.AddAtom(new_atom)  # add atom to generated mol

            groups.append(group)
            block_atom_cnt.append(atom_num)
            for bond in init_mol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if begin not in init_aid2aid or end not in init_aid2aid:
                    continue
                begin, end = init_aid2aid[begin], init_aid2aid[end]
                if begin not in group and end not in group:
                    continue
                if begin not in group or end not in group:
                    inter_group_edges.append((begin, end))
                attr = self.tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())
                edge_index.append([begin, end]) # not appear in GNN (no! should appear)
                edge_index.append([end, begin])
                edge_attr.append(attr)
                edge_attr.append(attr)
                edge_sets[begin].append(attr)
                edge_sets[end].append(attr)

                gen_mol.AddBond(begin, end, bond.GetBondType())  # add bond to generated mol

        for pos, pid in enumerate(piece_ids):
            if pos < len(init_pieces):
                continue
            smi = self.tokenizer.idx_to_piece(pid)
            try:
                mol = smiles2molecule(smi, kekulize=True)
            except Exception:
                print(smi)
            offset = len(x)
            group, atom_num = [], mol.GetNumAtoms()
            for aid in range(atom_num):
                atom = mol.GetAtomWithIdx(aid)
                group.append(len(x))
                aid2gid[len(x)], aid2bid[len(x)] = len(groups), len(groups)
                x.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
                edge_sets.append([])
                x_pieces.append(pid)
                x_pos.append(pos + 1)  # position starts from 1

                gen_mol.AddAtom(Chem.Atom(atom.GetSymbol()))  # add atom to generated mol

            groups.append(group)
            block_atom_cnt.append(atom_num)
            for bond in mol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                attr = self.tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())
                begin, end = begin + offset, end + offset
                edge_index.append([begin, end])
                edge_index.append([end, begin])
                edge_attr.append(attr)
                edge_attr.append(attr)
                edge_sets[begin].append(attr)
                edge_sets[end].append(attr)

                gen_mol.AddBond(begin, end, bond.GetBondType())  # add bond to generated mol

        atoms, edges, edge_types = x, edge_index, edge_attr
        node_x = self.embed_atom(torch.tensor(x, dtype=torch.long, device=z.device),
                                 torch.tensor(x_pieces, dtype=torch.long, device=z.device),
                                 torch.tensor(x_pos, dtype=torch.long, device=z.device))
        if len(edge_index) == 0:
            edge_index = torch.randn(2, 0, device=z.device).long()
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=z.device).t().contiguous()
        edge_attr = F.one_hot(torch.tensor(edge_attr, dtype=torch.long, device=z.device),
                              num_classes=self.tokenizer.chem_vocab.num_bond_type())
        node_embedding, _ = self.graph_embedding.embed_node(node_x, edge_index, edge_attr)  # [n, node_embeding_dim]
        # construct edge select mat, only care about up triangle
        node_num = len(x)
        edge_select = torch.triu(torch.ones(node_num, node_num, dtype=torch.long, device=z.device))
        for group in groups:
            group_len = len(group)
            for i in range(group_len):
                for j in range(i, group_len):
                    m, n = group[i], group[j]
                    edge_select[m][n] = edge_select[n][m] = 0
        # inter-group (added by init mol)
        for begin, end in inter_group_edges:
            edge_select[begin][end] = edge_select[end][begin] = 0
            bid1, bid2 = aid2bid[begin], aid2bid[end]  # revise connect block
            if bid1 != bid2:
                gid = aid2gid[begin]
                for aid in aid2bid:  # redirect all atom in block1 to block2
                    if aid2bid[aid] == bid1:
                        aid2bid[aid] = bid2
                block_atom_cnt[bid2] += block_atom_cnt[bid1]
        edge_select = edge_select.unsqueeze(0).bool()  # [1, node_num, node_num]
        # predict edge
        node_embedding = node_embedding.unsqueeze(0)  # [1, n, embedding_dim]
        src_embedding = torch.repeat_interleave(node_embedding, node_num, dim=1).view(batch_size, node_num, node_num, -1)
        src_embedding = src_embedding[edge_select]
        dst_embedding = torch.repeat_interleave(node_embedding, node_num, dim=0).view(batch_size, node_num, node_num, -1)
        dst_embedding = dst_embedding[edge_select]
        latent_repeat = torch.repeat_interleave(z, node_num ** 2, dim=0).view(batch_size, node_num, node_num, -1)
        latent_repeat = latent_repeat[edge_select]
        edge_pred_in = torch.cat([src_embedding, dst_embedding, latent_repeat], dim=-1)
        if edge_pred_in.shape[0]:   # maybe only one piece is generated -> no edges need to be predicted
            pred_edge = self.edge_predictor(edge_pred_in) # [E, num_edge_type]
            pred_edge = torch.softmax(pred_edge, dim=-1)
            # add edge to mol by confidence level
            pred_edge_index = torch.nonzero(edge_select.squeeze())  # [E, 2]
            none_bond = self.tokenizer.chem_vocab.bond_to_idx(None)
            confidence, edge_type = torch.max(pred_edge, dim=-1)  # [E], [E]
            possible_edge_idx = [i for i in range(len(pred_edge))
                                 if confidence[i] >= add_edge_th and edge_type[i] != none_bond]
            sorted_idx = sorted(possible_edge_idx, key=lambda i: confidence[i], reverse=True)

            # original edge generation algorithm: add the edges with higher confidence
            for i in sorted_idx:
                begin, end = pred_edge_index[i]
                begin, end = begin.item(), end.item()
                bond_type = edge_type[i]
                # the cycle check is very important (only generate cycles with 5 or 6 atoms)
                if valence_check(atoms[begin], atoms[end], edge_sets[begin],
                                 edge_sets[end], bond_type, self.tokenizer.chem_vocab,
                                 gen_mol.GetAtomWithIdx(begin).GetFormalCharge(),
                                 gen_mol.GetAtomWithIdx(end).GetFormalCharge()) and \
                                     cycle_check(begin, end, gen_mol):
                    gen_mol.AddBond(begin, end, self.tokenizer.chem_vocab.idx_to_bond(bond_type))
                    edge_sets[begin].append(bond_type)
                    edge_sets[end].append(bond_type)
                    # update connected block
                    bid1, bid2 = aid2bid[begin], aid2bid[end]
                    if bid1 != bid2:
                        gid = aid2gid[begin]
                        for aid in aid2bid:  # redirect all atom in block1 to block2
                            if aid2bid[aid] == bid1:
                                aid2bid[aid] = bid2
                        block_atom_cnt[bid2] += block_atom_cnt[bid1]
                    
        # delete isolated parts
        # find connect block with max atom num and must contain the original molecule
        sorted_bids = sorted(range(len(block_atom_cnt)), key=lambda i: block_atom_cnt[i], reverse=True)
        for i in sorted_bids:
            if i == aid2bid[0]:  # root must be in
                bid = i
                break
        # bid = max(range(len(block_atom_cnt)), key=lambda i: block_atom_cnt[i])
        atoms_to_remove = sorted([aid for aid in aid2bid.keys() if aid2bid[aid] != bid], reverse=True)
        for aid in atoms_to_remove:
            gen_mol.RemoveAtom(aid)
        gen_mol = gen_mol.GetMol()
        smis = Chem.MolToSmiles(gen_mol)
        gen_mol = Chem.MolFromSmiles(smis)
        return gen_mol
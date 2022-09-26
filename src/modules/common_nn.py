#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


# model utils
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, act_func, num_layers):
        super(MLP, self).__init__()
        assert num_layers > 0
        if num_layers == 1:
            self.seq = nn.Linear(dim_in, dim_out)
        else:
            seq = [nn.Linear(dim_in, dim_hidden), act_func()]
            for i in range(num_layers - 2):
                seq.append(nn.Linear(dim_hidden, dim_hidden))
                seq.append(act_func())
            seq.append(nn.Linear(dim_hidden, dim_out))
            self.seq = nn.Sequential(*seq)
    
    def forward(self, x):
        return self.seq(x)
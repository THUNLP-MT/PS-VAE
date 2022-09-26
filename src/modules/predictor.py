#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn


class Predictor(nn.Module):
    """given graph-level embedding, predict all properties"""
    def __init__(self, dim_feature, dim_hidden, num_property):
        """2-layer MLP"""
        super(Predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_feature, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU()
        )
        self.output = nn.Linear(dim_hidden, num_property)
    
    def forward(self, x):
        hidden = self.mlp(x)
        return self.output(hidden)
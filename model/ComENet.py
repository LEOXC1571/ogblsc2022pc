# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: ComENet.py
# @Author: Leo Xu
# @Date: 2022/9/9 14:25
# @Email: leoxc1571@163.com
# Description:

import sys
from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn.acts import swish
from torch_geometric.nn import inits

from .comenet.utils import dist_calc
from .comenet.features import angle_emb, torsion_emb

from torch_scatter import scatter, scatter_min

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt
import numpy as np


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='glorot',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'glorot_orthogonal':
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer == 'zeros':
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLayerLinear(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False,
    ):
        super(TwoLayerLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EdgeGraphConv(GraphConv):
    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


class InterBock(nn.Module):
    def __init__(self,
                 hidden_channels,
                 middle_channels,
                 num_radial,
                 num_spherical,
                 num_layers,
                 output_channels,
                 act=swish):
        super(InterBock, self).__init__()
        self.act = act

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.linear1 = Linear(hidden_channels, hidden_channels)
        self.linear2 = Linear(hidden_channels, hidden_channels)
        self.linear_cat = Linear(2 * hidden_channels, hidden_channels)
        self.norm = GraphNorm(hidden_channels)

        self.lin_feat1 = TwoLayerLinear(num_radial * num_spherical ** 2, middle_channels, hidden_channels)
        self.lin_feat2 = TwoLayerLinear(num_radial * num_spherical, middle_channels, hidden_channels)

        self.linear = Linear(hidden_channels, hidden_channels)
        self.linears = nn.ModuleList()
        for _ in range(num_layers):
            self.linears.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)
        self.reset_params()

    def reset_params(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.norm.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear.reset_parameters()
        self.linear_cat. reset_parameters()
        self.lin_feat1.reset_parameters()
        self.lin_feat2.reset_parameters()
        for lin in self.linears:
            lin.reset_parameters()
        self.final.reset_parameters()

    def forward(self, x, feature1, feature2, edge_index, batch):
        x = self.act(self.linear(x))
        # print('init', torch.cuda.memory_allocated())

        feature1 = self.lin_feat1(feature1)
        # print('lin_feat1', torch.cuda.memory_allocated())
        h1 = self.conv1(x, edge_index, feature1)
        # print('conv1', torch.cuda.memory_allocated())
        h1 = self.linear1(h1)
        # print('linear1', torch.cuda.memory_allocated())
        h1 = self.act(h1)

        feature2 = self.lin_feat2(feature2)
        # print('lin_feat2', torch.cuda.memory_allocated())
        h2 = self.conv2(x, edge_index, feature2)
        # print('conv2', torch.cuda.memory_allocated())
        h2 = self.linear2(h2)
        # print('linear2', torch.cuda.memory_allocated())
        h2 = self.act(h2)

        h = self.linear_cat(torch.cat([h1, h2], 1))
        # print('linear_cat', torch.cuda.memory_allocated())

        h = h + x
        for lin in self.linears:
            h = self.act(lin(h)) + h
        h = self.norm(h, batch)
        # print('norm', torch.cuda.memory_allocated())
        h = self.final(h)
        # print('final', torch.cuda.memory_allocated())
        return h


class ComENet(nn.Module):
    def __init__(self,
                 cutoff=8.0,
                 num_layers=4,
                 hidden_channels=256,
                 middle_channels=64,
                 out_channels=1,
                 num_radial=3,
                 num_spherical=2,
                 num_output_layers=3):
        super(ComENet, self).__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.out_channels = out_channels

        self.act = swish

        self.feat1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feat2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.x_emb = nn.Embedding(178, hidden_channels)

        self.inter_block = nn.ModuleList(
            [
                InterBock(
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    self.act,
                )
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.ModuleList()
        for _ in range(num_output_layers):
            self.linear.append(nn.Linear(hidden_channels, hidden_channels))
        self.out = nn.Linear(hidden_channels, out_channels)

    def reset_params(self):
        self.x_emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        for linear in self.linear:
            linear.reset_parameters()
        self.out.reset_parameters()

    def forward(self, *argv):
        # print('data loading', torch.cuda.memory_allocated())
        data = argv[0]
        batch = data.batch
        x = data.x
        num_nodes = x.size(0)
        pos = data.pos
        edge_idx = data.edge_index
        i, j = edge_idx[0], edge_idx[1]

        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        x = self.act(self.x_emb(x).sum(1))
        # print('emb',torch.cuda.memory_allocated())

        dist, theta, phi, tau = dist_calc(self.cutoff, vecs, dist, i, j, num_nodes)
        # print('dist_calc', torch.cuda.memory_allocated())

        feat1 = self.feat1(dist, theta, phi)
        feat2 = self.feat2(dist, tau)
        # print('feat', torch.cuda.memory_allocated())

        for inter in self.inter_block:
            x = inter(x, feat1, feat2, edge_idx, batch)
            # print('inter', torch.cuda.memory_allocated())

        for lin in self.linear:
            x = self.act(lin(x))
        x = self.out(x)

        energy = scatter(x, batch, dim=0)

        return energy

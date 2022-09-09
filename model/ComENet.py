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

from .comenet.features import angle_emb, torsion_emb

from torch_scatter import scatter, scatter_min


import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt


def dist_calc(cutoff, vecs, dist, i, j, num_nodes):
    _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
    argmin0[argmin0 >= len(i)] = 0
    n0 = j[argmin0]
    add = torch.zeros_like(dist).to(dist.device)
    add[argmin0] = cutoff
    dist1 = dist + add

    _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
    argmin1[argmin1 >= len(i)] = 0
    n1 = j[argmin1]
    # --------------------------------------------------------

    _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
    argmin0_j[argmin0_j >= len(j)] = 0
    n0_j = i[argmin0_j]

    add_j = torch.zeros_like(dist).to(dist.device)
    add_j[argmin0_j] = cutoff
    dist1_j = dist + add_j

    # i[argmin] = range(0, num_nodes)
    _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
    argmin1_j[argmin1_j >= len(j)] = 0
    n1_j = i[argmin1_j]

    # ----------------------------------------------------------

    # n0, n1 for i
    n0 = n0[i]
    n1 = n1[i]

    # n0, n1 for j
    n0_j = n0_j[j]
    n1_j = n1_j[j]

    # tau: (iref, i, j, jref)
    # when compute tau, do not use n0, n0_j as ref for i and j,
    # because if n0 = j, or n0_j = i, the computed tau is zero
    # so if n0 = j, we choose iref = n1
    # if n0_j = i, we choose jref = n1_j
    mask_iref = n0 == j
    iref = torch.clone(n0)
    iref[mask_iref] = n1[mask_iref]
    idx_iref = argmin0[i]
    idx_iref[mask_iref] = argmin1[i][mask_iref]

    mask_jref = n0_j == i
    jref = torch.clone(n0_j)
    jref[mask_jref] = n1_j[mask_jref]
    idx_jref = argmin0_j[j]
    idx_jref[mask_jref] = argmin1_j[j][mask_jref]

    pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
        vecs,
        vecs[argmin0][i],
        vecs[argmin1][i],
        vecs[idx_iref],
        vecs[idx_jref]
    )

    # Calculate angles.
    a = ((-pos_ji) * pos_in0).sum(dim=-1)
    b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
    theta = torch.atan2(b, a)
    theta[theta < 0] = theta[theta < 0] + math.pi

    # Calculate torsions.
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(-pos_ji, pos_in0)
    plane2 = torch.cross(-pos_ji, pos_in1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    phi = torch.atan2(b, a)
    phi[phi < 0] = phi[phi < 0] + math.pi

    # Calculate right torsions.
    plane1 = torch.cross(pos_ji, pos_jref_j)
    plane2 = torch.cross(pos_ji, pos_iref)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    tau = torch.atan2(b, a)
    tau[tau < 0] = tau[tau < 0] + math.pi
    return dist, theta, phi, tau

class ComENet(nn.Module):
    def __init__(self,
                 device,
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
            []
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
        data = argv[0]
        batch = data.batch
        x = data.x
        num_nodes = x.size(0)
        pos = data.pos
        edge_idx = data.edge_index
        i, j = edge_idx[0], edge_idx[1]

        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        x = self.act(self.x_emb(x))

        dist, theta, phi, tau = dist_calc(self.cutoff, vecs, dist, i, j, num_nodes)

        feat1 = self.feat1(dist, theta, phi)
        feat2 = self.feat2(dist, tau)

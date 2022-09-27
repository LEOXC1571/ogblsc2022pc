# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: add_conf.py
# @Author: Leo Xu
# @Date: 2022/9/26 16:10
# @Email: leoxc1571@163.com
# Description:

import torch
from tqdm import tqdm


def split_tensor(x: torch.Tensor, batch: torch.Tensor):
    length = torch.unique(batch, return_counts=True)[1].cpu()
    dup_count = tuple(length.numpy())
    split = x.split(dup_count, dim=0)
    return list(split), length


def add_conf(model, loader, device):
    model.eval()
    mol_pred = []
    n_nodes = []
    print('Start generating conformation for molecules in valid and test set...')
    for batch in tqdm(loader, desc='Iteration'):
        temp = []
        batch.to(device)
        with torch.no_grad():
            pred, _ = model(batch)
        split_pred, length = split_tensor(pred[-1], batch.batch)
        mol_pred.extend(split_pred)
        n_nodes.append(length)
    torch.cat(n_nodes)
    return mol_pred, n_nodes

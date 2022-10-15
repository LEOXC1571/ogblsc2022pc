# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: add_conf.py
# @Author: Leo Xu
# @Date: 2022/9/26 16:10
# @Email: leoxc1571@163.com
# Description:

import torch
import numpy as np
from tqdm import tqdm


def split_tensor(x: torch.Tensor, batch: torch.Tensor):
    length = torch.unique(batch, return_counts=True)[1].cpu()
    dup_count = tuple(length.numpy())
    split = x.cpu().split(dup_count, dim=0)
    split = np.array(split)
    return split


def add_conf(model, loader, device):
    model.eval()
    temp_mol = []
    node_batch = []
    print('Start generating conformation for molecules in valid and test set...')
    for step, batch in enumerate(tqdm(loader, desc='Iteration')):
        batch.to(device)
        with torch.no_grad():
            pred, _ = model(batch)
        temp_mol.append(pred[-1])
        node_batch.append(batch.batch + step * 2048)
    mol_pred = split_tensor(torch.cat(temp_mol), torch.cat(node_batch))
    idx = list(range(3746620))
    return mol_pred, idx

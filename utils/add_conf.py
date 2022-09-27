# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: add_conf.py
# @Author: Leo Xu
# @Date: 2022/9/26 16:10
# @Email: leoxc1571@163.com
# Description:

import torch
from tqdm import tqdm
import random
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops

def add_conf(model, loader, device):
    model.eval()
    mol_pred = []
    print('Start generating conformation for molecules in valid and test set...')
    for batch in tqdm(loader, desc='Iteration'):
        batch.to(device)
        with torch.no_grad():
            pred, _ = model(batch)
        mol_pred.append(pred[-1])
    return torch.cat(mol_pred)

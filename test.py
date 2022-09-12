# -*- coding: utf-8 -*-
# @Filename: test
# @Date: 2022-06-23 14:19
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os.path as osp
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.smiles2graph import smilestograph


# m = Chem.MolFromSmiles('C1OC1')
# print('\t'.join(['id', 'num', 'symbol', 'degree', 'charge', 'hybrid']))
# for atom in m.GetAtoms():
#     print(atom.GetIdx(), end='\t')
#     print(atom.GetAtomicNum(), end='\t')
#     print(atom.GetSymbol(), end='\t')
#     print(atom.GetDegree(), end='\t')
#     print(atom.GetFormalCharge(), end='\t')
#     print(atom.GetHybridization())
# #
# raw_dir = '../../../../data/xc/molecule_datasets/pcqm4m-v2/pcqm4m-v2/raw'
#
# data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
# smiles_list = data_df['smiles']
# homolumogap_list = data_df['homolumogap']
#
# print('Converting SMILES strings into graphs...')
# data_list = []
# for i in tqdm(range(len(smiles_list))):
#     smiles = smiles_list[i]
#     homolumogap = homolumogap_list[i]
#     rdkit_mol = AllChem.MolFromSmiles(smiles)
#     data = smilestograph(rdkit_mol)
#     assert (len(data['edge_attr']) == data['edge_index'].shape[1])
#     assert (len(data['x']) == data.num_nodes)
#
#     data.y = torch.Tensor([homolumogap])
#     data_list.append(data)

# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.distributed as dist
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
#
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler
#
#
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6'
# print(torch.cuda.device_count())

# from torch_cluster import radius_graph, radius
# # x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
# # batch = torch.tensor([0, 0, 0, 0])
# # edge_index = radius_graph(x, r=1, batch=batch, loop=False)
# #
# # print(edge_index)
# x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
# batch_x = torch.tensor([0, 0, 0, 0])
# y = torch.Tensor([[-1, 0], [1, 0]])
# batch_y = torch.tensor([0, 0])
# assign_index = radius(x, y, 1.5, batch_x, batch_y)
#
# print(assign_index)
def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
    """the atoms of mol will be changed in some cases."""
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        energy = res[index][1]
        conf = new_mol.GetConformer(id=int(index))
    except:
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        energy = 0
        conf = new_mol.GetConformer()

    atom_poses = get_atom_poses(new_mol, conf)
    if return_energy:
        return new_mol, atom_poses, energy
    else:
        return new_mol, atom_poses

def get_atom_poses(mol, conf):
    atom_poses = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    return atom_poses

def get_2d_atom_poses(mol):
    """get 2d atom poses"""
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    atom_poses = get_atom_poses(mol, conf)
    return atom_poses

smiles = "OCc1ccccc1CN"
# smiles = r"[H]/[NH+]=C(\N)C1=CC(=O)/C(=C\C=c2ccc(=C(N)[NH3+])cc2)C=C1"
mol = AllChem.MolFromSmiles(smiles)
# template = Chem.MolFromSmiles('c1nccc2n1ccc2')
# AllChem.Compute2DCoords(template)
m2 = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)
m3 = Chem.RemoveHs(m2)
atom_list = []
for atom in m3.GetAtoms():
    print(atom_list.append(atom.GetAtomicNum()))
print(len(smiles))
print(mol)
# data = mol_to_geognn_graph_data_MMFF3d(mol)
# data = mol_to_trans_data_w_meta_path(mol)
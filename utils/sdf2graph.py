# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: sdf2graph.py
# @Author: Leo Xu
# @Date: 2022/9/4 14:51
# @Email: leoxc1571@163.com
# Description:

# -*- coding: utf-8 -*-
# @Filename: smiles2graph
# @Date: 2022-08-03 10:17
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import numpy as np
import pandas as pd
import decimal
from decimal import Decimal
from rdkit import Chem
from rdkit.Chem import AllChem

import torch

from torch_geometric.data import Data

decimal.getcontext().rounding = "ROUND_HALF_UP"
allowable_features = {
    'atomic_num': list(range(1, 122)),  # 119for mask, 120 for collection
    'formal_charge': ['unk', -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'chirality': ['unk',
                  Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                  Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                  Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                  Chem.rdchem.ChiralType.CHI_OTHER
                  ],
    'hybridization': ['unk',
                      Chem.rdchem.HybridizationType.S,
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
                      ],
    'numH': ['unk', 0, 1, 2, 3, 4, 5, 6, 7, 8],
    'implicit_valence': ['unk', 0, 1, 2, 3, 4, 5, 6],
    'degree': ['unk', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'isaromatic': [False, True],

    'bond_type': ['unk',
                  Chem.rdchem.BondType.SINGLE,
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC
                  ],
    'bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'bond_isconjugated': [False, True],
    'bond_inring': [False, True],
    'bond_stereo': ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]
}

atom_dic = [len(allowable_features['atomic_num']), len(allowable_features['formal_charge']),
            len(allowable_features['chirality']),
            len(allowable_features['hybridization']), len(allowable_features['numH']),
            len(allowable_features['implicit_valence']),
            len(allowable_features['degree']), len(allowable_features['isaromatic'])]
bond_dic = [len(allowable_features['bond_type']), len(allowable_features['bond_dirs']),
            len(allowable_features['bond_isconjugated']),
            len(allowable_features['bond_inring']), len(allowable_features['bond_stereo'])]

atom_cumsum = np.cumsum(atom_dic)
bond_cumsum = np.cumsum(bond_dic)


def get_atom_poses(mol, conf):
    atom_poses = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    return atom_poses


def sdf2graph(rdkit_mol, sdf_mol):
    assert rdkit_mol is not None
    atom_features_list = []
    mol = sdf_mol if sdf_mol is not None else rdkit_mol

    for atom in mol.GetAtoms():
        atom_feature = \
            [allowable_features['atomic_num'].index(atom.GetAtomicNum())] + \
            [allowable_features['formal_charge'].index(atom.GetFormalCharge())] + \
            [allowable_features['chirality'].index(atom.GetChiralTag())] + \
            [allowable_features['hybridization'].index(atom.GetHybridization())] + \
            [allowable_features['numH'].index(atom.GetTotalNumHs())] + \
            [allowable_features['implicit_valence'].index(atom.GetImplicitValence())] + \
            [allowable_features['degree'].index(atom.GetDegree())] + \
            [allowable_features['isaromatic'].index(atom.GetIsAromatic())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 5
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = \
                [allowable_features['bond_type'].index(bond.GetBondType())] + \
                [allowable_features['bond_dirs'].index(bond.GetBondDir() if bond.GetBondDir() in allowable_features[
                    'bond_dirs'] else Chem.rdchem.BondDir.NONE) + bond_cumsum[0]] + \
                [allowable_features['bond_isconjugated'].index(bond.GetIsConjugated()) + bond_cumsum[1]] + \
                [allowable_features['bond_inring'].index(bond.IsInRing()) + bond_cumsum[2]] + \
                [allowable_features['bond_stereo'].index(str(bond.GetStereo())) + bond_cumsum[3]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        # print('mol has no bonds')
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    if sdf_mol is not None:
        position = sdf_mol.GetConformer().GetPositions()
    else:
        try:
            temp_mol = Chem.AddHs(rdkit_mol)
            # AllChem.EmbedMultipleConfs(temp_mol, numConfs=40)
            AllChem.EmbedMolecule(temp_mol, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(temp_mol)
            rdkit_mol = Chem.RemoveHs(temp_mol)
            position = rdkit_mol.GetConformer().GetPositions()
        except:
            AllChem.Compute2DCoords(rdkit_mol)
            position = rdkit_mol.GetConformer().GetPositions()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=position)

    return data


# def sdf2graph(rdkit_mol, sdf_mol):
#     assert rdkit_mol != None
#     atom_features_list = []
#     if sdf_mol is not None:
#         position = sdf_mol.GetConformer().GetPositions()
#         mol = sdf_mol
#     else:
#         position = np.zeros([len(rdkit_mol.GetAtoms()), 3], 'float32')
#         mol = rdkit_mol
#     for atom in mol.GetAtoms():
#         atom_feature = \
#             [allowable_features['atomic_num'].index(atom.GetAtomicNum())] + \
#             [allowable_features['formal_charge'].index(atom.GetFormalCharge())] + \
#             [allowable_features['chirality'].index(atom.GetChiralTag())] + \
#             [allowable_features['hybridization'].index(atom.GetHybridization())] + \
#             [allowable_features['numH'].index(atom.GetTotalNumHs())] + \
#             [allowable_features['implicit_valence'].index(atom.GetImplicitValence())] + \
#             [allowable_features['degree'].index(atom.GetDegree())] + \
#             [allowable_features['isaromatic'].index(atom.GetIsAromatic())]
#         atom_features_list.append(atom_feature)
#     x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
#     # bonds
#     num_bond_features = 6
#     if len(sdf_mol.GetBonds()) > 0:
#         edges_list = []
#         edge_features_list = []
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             dist = np.linalg.norm(position[i] - position[j])
#             dist = float(Decimal(str(dist)).quantize(Decimal('0.0')))
#             edge_feature = \
#                 [allowable_features['bond_type'].index(bond.GetBondType())] + \
#                 [allowable_features['bond_dirs'].index(bond.GetBondDir() if bond.GetBondDir() in allowable_features[
#                     'bond_dirs'] else Chem.rdchem.BondDir.NONE) + bond_cumsum[0]] + \
#                 [allowable_features['bond_isconjugated'].index(bond.GetIsConjugated()) + bond_cumsum[1]] + \
#                 [allowable_features['bond_inring'].index(bond.IsInRing()) + bond_cumsum[2]] + \
#                 [allowable_features['bond_stereo'].index(str(bond.GetStereo())) + bond_cumsum[3]] + \
#                 [bins.index(dist if dist in bins else (((dist * 2) // 1) / 2)) + bond_cumsum[4]]
#             edges_list.append((i, j))
#             edge_features_list.append(edge_feature)
#             edges_list.append((j, i))
#             edge_features_list.append(edge_feature)
#         edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
#         edge_attr = torch.tensor(np.array(edge_features_list),
#                                  dtype=torch.long)
#     else:  # mol has no bonds
#         # print('mol has no bonds')
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#         edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
#
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=position)
#
#     return data

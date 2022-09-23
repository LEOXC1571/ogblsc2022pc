# -*- coding: utf-8 -*-
# @Filename: test.py
# @Date: 2022/9/8 22:41
# @Author: LEO XU
# @Email: leoxc1571@163.com

import sys

# string, length = 'aaab', 4
#
# stack = []
# for ch in string:
#     if ch == ')':
#         if not stack or stack[-1] != '(':
#             print(False)
#         stack.pop()
#     elif ch == '(':
#         stack.append(ch)
#     else:
#         continue
# print(not stack)

import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.sdf2graph import sdf2graph



sdf_data = Chem.SDMolSupplier(os.path.join('../../../../data/xc/molecule_datasets/pcqm4m-v2',
                                           'raw/pcqm4m-v2-train.sdf'),
                              sanitize=True, removeHs=True, strictParsing=True)
csv_data = pd.read_csv(osp.join('../../../../data/xc/molecule_datasets/pcqm4m-v2', 'raw/data.csv.gz'))
smiles_list = csv_data['smiles']
homolumogap_list = csv_data['homolumogap']
idx_list = [3682021, 3682024, 3682073, 3682291, 3682311, 3682627, 3682637, 3682674]

for i in tqdm(idx_list):
    smiles = smiles_list[i]
    homolumogap = homolumogap_list[i]
    rdkit_mol = AllChem.MolFromSmiles(smiles)
    if i < len(sdf_data):
        sdf_mol = sdf_data[i]
    else:
        sdf_mol = None
    data = sdf2graph(rdkit_mol, sdf_mol)
# -*- coding: utf-8 -*-
# @Filename: test.py
# @Date: 2022/9/8 22:41
# @Author: LEO XU
# @Email: leoxc1571@163.com

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import os.path as osp
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.sdf2graph import sdf2graph
from dataset.pcqm4mv2_3d import PCQM4Mv2Dataset_3D



# sdf_data = Chem.SDMolSupplier(os.path.join('../../../../data/xc/molecule_datasets/pcqm4m-v2',
#                                            'raw/pcqm4m-v2-train.sdf'),
#                               sanitize=True, removeHs=True, strictParsing=True)
# csv_data = pd.read_csv(osp.join('../../../../data/xc/molecule_datasets/pcqm4m-v2', 'raw/data.csv.gz'))
# smiles_list = csv_data['smiles']
# homolumogap_list = csv_data['homolumogap']
# idx_list = [3682021, 3682024, 3682073, 3682291, 3682311, 3682627, 3682637, 3682674]
#
# for i in tqdm(idx_list):
#     smiles = smiles_list[i]
#     homolumogap = homolumogap_list[i]
#     rdkit_mol = AllChem.MolFromSmiles(smiles)
#     if i < len(sdf_data):
#         sdf_mol = sdf_data[i]
#     else:
#         sdf_mol = None
#     data = sdf2graph(rdkit_mol, sdf_mol)



def eval(model, device, loader, evaluator, batch_size, pos=None):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            if pos is not None:
                batch_pos = torch.cat(list(pos[step*batch_size: (step*batch_size+batch.batch[-1])+1])).to(device)
                batch.pos = batch_pos
            pred = model(batch).view(-1)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

device = torch.device("cuda:0")
dataset = PCQM4Mv2Dataset_3D(root='../../../../data/xc/molecule_datasets/pcqm4m-v2',
                             sdf2graph=sdf2graph)
split_idx = dataset.get_idx_split()

from model import ComENet
num_tasks = 1
model = ComENet(cutoff=8.0,
                num_layers=4,
                hidden_channels=256,
                middle_channels=64,
                out_channels=1,
                num_radial=3,
                num_spherical=2,
                num_output_layers=3)

pos_ckpt = torch.load('ckpt/pos_ckpt.pt')
valid_pos = pos_ckpt['pos'][split_idx['valid'] - 3378606]
valid_mae = eval(model, device, valid_loader, evaluator, 4096, pos=valid_pos)

# -*- coding: utf-8 -*-
# @Filename: 3d_pcqm4v2.py
# @Date: 2022-09-02 14:12
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os
import os.path as osp
import random
import shutil

from rdkit import Chem
from rdkit.Chem import AllChem
from ogb.utils.torch_util import replace_numpy_with_torchtensor

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from utils.smiles2graph import smilestograph
from utils.sdf2graph import sdf2graph


class PCQM4Mv2Dataset_3D(InMemoryDataset):
    def __init__(self, root='../../../../../data/xc/molecule_datasets',
                 sdf2graph=sdf2graph,
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.smiles2graph = sdf2graph
        self.folder = os.path.join(root, 'pcqm4m-v2')
        self.transform, self.pre_transform = transform, pre_transform
        super(PCQM4Mv2Dataset_3D, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # data = self.get(idx)
            data = self.get(self.indices()[idx])  # attention: must self.indices()[idx]
            idx_rd = random.randrange(self.__len__())
            data_random = self.get(self.indices()[idx_rd])  # self.get(idx_rd)
            data = data if self.transform is None else self.transform(data, data_random)
            return data
        else:
            # for split datasets
            return self.index_select(idx)

    @property
    def raw_file_names(self):
        return 'pcqm4m-v2-subset.sdf'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed_3d.pt'

    def download(self):
        print('Please checkout if the raw file is downloaded!')


    def process(self):
        raw_data = Chem.SDMolSupplier(os.path.join(self.folder, self.raw_file_names),
                                      sanitize=True, removeHs=True, strictParsing=True)
        data_list = []
        for mol in tqdm(raw_data):
            data = sdf2graph(mol)

        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            # data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            rdkit_mol = AllChem.MolFromSmiles(smiles)
            data = smilestograph(rdkit_mol)

            assert (len(data['edge_attr']) == data['edge_index'].shape[1])
            assert (len(data['x']) == data.num_nodes)

            # data.__num_nodes__ = int(graph.num_nodes)
            # data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            # data.edge_attr = torch.from_numpy(graph['edge_attr']).to(torch.int64)
            # data.x = torch.from_numpy(graph['x']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


if __name__ == '__main__':
    dataset = PCQM4Mv2Dataset_3D()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())

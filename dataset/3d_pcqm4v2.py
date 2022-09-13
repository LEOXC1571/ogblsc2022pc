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

# from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.lsc import PygPCQM4Mv2Dataset

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from utils.smiles2graph import smilestograph

# class PCQM4Mv2Dataset_3D(PygPCQM4Mv2Dataset):
#     def __init_(self, root='dataset',
#                  smiles2graph=smilestograph,
#                  transform=None,
#                  pre_transform=None):
#         self.root = root
#
#
# class PCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
#     def __init__(self, root='dataset',
#                  smiles2graph=smilestograph,
#                  transform=None,
#                  pre_transform=None):
#         super(PCQM4Mv2Dataset, self).__init__(root, smiles2graph, transform, pre_transform)
#
#         self.smiles2graph = smiles2graph
#         self.transform, self.pre_transform = transform, pre_transform
#         # self.folder = osp.join(root, 'pcqm4m-v2')   # self.folder从PygPCQM4Mv2Dataset中继承，不需要再join
#         self.version = 1
#         self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
#
#         # check version and update if necessary
#         if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
#             print('PCQM4Mv2 dataset has been updated.')
#             if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
#                 shutil.rmtree(self.folder)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     def __getitem__(self, idx):
#         r"""Gets the data object at index :obj:`idx` and transforms it (in case
#         a :obj:`self.transform` is given).
#         Returns a data object, if :obj:`idx` is a scalar, and a new dataset in
#         case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a LongTensor
#         or a ByteTensor."""
#         if isinstance(idx, int):
#             # data = self.get(idx)
#             data = self.get(self.indices()[idx])  # attention: must self.indices()[idx]
#             idx_rd = random.randrange(self.__len__())
#             data_random = self.get(self.indices()[idx_rd])  # self.get(idx_rd)
#             data = data if self.transform is None else self.transform(data, data_random)
#             return data
#         else:
#             # for split datasets
#             return self.index_select(idx)
#
#     @property
#     def raw_file_names(self):
#         return 'data.csv.gz'
#
#     @property
#     def processed_file_names(self):
#         return 'geometric_data_processed.pt'
#
#     def download(self):
#         if decide_download(self.url):
#             path = download_url(self.url, self.original_root)
#             extract_zip(path, self.original_root)
#             os.unlink(path)
#         else:
#             print('Stop download.')
#             exit(-1)
#
#     def process(self):
#         data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
#         smiles_list = data_df['smiles']
#         homolumogap_list = data_df['homolumogap']
#
#         print('Converting SMILES strings into graphs...')
#         data_list = []
#         for i in tqdm(range(len(smiles_list))):
#             # data = Data()
#
#             smiles = smiles_list[i]
#             homolumogap = homolumogap_list[i]
#             rdkit_mol = AllChem.MolFromSmiles(smiles)
#             data = smilestograph(rdkit_mol)
#
#             assert (len(data['edge_attr']) == data['edge_index'].shape[1])
#             assert (len(data['x']) == data.num_nodes)
#
#             # data.__num_nodes__ = int(graph.num_nodes)
#             # data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
#             # data.edge_attr = torch.from_numpy(graph['edge_attr']).to(torch.int64)
#             # data.x = torch.from_numpy(graph['x']).to(torch.int64)
#             data.y = torch.Tensor([homolumogap])
#
#             data_list.append(data)
#
#         # double-check prediction target
#         split_dict = self.get_idx_split()
#         assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
#         assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
#         assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
#         assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))
#
#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]
#
#         data, slices = self.collate(data_list)
#
#         print('Saving...')
#         torch.save((data, slices), self.processed_paths[0])
#
#     def get_idx_split(self):
#         split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
#         return split_dict
#
#
# if __name__ == '__main__':
#     dataset = PygPCQM4Mv2Dataset()
#     print(dataset)
#     print(dataset.data.edge_index)
#     print(dataset.data.edge_index.shape)
#     print(dataset.data.x.shape)
#     print(dataset[100])
#     print(dataset[100].y)
#     print(dataset.get_idx_split())

allowable_features = {
    'atomic_num' : list(range(1, 122)),# 119for mask, 120 for collection
    'formal_charge' : ['unk',-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'chirality' : ['unk',
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'hybridization' : ['unk',
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'numH' : ['unk',0, 1, 2, 3, 4, 5, 6, 7, 8],
    'implicit_valence' : ['unk',0, 1, 2, 3, 4, 5, 6],
    'degree' : ['unk',0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'isaromatic':[False,True],

    'bond_type' : ['unk',
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'bond_isconjugated':[False,True],
    'bond_inring':[False,True],
    'bond_stereo': ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE","STEREOCIS", "STEREOTRANS"]
}
atom_dic = [len(allowable_features['atomic_num']),len(allowable_features['formal_charge']),len(allowable_features['chirality' ]),
            len(allowable_features['hybridization']),len(allowable_features['numH' ]),len(allowable_features['implicit_valence']),
            len(allowable_features['degree']),len(allowable_features['isaromatic'])]
bond_dic = [len(allowable_features['bond_type']),len(allowable_features['bond_dirs' ]),len(allowable_features['bond_isconjugated']),
            len(allowable_features['bond_inring']),len(allowable_features['bond_stereo'])]
atom_cumsum = np.cumsum(atom_dic)
bond_cumsum = np.cumsum(bond_dic)

from rdkit.Chem.rdchem import Mol
if __name__ == '__main__':
    root = '/data/xc/molecule_datasets/pcqm4m-v2'
    suppl = Chem.SDMolSupplier(os.path.join(root, 'pcqm4m-v2-train.sdf'))
    for idx, mol in enumerate(suppl):
        print(f'{idx}-th rdkit mol obj: {mol}')
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = \
                [allowable_features['atomic_num'].index(atom.GetAtomicNum())] + \
                [allowable_features['formal_charge'].index(atom.GetFormalCharge()) + atom_cumsum[0]] + \
                [allowable_features['chirality'].index(atom.GetChiralTag()) + atom_cumsum[1]] + \
                [allowable_features['hybridization'].index(atom.GetHybridization()) + atom_cumsum[2]] + \
                [allowable_features['numH'].index(atom.GetTotalNumHs()) + atom_cumsum[3]] + \
                [allowable_features['implicit_valence'].index(atom.GetImplicitValence()) + atom_cumsum[4]] + \
                [allowable_features['degree'].index(atom.GetDegree()) + atom_cumsum[5]] + \
                [allowable_features['isaromatic'].index(atom.GetIsAromatic()) + atom_cumsum[6]]
            atom_features_list.append(atom_feature)
        if idx > 100:
            break


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
from torch_geometric.data import DataLoader

from utils import sdf2graph, rdf2graph

class PCQM4Mv2Dataset_3D(InMemoryDataset):
    def __init__(self, root='../../../../../data/xc/molecule_datasets/pcqm4m-v2',
                 sdf2graph=sdf2graph,
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.smiles2graph = sdf2graph
        # self.root = os.path.join(root, 'pcqm4m-v2')
        self.pre_transform = pre_transform
        super(PCQM4Mv2Dataset_3D, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.get(self.indices()[idx])
            idx_rd = random.randrange(self.__len__())
            data_random = self.get(self.indices()[idx_rd])
            data = data if self.transform is None else self.transform(data, data_random)
            return data
        else:
            return self.index_select(idx)

    @property
    def raw_file_names(self):
        return ['data.csv.gz', 'pcqm4m-v2-train.sdf']

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_conf')

    @property
    def processed_file_names(self):
        return 'geometric_data_processed_3dconf.pt'

    def download(self):
        print('Please checkout if the raw file is downloaded!')

    # @property
    # def processed_dir(self) -> str:
    #     return osp.join(self.root, 'processed_3d')


    def process(self):
        csv_data = pd.read_csv(self.raw_paths[0])
        sdf_data = Chem.SDMolSupplier(self.raw_paths[1],
                                      sanitize=True, removeHs=True, strictParsing=True)
        smiles_list = csv_data['smiles']
        homolumogap_list = csv_data['homolumogap']
        data_list = []
        print("Converting SDF file into graphs")
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            rdkit_mol = AllChem.MolFromSmiles(smiles)
            if i < len(sdf_data):
                sdf_mol = sdf_data[i]
            else:
                sdf_mol = None
            data = rdf2graph(rdkit_mol, sdf_mol)

            assert (len(data['edge_attr']) == data['edge_index'].shape[1])
            assert (len(data['x']) == data.num_nodes)

            # data.__num_nodes__ = int(graph.num_nodes)
            # data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            # data.edge_attr = torch.from_numpy(graph['edge_attr']).to(torch.int64)
            # data.x = torch.from_numpy(graph['x']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            data_list.append(data)


        # double-check prediction target
        # if self.pre_transform is not None:
        #     loader = DataLoader(data_list, batch_size=128, shuffle=False, num_workers=1)
        #     data_pos = self.pre_transform(loader)
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
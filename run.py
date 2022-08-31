# -*- coding: utf-8 -*-
# @Filename: run
# @Date: 2022-06-21 08:52
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import torch_geometric


import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import RandomSampler
from torch.optim.lr_scheduler import StepLR

from pcqm4m.gnn import GNN
from model.GTransformer import MolGNet

import os
from tqdm import tqdm
import argparse
import numpy as np
import random

from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from ogb.utils import smiles2graph

from dataset.pcqm4mv2 import PCQM4Mv2Dataset
from utils.loader import DataLoaderMasking
from utils.compose import *


parser = argparse.ArgumentParser()
parser.add_argument('--gnn', type=str, default='gin-virtual',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--graph_pooling', type=str, default='sum',
                    help='graph pooling strategy mean or sum (default: sum)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--drop_ratio', type=float, default=0.1,
                    help='dropout ratio (default: 0)')
parser.add_argument('--heads', type=int, default=10,
                    help='multi heads (default: 10)')
parser.add_argument('--num_message_passing', type=int, default=3,
                    help='message passing steps (default:3)')
parser.add_argument('--num_layers', type=int, default=5,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=600,
                    help='dimensionality of hidden units in GNNs (default: 600)')
parser.add_argument('--train_subset', default=True,action='store_true')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--log_dir', type=str, default="log",
                    help='tensorboard log directory')
parser.add_argument('--checkpoint_dir', type=str, default='ckpt', help='directory to save checkpoint')
parser.add_argument('--save_test_dir', type=str, default='saved', help='directory to save test submission file')
args = parser.parse_args()
print(args)

def train(model, rank, loader,  criterion, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(loader):
        batch = batch.to(rank)

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss =  criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, rank, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(rank)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, rank, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(rank)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred

def run(rank, world_size, dataset_root):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '32770'
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank, world_size=world_size)
    device = torch.device("cuda", rank)

    transform = Compose(
        [
            Self_loop(),
            Add_seg_id(),
            Add_collection_node(num_atom_type=119, bidirection=False)
        ]
    )

    # if args.gnn == 'GTransformer':
    dataset = PCQM4Mv2Dataset(root=dataset_root, transform=transform)
    split_idx = dataset.get_idx_split()
    evaluator = PCQM4Mv2Evaluator()
    
    train_dataset = dataset[split_idx['train']]
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank)                           
    train_loader = DataLoaderMasking(train_dataset, batch_size=args.batch_size,
    num_workers=args.num_workers, sampler=train_sampler)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    
    num_tasks = 1
    
    # 将模型参数传递到每个GPU上
    if args.gnn == 'GTransformer':
        model = MolGNet(args.num_layers, args.emb_dim, args.heads, args.num_message_passing,
                        num_tasks, args.drop_ratio, args.graph_pooling, device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', virtual_node = True, JK='sum',
            residual = True, num_layers=args.num_layers,
            emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, graph_pooling=args.graph_pooling).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    reg_criterion = torch.nn.L1Loss()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    if rank == 0:
        valid_loader = DataLoaderMasking(dataset[split_idx['valid']], batch_size=args.batch_size,
                                                shuffle=False, num_workers = args.num_workers)
        if args.save_test_dir != '':
            testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
            testchallenge_loader = DataLoader(dataset[split_idx["test-challenge"]], batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)

        if args.checkpoint_dir != '':
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        if args.log_dir != '':
            writer = SummaryWriter(log_dir=args.log_dir)

        best_valid_mae = 1000



    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, rank, train_loader, reg_criterion, optimizer)
        
        # 数据同步
        dist.barrier()

        if rank == 0:  # evaluate and write log on a single GPU
            print('Evaluating...')
            valid_mae = eval(model, rank, valid_loader, evaluator)

            print({'Train': train_mae, 'Validation': valid_mae})

            if args.log_dir != '':
                writer.add_scalar('valid/mae', valid_mae, epoch)
                writer.add_scalar('train/mae', train_mae, epoch)

            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                if args.checkpoint_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae,
                                'num_params': num_params}
                    torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

                if args.save_test_dir != '':
                    testdev_pred = test(model, rank, testdev_loader)
                    testdev_pred = testdev_pred.cpu().detach().numpy()

                    testchallenge_pred = test(model, rank, testchallenge_loader)
                    testchallenge_pred = testchallenge_pred.cpu().detach().numpy()

                    print('Saving test submission file...')
                    evaluator.save_test_submission({'y_pred': testdev_pred}, args.save_test_dir, mode='test-dev')
                    evaluator.save_test_submission({'y_pred': testchallenge_pred}, args.save_test_dir,
                                                mode='test-challenge')

            print(f'Best validation MAE so far: {best_valid_mae}')
            if args.log_dir != '':
                writer.close()

        dist.barrier()
        scheduler.step()

    # finish training
    dist.destroy_process_group()

if __name__ == '__main__':
    dataset_root = 'dataset/'
    world_size = torch.cuda.device_count()  # 进程数=GPU个数
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset_root), nprocs=world_size, join=True) # 启动4个线程
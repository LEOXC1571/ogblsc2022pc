# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: pretrain.py
# @Author: Leo Xu
# @Date: 2022/9/3 10:33
# @Email: leoxc1571@163.com
# Description:

import os
import random
import argparse
import numpy as np
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from ogb.lsc import PCQM4Mv2Evaluator
from utils.smiles2graph import smilestograph

from dataset.pcqm4mv2 import PCQM4Mv2Dataset
from utils.loader import DataLoaderMasking
from utils.compose import *


reg_criterion = torch.nn.L1Loss()


def train(model_2d, model_3d, loader_2d, loader_3d, optimizer_2d, optimizer_3d, device):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()
    return loss_accum / (step + 1)


def eval(model_2d, model_3d, loader_2d, loader_3d, evaluator, device):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def main(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10422'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    print('Parameters preparation complete! Start loading networks...')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    current_path = os.path.dirname(os.path.realpath(__file__))
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    transform = Compose(
        [
            Self_loop(),
            Add_seg_id(),
            Add_collection_node(num_atom_type=119, bidirection=False)
        ]
    )
    dataset_2d = PCQM4Mv2Dataset(root=args.dataset_root, smiles2graph=smilestograph, transform=transform)
    dataset_3d = PCQM4Mv2Dataset(root=args.dataset_root, smiles2graph=smilestograph, transform=transform)

    # split_idx = dataset_2d.get_idx_split()
    split_idx = dataset_3d.get_idx_split()

    evaluator = PCQM4Mv2Evaluator()

    train_sampler = DistributedSampler(dataset_2d[split_idx['train']], num_replicas=world_size,
                                          rank=rank, shuffle=True)
    train_loader_2d = DataLoaderMasking(dataset_2d[split_idx['train']], batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, sampler=train_sampler)
    train_loader_3d = DataLoaderMasking(dataset_3d[split_idx['train']], batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, sampler=train_sampler)

    if args.gnn_2d == 'GTransformer':
        from model.GTransformer import MolGNet
        num_tasks = 1
        model_2d = MolGNet(args.num_layers, args.emb_dim, args.heads, args.num_message_passing,
                        num_tasks, args.drop_ratio, args.graph_pooling, device)
        model_2d = DistributedDataParallel(model_2d, device_ids=[rank])
    elif args.gnn_2d == 'GTransformer_graseq':
        from model.GTransformer_graseq import MolGNet
        num_tasks = 1
        model_2d = MolGNet(args.num_layers, args.emb_dim, args.heads, args.num_message_passing,
                        num_tasks, args.drop_ratio, args.graph_pooling, device)
        model_2d = DistributedDataParallel(model_2d, device_ids=[rank], find_unused_parameters=True)
    else:
        raise ValueError('Invalid 2D-GNN type')

    if args.gnn_3d == 'GTransformer':
        from model.GTransformer import MolGNet
        num_tasks = 1
        model_3d = MolGNet(args.num_layers, args.emb_dim, args.heads, args.num_message_passing,
                        num_tasks, args.drop_ratio, args.graph_pooling, device)
        model_3d = DistributedDataParallel(model_3d, device_ids=[rank])
    elif args.gnn_3d == 'GTransformer_graseq':
        from model.GTransformer_graseq import MolGNet
        num_tasks = 1
        model_3d = MolGNet(args.num_layers, args.emb_dim, args.heads, args.num_message_passing,
                        num_tasks, args.drop_ratio, args.graph_pooling, device)
        model_3d = DistributedDataParallel(model_3d, device_ids=[rank], find_unused_parameters=True)
    else:
        raise ValueError('Invalid 3D-GNN type')

    num_params_2d = sum(p.numel() for p in model_2d.parameters())
    num_params_3d = sum(p.numel() for p in model_2d.parameters())
    print(f'#2D Params: {num_params_2d}')
    print(f'#3D Params: {num_params_3d}')


    optimizer_2d = optim.Adam(model_2d.parameters(), lr=args.lr)
    optimizer_3d = optim.Adam(model_3d.parameters(), lr=args.lr)
    scheduler_2d = StepLR(optimizer_2d, step_size=30, gamma=0.25)
    scheduler_3d = StepLR(optimizer_3d, step_size=30, gamma=0.25)

    if rank == 0:
        valid_loader_2d = DataLoaderMasking(dataset_2d[split_idx['valid']], batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)
        valid_loader_3d = DataLoaderMasking(dataset_3d[split_idx['valid']], batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)
        if args.checkpoint_dir != '':
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        if args.log_dir != '':
            writer = SummaryWriter(log_dir=args.log_dir)
        best_valid_mae = 1000

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_loss = train(model_2d, model_3d, train_loader_2d, train_loader_3d, optimizer_2d, optimizer_3d, device)

        dist.barrier()

        if rank == 0:
            print('Evaluating...')
            valid_loss = eval(model_2d, model_3d, valid_loader_2d, valid_loader_3d, evaluator, device)

            print({'Train': train_loss, 'Validation': valid_loss})

            if args.log_dir != '':
                writer.add_scalar('valid/mae', valid_loss, epoch)
                writer.add_scalar('train/mae', train_loss, epoch)

            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                if args.checkpoint_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae,
                                  'num_params': num_params}
                    torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))
            print(f'Best validation MAE so far: {best_valid_mae}')
            if args.log_dir != '':
                writer.close()
        scheduler.step()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--dataset_root', type=str, default='../../../../data/xc/molecule_datasets')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gnn_2d', type=str, default='GTransformer')
    parser.add_argument('--gnn_3d', type=str, default='GTransformer')
    parser.add_argument('--drop_ratio', type=float, default=0)
    parser.add_argument('--heads', type=int, default=10)
    parser.add_argument('--graph_pooling', type=str, default='sum')
    parser.add_argument('--num_message_passing', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=600)
    parser.add_argument('--train_subset', default=False, action='store_true')

    parser.add_argument('--log_dir', type=str, default="pretrain/log")
    parser.add_argument('--checkpoint_dir', type=str, default='pretrain/ckpt')
    parser.add_argument('--save_test_dir', type=str, default='pretrain/saved')
    print("Start preparing for parameters...")
    args = parser.parse_args()
    print(args)

    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)

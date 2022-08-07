# -*- coding: utf-8 -*-
# @Filename: run
# @Date: 2022-06-21 08:52
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import RandomSampler
from torch.optim.lr_scheduler import StepLR

from pcqm4m.gnn import GNN

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

reg_criterion = torch.nn.L1Loss()


def train(model, device, loader, optimizer):
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


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--heads', type=int, default=10,
                        help='multi heads (default: 10)')
    parser.add_argument('--num_message_passing', type=int, default=3,
                        help='message passing steps (default:3)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='', help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    current_path = os.path.dirname(os.path.realpath(__file__))
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    transform = Compose(
        [
            Self_loop(),
            Add_seg_id(),
            Add_collection_node(num_atom_type=119, bidirection=False)
        ]
    )

    if args.gnn == 'GTransformer':
        dataset = PCQM4Mv2Dataset(root='../../../../data/xc/molecule_datasets',
                                  smiles2graph=smiles2graph, transform=transform)
    else:
        dataset = PygPCQM4Mv2Dataset(root='../../../../data/xc/molecule_datasets')

    if args.gnn == 'GTransformer':
        train_sampler = RandomSampler(dataset)

    split_idx = dataset.get_idx_split()

    evaluator = PCQM4Mv2Evaluator()

    if args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio * len(split_idx["train"]))]
        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
    else:
        if args.gnn == 'GTransformer':
            train_loader = DataLoaderMasking(dataset[split_idx['train']], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoaderMasking(dataset[split_idx['valid']], batch_size=args.batch_size,
                                             shuffle=False, num_workers = args.num_workers)
        else:
            train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers)

    if args.save_test_dir != '':
        testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
        testchallenge_loader = DataLoader(dataset[split_idx["test-challenge"]], batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }


    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'GTransformer':
        from model.GTransformer import MolGNet
        num_tasks = 1
        model = MolGNet(args.num_layers, args.emb_dim, args.heads, args.num_message_passing,
                        num_tasks, args.drop_ratio, args.graph_pooling, device)
    else:
        raise ValueError('Invalid GNN type')

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, device, train_loader, optimizer)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator)

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
                testdev_pred = test(model, device, testdev_loader)
                testdev_pred = testdev_pred.cpu().detach().numpy()

                testchallenge_pred = test(model, device, testchallenge_loader)
                testchallenge_pred = testchallenge_pred.cpu().detach().numpy()

                print('Saving test submission file...')
                evaluator.save_test_submission({'y_pred': testdev_pred}, args.save_test_dir, mode='test-dev')
                evaluator.save_test_submission({'y_pred': testchallenge_pred}, args.save_test_dir,
                                               mode='test-challenge')

        scheduler.step()

        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir != '':
        writer.close()


if __name__ == '__main__':
    main()

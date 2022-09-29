# -*- coding: utf-8 -*-
# @Filename: run
# @Date: 2022-06-21 08:52
# @Author: Leo Xu
# @Email: leoxc1571@163.com


import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from ogb.lsc import PCQM4Mv2Evaluator
from utils import sdf2graph, add_conf

# from dataset.pcqm4mv2_gen import PCQM4Mv2Dataset_3D
from dataset.pcqm4mv2_3d import PCQM4Mv2Dataset_3D


def train(model, device, loader, criterion, optimizer):
    model.train()
    loss_accum = 0

    start_time = time.time()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        load_time = time.time()
        print('Load time:', load_time-start_time)
        batch = batch.to(device)
        pred = model(batch).view(-1,)
        fw_time = time.time()
        print('Forward time:', fw_time-load_time)
        optimizer.zero_grad()
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        opt_time = time.time()
        print('Opt time:', opt_time-fw_time)
        loss_accum += loss.detach().cpu().item()
        print('Train time:', opt_time - start_time)
        start_time = time.time()
    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator, batch_size, pos=None):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            if pos is not None:
                pos = torch.cat(list(pos[step*batch_size: (step*batch_size+batch.batch[-1])+1])).to(device)
                batch.pos = pos
            pred = model(batch).view(-1)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader, batch_size, pos=None):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            if pos is not None:
                pos = torch.cat(list(pos[step*batch_size: (step*batch_size+batch.batch[-1])+1])).to(device)
                batch.pos = pos
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


def import_model(args):
    if args.gnn == 'ComENet':
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
    else:
        raise ValueError('Invalid MODEL type')
    return model


def main(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10192'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    # print('Parameters preparation complete! Start loading networks...')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    current_path = os.path.dirname(os.path.realpath(__file__))
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    dataset = PCQM4Mv2Dataset_3D(root=args.dataset_root, sdf2graph=sdf2graph)
    split_idx = dataset.get_idx_split()

    valid_idx = int(len(split_idx['train']) * 0.9)

    if rank == 0 and args.conf_gen and args.conf_ckpt is not None:
        if not os.path.exists('ckpt/pos_ckpt.pt'):
            from model import DMCG
            gen_model = DMCG().to(device)
            conf_ckpt = torch.load(args.conf_ckpt, map_location=device)["model_state_dict"]
            cur_state_dict = gen_model.state_dict()
            del_keys = []
            for k in conf_ckpt.keys():
                if k not in cur_state_dict:
                    del_keys.append(k)
            for k in del_keys:
                del conf_ckpt[k]
            gen_model.load_state_dict(conf_ckpt)
            unk_loader = DataLoader(dataset[len(split_idx['train']):], batch_size=2048, shuffle=False,
                                      num_workers=args.num_workers)
            mol_pred, idx = add_conf(gen_model, unk_loader, device)
            pos_ckpt = {
                'idx': idx,
                'pos': mol_pred
            }
            torch.save(pos_ckpt, 'ckpt/pos_ckpt.pt')
            del gen_model, conf_ckpt, cur_state_dict, del_keys, unk_loader, mol_pred, idx
        else:
            pass

    dataset_start_time = time.time()
    train_sampler = DistributedSampler(dataset[split_idx['train']][:valid_idx], num_replicas=world_size,
                                       rank=rank, shuffle=True)
    dataset_middle_time = time.time()
    train_loader = DataLoader(dataset[split_idx['train']][:valid_idx], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, sampler=train_sampler)
    dataset_end_time = time.time()
    print(dataset_end_time-dataset_start_time)

    model_start_time = time.time()
    model = import_model(args)
    model.to(device)
    model_load_time = time.time()
    model = DistributedDataParallel(model, device_ids=[rank])
    model_dist_time = time.time()
    print(model_load_time-model_start_time, model_dist_time-model_load_time)


    criterion = torch.nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.25)


    if rank == 0:
        evaluator = PCQM4Mv2Evaluator()
        valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

        pos_ckpt = torch.load('ckpt/pos_ckpt.pt')

        if args.checkpoint_dir != '':
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        if args.log_dir != '':
            writer = SummaryWriter(log_dir=args.log_dir)

        best_valid_mae = 1000

        print(f"Number of training samples: {len(dataset[split_idx['train']])}, Number of validation samples: {len(dataset[split_idx['valid']])}")
        print(f"Number of test-dev samples: {len(dataset[split_idx['test-dev']])}, Number of test-challenge samples: {len(dataset[split_idx['test-challenge']])}")

        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}', f'#GPU Memory Used: {torch.cuda.memory_allocated()}')

    dist.barrier()
    
    for epoch in range(1, args.epochs + 1):
        train_start = time.time()
        train_loader.sampler.set_epoch(epoch)
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, device, train_loader, criterion, optimizer)
        dist.barrier()
        train_end = time.time()
        print(train_end - train_start)

        if rank == 0:
            print('Evaluating...')
            if args.conf_gen:
                valid_pos = pos_ckpt['pos'][split_idx['valid'] - 3378606]
                valid_mae = eval(model, device, valid_loader, evaluator, args.batch_size, pos=valid_pos)
            else:
                valid_mae = eval(model, device, valid_loader, evaluator)

            print(f'Epoch: {epoch:03d}, Train: {train_mae:.4f}, \
            Validation: {valid_mae:.4f}')

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

            print(f'Best validation MAE so far: {best_valid_mae:.4f}')

        scheduler.step()

    # save submission file use best model in main process
    if rank == 0:
        # close log file
        if args.log_dir != '':
            writer.close()
        if args.save_test_dir != '' and args.checkpoint_dir != '':
            testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False)
            testchallenge_loader = DataLoader(dataset[split_idx["test-challenge"]], batch_size=args.batch_size,
                                        shuffle=False)
            test_pos = pos_ckpt['pos'][split_idx['test-dev'] - 3378606]
            testchallenge_pos = pos_ckpt['pos'][split_idx['test-challenge'] - 3378606]

            best_model = import_model(args)
            best_model.to(device)
            best_model_ckpt = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pt'))
            print(f"best val mae: {best_model_ckpt['best_val_mae']:.4f}")
            best_model.load_state_dict(best_model_ckpt['model_state_dict'])

            testdev_pred = test(best_model, device, testdev_loader, args.batch_size, pos=test_pos)
            testdev_pred = testdev_pred.cpu().detach().numpy()
            testchallenge_pred = test(best_model, device, testchallenge_loader, args.batch_size, pos=testchallenge_pos)
            testchallenge_pred = testchallenge_pred.cpu().detach().numpy()

            print('Saving test submission file...')
            evaluator.save_test_submission({'y_pred': testdev_pred}, args.save_test_dir, mode='test-dev')
            evaluator.save_test_submission({'y_pred': testchallenge_pred}, args.save_test_dir,
                                        mode='test-challenge')

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--dataset_root', type=str, default='../../../../data/xc/molecule_datasets/pcqm4m-v2')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--conf_gen', action='store_true', default=False)
    parser.add_argument('--conf_ckpt', type=str, default='conf_ckpt/checkpoint_20.pt')

    parser.add_argument('--gnn', type=str, default='ComENet')
    parser.add_argument('--drop_ratio', type=float, default=0)
    parser.add_argument('--heads', type=int, default=10)
    parser.add_argument('--graph_pooling', type=str, default='sum')
    parser.add_argument('--num_message_passing', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=600)
    parser.add_argument('--train_subset', default=False, action='store_true')

    parser.add_argument('--log_dir', type=str, default="log")
    parser.add_argument('--checkpoint_dir', type=str, default='ckpt')
    parser.add_argument('--save_test_dir', type=str, default='saved')
    print("Start preparing for parameters...")
    args = parser.parse_args()
    print(args)

    os.environ['NCCL_SHM_DISABLE'] = '1'
    # PCQM4Mv2Dataset_3D(root=args.dataset_root, sdf2graph=sdf2graph)
    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
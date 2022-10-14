# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: confgen.py
# @Author: Leo Xu
# @Date: 2022/9/21 9:06
# @Email: leoxc1571@163.com
# Description:

import os
<<<<<<< HEAD
=======
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
>>>>>>> ff481610a47f6a4a130a33cb757ce67a5fd79176
import argparse
import torch
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
# from dataset.pcqm4mv2_gen import ConfGenDataset
# from dataset.pcqm4mv2_3d import PCQM4Mv2Dataset_3D
from dataset.pcqm4mv2_gen import PCQM4Mv2Dataset_3D
from model import DMCG
from torch.optim.lr_scheduler import LambdaLR
from model.dmcg.utils import Cosinebeta, WarmCosine, set_rdmol_positions, get_best_rmsd, init_distributed_mode
import io
import json
from collections import defaultdict
from torch.utils.data import DistributedSampler

# os.environ['NCCL_SHM_DISABLE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def train(model, device, loader, optimizer, scheduler, args):
    model.train()
    loss_accum_dict = defaultdict(float)
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            atom_pred_list, extra_output = model(batch)
            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(atom_pred_list, extra_output, batch, args)
            loss.backward()
            if args.grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            scheduler.step()

            for k, v in loss_dict.items():
                loss_accum_dict[k] += v.detach().item()

            if step % args.log_interval == 0:
                description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
                description += f" lr: {scheduler.get_last_lr()[0]:.5e}"
                description += f" vae_beta: {args.vae_beta:6.4f}"
                # for k in loss_accum_dict.keys():
                #     description += f" {k}: {loss_accum_dict[k]/(step+1):6.4f}"

                pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    return loss_accum_dict


def evaluate(model, device, loader, args):
    model.eval()
    mol_labels = []
    mol_preds = []
    for batch in tqdm(loader, desc="Iteration", disable=args.disable_tqdm):
        batch = batch.to(device)
        with torch.no_grad():
            pred, _ = model(batch)
        pred = pred[-1]
        batch_size = batch.num_graphs
        n_nodes = torch.cat((batch.batch.diff(), torch.LongTensor([1,]).to(batch.batch.device)))
        n_nodes = torch.where(n_nodes==1)[0]
        n_nodes = torch.cat((torch.LongTensor([-1,]).to(batch.batch.device), n_nodes)).diff()
        n_nodes = n_nodes.tolist()
        pre_nodes = 0
        for i in range(batch_size):
            mol_labels.append(batch.mol[i])
            mol_preds.append(
                set_rdmol_positions(batch.mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]

    rmsd_list = []
    for gen_mol, ref_mol in zip(mol_preds, mol_labels):
        try:
            rmsd_list.append(get_best_rmsd(gen_mol, ref_mol))
        except Exception as e:
            continue

    return np.mean(rmsd_list)


def main(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10192'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    CosineBeta = Cosinebeta(args)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset = PCQM4Mv2Dataset_3D(root='/home/tiger/lsc2022self/dataset/pcqm4m-v2')
    split_idx = dataset.get_idx_split()
    index = torch.LongTensor(random.sample(range(len(split_idx['train'])), int(len(split_idx['train'])/4)))
    valid_idx = int(len(index) * 0.8)
    dataset_train = dataset[index[:valid_idx]]
    dataset_valid = dataset[index[valid_idx:]]

    del dataset
    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, sampler=sampler_train)

    del dataset_train
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

    del dataset_valid

    shared_params = {
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_layers": args.mlp_layers,
        "latent_size": args.latent_size,
        "use_layer_norm": args.use_layer_norm,
        "num_message_passing_steps": args.num_layers,
        "global_reducer": args.global_reducer,
        "node_reducer": args.node_reducer,
        "dropedge_rate": args.dropedge_rate,
        "dropnode_rate": args.dropnode_rate,
        "dropout": args.dropout,
        "layernorm_before": args.layernorm_before,
        "encoder_dropout": args.encoder_dropout,
        "use_bn": args.use_bn,
        "vae_beta": args.vae_beta,
        "decoder_layers": args.decoder_layers,
        "reuse_prior": args.reuse_prior,
        "cycle": args.cycle,
        "pred_pos_residual": args.pred_pos_residual,
        "node_attn": args.node_attn,
        "shared_decoder": args.shared_decoder,
        "use_global": args.use_global,
        "sg_pos": args.sg_pos,
        "shared_output": args.shared_output,
        "use_ss": args.use_ss,
        "rand_aug": args.rand_aug,
        "no_3drot": args.no_3drot,
        "not_origin": args.not_origin,
    }

    model = DMCG(**shared_params).to(device)
<<<<<<< HEAD
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # model = model.module
=======
    model = DistributedDataParallel(model, device_ids=[rank])
    model_without_ddp = model.module
>>>>>>> ff481610a47f6a4a130a33cb757ce67a5fd79176
    args.checkpoint_dir = "" if rank != 0 else args.checkpoint_dir
    args.enable_tb = False if rank != 0 else args.enable_tb
    args.disable_tqdm = rank != 0

    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, args.beta2),
        weight_decay=args.weight_decay,
    )

    if not args.lr_warmup:
        scheduler = LambdaLR(optimizer, lambda x: 1.0)
    else:
        lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=int(4e3))
        scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))

    if args.checkpoint_dir and args.enable_tb:
        tb_writer = SummaryWriter(args.checkpoint_dir)

    train_curve = []
    valid_curve = []
    test_curve = []

    dist.barrier()

    for epoch in range(1, args.epochs + 1):
        sampler_train.set_epoch(epoch)
        CosineBeta.step(epoch - 1)
        print("=====Epoch {}".format(epoch))
        print("Training...")
        loss_dict = train(model, device, train_loader, optimizer, scheduler, args)
        if rank == 0:
            print("Evaluating...")
            valid_pref = evaluate(model, device, valid_loader, args)
            if args.checkpoint_dir:
                print(f"Setting {os.path.basename(os.path.normpath(args.checkpoint_dir))}...")
            valid_curve.append(valid_pref)
<<<<<<< HEAD
            # "Train": train_pref
=======

>>>>>>> ff481610a47f6a4a130a33cb757ce67a5fd79176
            logs = {"Valid": valid_pref}
            with io.open(
                os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)

            # ddp save
            if epoch % args.ckpt_interval == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    # "args": args,
                }

                torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))

            if args.enable_tb:
<<<<<<< HEAD
                #tb_writer.add_scalar("evaluation/train", train_pref, epoch)
=======
>>>>>>> ff481610a47f6a4a130a33cb757ce67a5fd79176
                tb_writer.add_scalar("evaluation/valid", valid_pref, epoch)
                # tb_writer.add_scalar("evaluation/test", test_pref, epoch)
                for k, v in loss_dict.items():
                    tb_writer.add_scalar(f"training/{k}", v, epoch)

    best_val_epoch = np.argmin(np.array(valid_curve))
    if args.checkpoint_dir and args.enable_tb:
        tb_writer.close()
    torch.distributed.destroy_process_group()
    print("Finished traning!")
    print(f"Best validation epoch: {best_val_epoch+1}")
    print(f"Best validation score: {valid_curve[best_val_epoch]}")
    print(f"Test score: {test_curve[best_val_epoch]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_warmup", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--global_reducer", type=str, default="sum")
    parser.add_argument("--node_reducer", type=str, default="sum")
    parser.add_argument("--graph_pooling", type=str, default="sum")
    parser.add_argument("--dropedge_rate", type=float, default=0.1)
    parser.add_argument("--dropnode_rate", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--decoder_layers", type=int, default=None)
    parser.add_argument("--latent_size", type=int, default=256)
    parser.add_argument("--mlp_hidden_size", type=int, default=512)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use_layer_norm", action="store_true", default=False)

    # parser.add_argument("--log-dir", type=str, default="", help="tensorboard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default='conf_ckpt')
    parser.add_argument("--ckpt_interval", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)  # 0.2
    parser.add_argument("--encoder_dropout", type=float, default=0.0)
    parser.add_argument("--layernorm_before", action="store_true", default=False)
    parser.add_argument("--use_bn", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--use_adamw", action="store_true", default=True)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--enable_tb", action="store_true", default=True)

    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--aux_loss", type=float, default=0.1)  # 0.2
    parser.add_argument("--train_subset", action="store_true", default=False)
    parser.add_argument("--extend_edge", action="store_true", default=False)
    parser.add_argument("--reuse_prior", action="store_true", default=False)
    parser.add_argument("--cycle", type=int, default=1)

    parser.add_argument("--vae_beta", type=float, default=1.0)
    parser.add_argument("--vae_beta_max", type=float, default=0.05)
    parser.add_argument("--vae_beta_min", type=float, default=0.0001)
    parser.add_argument("--pred_pos_residual", action="store_true", default=True)
    parser.add_argument("--node_attn", action="store_true", default=True)
    parser.add_argument("--global_attn", action="store_true", default=False)
    parser.add_argument("--shared_decoder", action="store_true", default=False)
    parser.add_argument("--shared_output", action="store_true", default=True)
    parser.add_argument("--clamp_dist", type=float, default=None)
    parser.add_argument("--use_global", action="store_true", default=False)
    parser.add_argument("--sg_pos", action="store_true", default=False)
    parser.add_argument("--remove_hs", action="store_true", default=True)
    parser.add_argument("--grad_norm", type=float, default=None)  # 10.0
    parser.add_argument("--use_ss", action="store_true", default=False)
    parser.add_argument("--rand_aug", action="store_true", default=False)
    parser.add_argument("--not_origin", action="store_true", default=False)
    parser.add_argument("--ang_lam", type=float, default=0.)
    parser.add_argument("--bond_lam", type=float, default=0.)
    parser.add_argument("--no_3drot", action="store_true", default=True)

    args = parser.parse_args()
    os.environ['NCCL_SHM_DISABLE'] = '1'
    world_size = torch.cuda.device_count()

    PCQM4Mv2Dataset_3D(root='/home/tiger/lsc2022self/dataset/pcqm4m-v2')
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
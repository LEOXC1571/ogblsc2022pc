# ogblsc2022

## log
- 2022-7-18 model: GIN-Virtual {'Train': 0.0948409306068622, 'Validation': 0.11321253329515457} Best validation MAE so far: 0.10685155540704727
{'Train': 0.10280203166671775, 'Validation': 0.1292327642440796}
Best validation MAE so far: 0.12899625301361084

## Workflow
- Tiger: GNN简化，EGAT等
- Leo: 基于3D的GNN构建，Pretrain任务
- Ruifeng: 官能团信息的引入

## Distributed Parallel
You should add the following code in .bashrc, to keep Pytroch multiGPU works.

## New Model
### Dataset pipeline: 
- Train Set (3D Geometry original + RDKIT extracted, 2D Graph, Node & Edge Attr.)
- Valid/Test Set (3D Geometry extracted by RDKIT, 2D Graph, Node & Edge Attr.)
- Train Split
### Model pipeline:
- Geometry learning (Random Sampling from train set)
- ComENet Based + Edge Attr. + Attention mechanism 

``` shell
export NCCL_SHM_DISABLE=1
```


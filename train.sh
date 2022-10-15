export CUDA_VISIBLE_DEVICES=1,2

python run_3d.py --dataset_root=../../../../data/xc/molecule_datasets/pcqm4m-v2 --num_workers=1 --batch_size=3072 --conf_gen
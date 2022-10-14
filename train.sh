export CUDA_VISIBLE_DEVICES=0
python run_3d.py --dataset_root=/home/tiger/lsc2022self/dataset/pcqm4m-v2/ --num_workers=2 \
--batch_size=3072 --conf_gen
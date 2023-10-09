#!/bin/bash

# train consequently each model on a single GPU
python ./main_train.py --setting=./configs/unet.yaml --split=split_1
python ./main_train.py --setting=./configs/unet.yaml --split=split_2
python ./main_train.py --setting=./configs/unet.yaml --split=split_3
python ./main_train.py --setting=./configs/unet.yaml --split=split_4
python ./main_train.py --setting=./configs/unet.yaml --split=split_5


# to run in parallel on multiple GPUs (specify which GPU to use in each line)
#python ./main_train.py --setting=./configs/unet.yaml --split=split_1 --gpu_id=0 &
#sleep 30s
#python ./main_train.py --setting=./configs/unet.yaml --split=split_2 --gpu_id=1 &
#sleep 30s
#python ./main_train.py --setting=./configs/unet.yaml --split=split_3 --gpu_id=2 &
#sleep 30s
#python ./main_train.py --setting=./configs/unet.yaml --split=split_4 --gpu_id=3 &
#sleep 30s
#python ./main_train.py --setting=./configs/unet.yaml --split=split_5 --gpu_id=0 &

wait

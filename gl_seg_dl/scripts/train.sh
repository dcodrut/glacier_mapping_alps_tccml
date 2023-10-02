#!/bin/bash

source ~/.bashrc

# go to the project main directory
cd $CWD

# activate the conda environment
conda activate glsegenv

# train consequently each model
python ./main_train.py --setting=./configs/cv/unet_1.yaml
python ./main_train.py --setting=./configs/cv/unet_2.yaml
python ./main_train.py --setting=./configs/cv/unet_3.yaml
python ./main_train.py --setting=./configs/cv/unet_4.yaml
python ./main_train.py --setting=./configs/cv/unet_5.yaml


# to run in parallel on multiple GPUs
#python ./main_train.py --setting=./configs/cv/unet_1.yaml &
#sleep 30s
#python ./main_train.py --setting=./configs/cv/unet_2.yaml &
#sleep 30s
#python ./main_train.py --setting=./configs/cv/unet_3.yaml &
#sleep 30s
#python ./main_train.py --setting=./configs/cv/unet_4.yaml &
#sleep 30s
#python ./main_train.py --setting=./configs/cv/unet_5.yaml &

wait

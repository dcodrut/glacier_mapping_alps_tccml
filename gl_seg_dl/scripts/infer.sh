#!/bin/bash

# specify which version of the model to use (assuming the same version for all the splits)
VERSION="version_0"

python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_1/$VERSION/checkpoints --fold=s_test --test_per_glacier true
python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_2/$VERSION/checkpoints --fold=s_test --test_per_glacier true
python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_3/$VERSION/checkpoints --fold=s_test --test_per_glacier true
python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_4/$VERSION/checkpoints --fold=s_test --test_per_glacier true
python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_5/$VERSION/checkpoints --fold=s_test --test_per_glacier true

# to run in parallel on multiple GPUs (specify which GPU to use in each line)
#python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_1/$VERSION/checkpoints --fold=s_test --test_per_glacier true --gpu_id=0 &
#python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_2/$VERSION/checkpoints --fold=s_test --test_per_glacier true --gpu_id=1 &
#python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_3/$VERSION/checkpoints --fold=s_test --test_per_glacier true --gpu_id=2 &
#python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_4/$VERSION/checkpoints --fold=s_test --test_per_glacier true --gpu_id=3 &
#python main_test.py --checkpoint_dir=../data/external/scratch/experiments_server/unet/cv/split_5/$VERSION/checkpoints --fold=s_test --test_per_glacier true --gpu_id=0 &

wait

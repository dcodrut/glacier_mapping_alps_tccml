#!/bin/bash

# specify which version of the model to use (assuming the same version for all the splits)
VERSION="version_0"

# each script runs in parallel already (increase NUM_CORES_EVAL from config if needed)
python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_1/$VERSION/output/preds/rasters_orig
python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_2/$VERSION/output/preds/rasters_orig
python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_3/$VERSION/output/preds/rasters_orig
python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_4/$VERSION/output/preds/rasters_orig
python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_5/$VERSION/output/preds/rasters_orig

python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_1/$VERSION/output/preds/rasters_2023
python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_2/$VERSION/output/preds/rasters_2023
python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_3/$VERSION/output/preds/rasters_2023
python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_4/$VERSION/output/preds/rasters_2023
python main_eval.py --inference_dir=../data/external/scratch/experiments_server/unet/cv/split_5/$VERSION/output/preds/rasters_2023


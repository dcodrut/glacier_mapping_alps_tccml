import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import geopandas as gpd
import pytorch_lightning as pl
import yaml
from tqdm import tqdm
import logging

# local imports
import models
from task.data import GlSegDataModule
from task.seg import GlSegTask
from utils.general import str2bool


def test_model(settings: dict, fold: str, test_per_glacier: bool, checkpoint: str = None):
    # Logger (console and TensorBoard)
    root_logger = logging.getLogger('pytorch_lightning')
    root_logger.setLevel(logging.INFO)
    fmt = '[%(levelname)s] - %(asctime)s - %(name)s: %(message)s (%(filename)s:%(funcName)s:%(lineno)d)'
    root_logger.handlers[0].setFormatter(logging.Formatter(fmt))
    logger = logging.getLogger('pytorch_lightning.core')
    logger.info(f'Settings: {settings}')

    # Data
    data_params = settings['data']
    data_params['input_settings'] = settings['model']['inputs']
    dm = GlSegDataModule(**data_params)
    dm.train_shuffle = False  # disable shuffling for the training dataloader

    # Model
    model_class = getattr(models, settings['model']['class'])
    model = model_class(
        input_settings=settings['model']['inputs'] if 'inputs' in settings['model'] else None,
        training_settings=settings['model']['training_settings'] if 'training_settings' in settings['model'] else None,
        model_name=settings['model']['name'],
        model_args=settings['model']['args'],
    )

    # Task
    task_params = settings['task']
    task = GlSegTask(model=model, task_params=task_params, outdir=Path(checkpoint).parent.parent)
    logger.info(f'Loading model from {checkpoint}')
    if checkpoint is not None:  # allow using non-trainable models
        task.load_from_checkpoint(
            checkpoint_path=checkpoint,
            model=model,
            task_params=task_params
        )

    # Trainer
    trainer_dict = settings['trainer']
    trainer = pl.Trainer(**trainer_dict)

    logger.info(f'test_per_glacier = {test_per_glacier}')
    if test_per_glacier:
        ds_name = Path(data_params['rasters_dir']).name
        root_outdir = Path(task.outdir) / 'output' / 'preds' / ds_name
    else:
        ds_name = Path(data_params['data_root_dir']).name
        root_outdir = Path(task.outdir) / 'output' / 'stats' / ds_name

    assert fold in ['s_train', 's_valid', 's_test']
    logger.info(f'Testing for fold = {fold}')

    task.outdir = root_outdir / fold
    if not test_per_glacier:
        assert fold in ('s_train', 's_valid', 's_test')
        dm.setup('test' if fold == 's_test' else 'fit')
        if fold == 's_train':
            dl = dm.train_dataloader()
        elif fold == 's_valid':
            dl = dm.val_dataloader()
        else:
            dl = dm.test_dataloader()

        results = trainer.validate(model=task, dataloaders=dl)
        logger.info(f'results = {results}')
    else:
        fold = {'s_train': dm.train_dir_name, 's_valid': dm.val_dir_name, 's_test': dm.test_dir_name}[fold]
        split_name = str(checkpoint).split('split_')[1].split('/')[0]
        shp_fp = f'../data/s2_data/outlines_split/split_{split_name}/{fold}.shp'
        logger.info(f'Reading the glaciers numbers of the current region based on the shapefile from {shp_fp}')
        gdf = gpd.read_file(shp_fp)
        gl_num_list_crt_region = set(gdf.GLACIER_NR.astype(str))
        logger.info(f'#glaciers in the current region = {len(gl_num_list_crt_region)}')

        dir_fp = Path(dm.rasters_dir)
        logger.info(f'Reading the glaciers numbers based on the rasters from {dir_fp}')
        fp_list = list(dir_fp.glob('**/*.nc'))
        gl_num_list_crt_dir = set([p.parent.name for p in fp_list])
        logger.info(f'#glaciers in the current rasters dir = {len(gl_num_list_crt_dir)}')

        gl_num_list_crt_dir &= gl_num_list_crt_region
        logger.info(f'After keeping only the glaciers from the current region: #glaciers = {len(gl_num_list_crt_dir)}')

        dl_list = dm.test_dataloaders_per_glacier(gid_list=gl_num_list_crt_dir)
        for dl in tqdm(dl_list, desc='Testing per glacier'):
            trainer.test(model=task, dataloaders=dl)


def get_best_model_ckpt(checkpoint_dir, metric_name='val_loss_epoch', sort_method='min'):
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.exists(), f'{checkpoint_dir} not found'
    assert sort_method in ('max', 'min')

    ckpt_list = sorted(list(checkpoint_dir.glob('*.ckpt')))
    ens_list = np.array([float(p.stem.split(f'{metric_name}=')[1]) for p in ckpt_list if metric_name in str(p)])

    # get the index of the last best value
    sort_method_f = np.argmax if sort_method == 'max' else np.argmin
    i_best = len(ens_list) - sort_method_f(ens_list[::-1]) - 1
    ckpt_best = ckpt_list[i_best]

    return ckpt_best


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--settings_fp', type=str, metavar='path/to/settings.yaml', help='yaml with all the settings')
    parser.add_argument('--checkpoint', type=str, metavar='path/to/checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--checkpoint_dir', type=str, metavar='path/to/checkpoint_dir',
                        help='a directory from which the model with the lowest L1 distance will be selected '
                             '(alternative to checkpoint_file)', default=None)
    parser.add_argument('--fold', type=str, metavar='s_train|s_valid|s_test', required=True,
                        help='which subset to test on: either s_train, s_valid or s_test')
    parser.add_argument('--test_per_glacier', type=str2bool, required=True,
                        help='whether to apply the model separately for each glacier instead of using the patches'
                             '(by generating in-memory all the patches)')
    parser.add_argument('--rasters_dir', type=str, required=False,
                        help='directory on which to test the model, for the case when test_per_glacier is True; '
                             'if not provided, the one from the config file is used'
                        )
    parser.add_argument('--gpu_id', type=int, default=None, required=False
                        , help='GPU ID (if not given, the one from the config is used)'
                        )
    args = parser.parse_args()

    # prepare the checkpoint
    if args.checkpoint_dir is not None:
        # get the best checkpoint
        checkpoint_file = get_best_model_ckpt(
            checkpoint_dir=args.checkpoint_dir,
            metric_name='BinaryJaccardIndex_val_epoch_avg_per_g',
            sort_method='max'
        )
    else:
        checkpoint_file = args.checkpoint

    # get the settings (assuming it was saved in the model's results directory if not given)
    if args.settings_fp is None:
        model_dir = Path(checkpoint_file).parent.parent
        settings_fp = model_dir / 'settings.yaml'
    else:
        settings_fp = args.settings_fp

    with open(settings_fp, 'r') as fp:
        all_settings = yaml.load(fp, Loader=yaml.FullLoader)

    # overwrite the gpu id if provided
    if args.gpu_id is not None:
        all_settings['trainer']['devices'] = [args.gpu_id]

    # overwrite the raster directory from the config if needed
    if args.rasters_dir is not None:
        all_settings['data']['rasters_dir'] = args.rasters_dir

    test_model(
        settings=all_settings,
        checkpoint=checkpoint_file,
        test_per_glacier=args.test_per_glacier,
        fold=args.fold
    )

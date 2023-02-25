from pathlib import Path
from typing import Union

import xarray as xr
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np


def extract_inputs(ds, fp):
    s2_bands = ds.band_data.values[:13].astype(np.float32)  # TODO: parametrize the bands

    # extract the provided nodata mask
    mask_no_data = ds.band_data.values[ds.band_data.attrs['long_name'].index('FILL_MASK')] == 0

    # include to the nodata mask any NaN pixel
    # (it happened once that a few pixels were missing only from the last band but the mask did not include them)
    mask_na_per_band = np.isnan(s2_bands)
    if mask_na_per_band.sum() > 0:
        idx_na = np.where(mask_na_per_band)

        # fill in the gaps with the average
        avg_per_band = np.nansum(np.nansum(s2_bands, axis=-1), axis=-1) / np.prod(s2_bands.shape[-2:])
        s2_bands[idx_na[0], idx_na[1], idx_na[2]] = avg_per_band[idx_na[0]]

        # make sure that these pixels are masked in mask_no_data too
        mask_na = mask_na_per_band.any(axis=0)
        mask_no_data |= mask_na

    data = {
        's2_bands': s2_bands,
        'mask_no_data': mask_no_data,
        'mask_crt_g': ds.mask_crt_g.values == 1,
        'mask_all_g': ds.mask_all_g_id.values != -1,
        'mask_debris_crt_g': ds.mask_debris_crt_g.values == 1,
        'dem': ds.dem.values,
        'fp': str(fp),
    }

    return data


class GlSegPatchDataset(Dataset):
    def __init__(self, folder=None, fp_list=None):
        assert folder is not None or fp_list is not None

        if folder is not None:
            folder = Path(folder)
            self.fp_list = sorted(list(folder.glob('**/*.nc')))

            assert len(self.fp_list) > 0, f'No files found at: {str(folder)}'
        else:
            assert all([Path(fp).exists() for fp in fp_list])
            self.fp_list = fp_list

    def __getitem__(self, idx):
        # read the current file
        fp = self.fp_list[idx]
        ds = xr.open_dataset(fp, decode_coords='all')

        # extract the inputs
        data = extract_inputs(ds=ds, fp=fp)

        return data

    def __len__(self):
        return len(self.fp_list)


class GlSegDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_root_dir: Union[Path, str],
                 train_dir_name: str,
                 val_dir_name: str,
                 test_dir_name: str,
                 rasters_dir_name: str,
                 train_batch_size: int = 16,
                 val_batch_size: int = 32,
                 test_batch_size: int = 32,
                 train_shuffle: bool = True,
                 num_workers: int = 16,
                 pin_memory: bool = False):
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.train_dir_name = train_dir_name
        self.val_dir_name = val_dir_name
        self.test_dir_name = test_dir_name
        self.rasters_dir_name = rasters_dir_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # the following will be set when calling setup
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.test_ds_list = None

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_ds = GlSegPatchDataset(self.data_root_dir / self.train_dir_name)
            self.valid_ds = GlSegPatchDataset(self.data_root_dir / self.val_dir_name)
        if stage == 'test':
            self.test_ds = GlSegPatchDataset(self.data_root_dir / self.test_dir_name)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

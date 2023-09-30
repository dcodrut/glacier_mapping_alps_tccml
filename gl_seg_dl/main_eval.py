import os
from pathlib import Path
import multiprocessing
import xarray as xr
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
import itertools


def compute_stats(fp, mask_name='mask_crt_g', exclude_bad_pixels=True, return_rasters=False):
    stats = {'fp': fp}

    nc = xr.open_dataset(fp)

    # get the predictions and the ground truth
    mask = (nc[mask_name].values == 1)

    if 'crt_g_pred_b' in nc.data_vars:
        # TODO fix this
        preds = nc.crt_g_pred_b.values
    else:
        preds = nc.pred_b.values

    # extract the mask for cloudy, shadow or no-data pixels
    if exclude_bad_pixels:
        band_names = nc.band_data.attrs['long_name']
        mask_no_data = nc.band_data.values[band_names.index('FILL_MASK')] == 0
        # it happens (rarely) that the data has NAs, but they are not captured in the no-data mask
        mask_na = np.isnan(nc.band_data.values[:13]).any(axis=0)
        mask_clouds = nc.band_data.values[band_names.index('CLOUD_MASK')] == 1
        mask_shadow = nc.band_data.values[band_names.index('SHADOW_MASK')] == 1
        mask_exclude = (mask_no_data | mask_na | mask_clouds | mask_shadow)
    else:
        mask_exclude = np.zeros_like(mask)
    mask[mask_exclude] = False
    mask.sum()
    preds[mask_exclude] = False

    dx = nc.rio.resolution()[0]
    f_area = (dx ** 2) / 1e6
    area = np.sum(mask) * f_area
    area_excluded = np.sum((nc.mask_crt_g.values == 1) & mask_exclude) * f_area
    area_recalled = np.sum(mask & preds) * f_area
    recall = area_recalled / area if area > 0 else np.nan

    stats['area_ok'] = area
    stats['area_excluded'] = area_excluded
    stats['area_recalled'] = area_recalled
    stats['recall'] = recall

    # add debris-specific stats
    mask_debris = (nc.mask_debris_crt_g.values == 1) & mask
    area_debris = np.sum(mask_debris) * f_area
    area_debris_recalled = np.sum(mask_debris & preds) * f_area
    recall_debris = area_debris_recalled / area_debris if area_debris > 0 else np.nan
    stats['area_debris'] = area_debris
    stats['area_debris_recalled'] = area_debris_recalled
    stats['recall_debris'] = recall_debris

    # compute the FPs for the non-glacierized area (where predictions are made)
    mask_preds_exist = ~np.isnan(nc.pred.values)
    mask_non_g = (nc.mask_all_g_id.values == -1) & mask_preds_exist & (~mask_exclude)
    area_non_g = np.sum(mask_non_g) * f_area
    mask_fp = preds & mask_non_g
    area_fp = np.sum(mask_fp) * f_area
    stats['area_non_g'] = area_non_g
    stats['area_fp'] = area_fp

    # compute the FPs for the non-glacierized area but only within a certain buffer
    nc['mask_crt_g_b0'] = nc['mask_crt_g']
    for b1, b2 in list(itertools.combinations(['b0', 'b10', 'b20', 'b50'], 2)):
        mask_crt_b_interval = (nc[f'mask_crt_g_{b1}'].values == 0) & (nc[f'mask_crt_g_{b2}'].values == 1)
        mask_non_g_crt_b = mask_non_g & mask_crt_b_interval
        area_non_g_crt_b = np.sum(mask_non_g_crt_b) * f_area
        mask_fp_crt_b = preds & mask_non_g_crt_b
        area_fp_crt_b = np.sum(mask_fp_crt_b) * f_area
        stats[f'area_non_g_{b1}_{b2}'] = area_non_g_crt_b
        stats[f'area_non_g_{b1}_{b2}_excluded'] = np.sum(mask_crt_b_interval & mask_preds_exist & mask_exclude) * f_area
        stats[f'area_fp_{b1}_{b2}'] = area_fp_crt_b

    # estimate the altitude & location of the terminus
    # first by the lower ice-predicted pixel (if it's not masked), then by the median of the lowest 30 pixels
    # if there are multiple pixels with the same minimum altitude, use the average
    nc_g = nc.where(nc.mask_crt_g == 1)
    dem_pred_on = nc_g.dem.values.copy()
    dem_pred_on[~preds] = np.nan
    h_pred_on_sorted = np.sort(np.unique(dem_pred_on.flatten()))
    h_pred_on_sorted = h_pred_on_sorted[~np.isnan(h_pred_on_sorted)]

    for num_px_thr in [1, 30]:
        if len(h_pred_on_sorted) > 0:  # it can happen that all the pixels are masked
            i = 0
            h_thr = h_pred_on_sorted[i]
            while i < len(h_pred_on_sorted) - 1 and np.sum(dem_pred_on <= h_thr) < num_px_thr:
                i += 1
                h_thr = h_pred_on_sorted[i]
        else:
            h_thr = -1

        # exclude the masked pixels; if all of them are masked, the result will be NaN
        mask_lowest = (dem_pred_on <= h_thr)
        mask_lowest[mask_exclude] = False

        all_masked = (np.sum(mask_lowest) == 0)
        idx = np.where(mask_lowest)
        stats[f'term_h_{num_px_thr}_px'] = np.nan if all_masked else h_thr
        stats[f'term_x_i_{num_px_thr}_px'] = np.nan if all_masked else int(np.median(idx[1]))
        stats[f'term_y_i_{num_px_thr}_px'] = np.nan if all_masked else int(np.median(idx[0]))
        stats[f'term_x_m_{num_px_thr}_px'] = np.nan if all_masked else int(np.median(nc.x.values[idx[1]]))
        stats[f'term_y_m_{num_px_thr}_px'] = np.nan if all_masked else int(np.median(nc.y.values[idx[0]]))

    # save the filename of the original S2 data
    stats['s2_fn'] = nc.attrs['s2_fn']

    if not return_rasters:
        return stats

    rasters = {
        'mask': mask,
        'mask_exclude': mask_exclude,
        'preds': preds,
        'mask_debris': mask_debris,
    }

    return stats, rasters


if __name__ == "__main__":
    num_cores = 32
    ds_name = 'rasters_orig'
    model_output_dir = '../data/scratch/experiments_server/unet/exp_02_bands/version_52/output'
    for year in [2015 + i for i in range(8)]:
        ds_name = f'rasters_{year}'
        res_dir_root = Path(model_output_dir, 'preds', ds_name)
        stats_dir_root = Path(model_output_dir, 'stats', ds_name)
        assert res_dir_root.exists()

        for mask_name in ('mask_crt_g', 'mask_crt_g_b20'):
            for exclude_bad_pixels in (True, False):
                for split in ('s_train', 's_valid', 's_test'):
                    preds_dir = res_dir_root / split
                    fp_list = list(preds_dir.glob('**/*.nc'))
                    print(f'ds_name = {ds_name}; split = {split}; #glaciers = {len(fp_list)}')
                    if len(fp_list) == 0:
                        print(f'No rasters found for split = {split}. Skipping.')
                        continue

                    _compute_stats = partial(compute_stats, exclude_bad_pixels=exclude_bad_pixels, mask_name=mask_name)

                    with multiprocessing.Pool(num_cores) as pool:
                        all_metrics = []
                        for metrics in tqdm(
                                pool.imap_unordered(_compute_stats, fp_list, chunksize=1), total=len(fp_list),
                                desc=f'Computing evaluation metrics'):
                            all_metrics.append(metrics)
                        metrics_df = pd.DataFrame.from_records(all_metrics)

                        stats_fp = stats_dir_root / split / f'stats_excl_{exclude_bad_pixels}_{mask_name}.csv'
                        stats_fp.parent.mkdir(parents=True, exist_ok=True)
                        metrics_df = metrics_df.sort_values('fp')
                        metrics_df.to_csv(stats_fp, index=False)
                        print(f'Evaluation metrics exported to {stats_fp}')

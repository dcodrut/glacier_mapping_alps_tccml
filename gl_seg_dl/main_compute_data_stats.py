"""
    Script which computes the mean and standard deviation of the training patches from each cross-validation fold.
"""
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm

# local imports
import config as C


def compute_stats(fp):
    nc = xr.open_dataset(fp)
    band_data = nc.band_data.values
    data = np.concatenate([band_data, nc.dem.values[None, ...]], axis=0)

    stats = {
        'fp': str(fp),
    }

    # add the stats for the band data
    n_list = []
    s_list = []
    ssq_list = []
    vmin_list = []
    vmax_list = []
    for i_band in range(len(data)):
        data_crt_band = data[i_band, :, :].flatten()
        all_na = np.all(np.isnan(data_crt_band))
        n_list.append(np.sum(~np.isnan(data_crt_band), axis=0) if not all_na else 0)
        s_list.append(np.nansum(data_crt_band, axis=0) if not all_na else np.nan)
        ssq_list.append(np.nansum(data_crt_band ** 2, axis=0) if not all_na else np.nan)
        vmin_list.append(np.nanmin(data_crt_band, axis=0) if not all_na else np.nan)
        vmax_list.append(np.nanmax(data_crt_band, axis=0) if not all_na else np.nan)

    stats['n'] = n_list
    stats['sum_1'] = s_list
    stats['sum_2'] = ssq_list
    stats['vmin'] = vmin_list
    stats['vmax'] = vmax_list
    stats['var_name'] = [f'band_{i}' for i in range(len(band_data))] + ['dem']

    return stats


if __name__ == '__main__':
    data_dir_root = Path(C.S2.DIR_GL_PATCHES)
    out_dir_root = Path(C.S2.DIR_AUX_DATA) / Path(C.S2.DIR_GL_PATCHES).name
    for i_split in range(1, C.S2.NUM_CV_FOLDS + 1):
        data_dir_crt_split = data_dir_root / f'split_{i_split}' / 'fold_train'
        fp_list = sorted(list(Path(data_dir_crt_split).glob('**/*.nc')))

        all_df = []
        for fp_crt_patch in tqdm(fp_list, desc=f'Compute stats for train patches from split = {i_split}'):
            stats_crt_patch = compute_stats(fp_crt_patch)
            df = pd.DataFrame(stats_crt_patch)
            all_df.append(df)

        df = pd.concat(all_df)
        out_dir_crt_split = out_dir_root / f'split_{i_split}'
        out_dir_crt_split.mkdir(parents=True, exist_ok=True)
        fp_out = Path(out_dir_crt_split) / 'stats_train_patches.csv'
        df.to_csv(fp_out, index=False)
        print(f'Stats saved to {fp_out}')

        # compute mean and standard deviation based only on the training folds
        stats_agg = {k: [] for k in ['var_name', 'mu', 'stddev', 'vmin', 'vmax']}
        for var_name in df.var_name.unique():
            df_r1_crt_var = df[df.var_name == var_name]
            n = max(df_r1_crt_var.n.sum(), 1)
            s1 = df_r1_crt_var.sum_2.sum()
            s2 = (df_r1_crt_var.sum_1.sum() ** 2) / n
            std = np.sqrt((s1 - s2) / n)
            mu = df_r1_crt_var.sum_1.sum() / n
            stats_agg['var_name'].append(var_name)
            stats_agg['mu'].append(mu)
            stats_agg['stddev'].append(std)
            stats_agg['vmin'].append(df_r1_crt_var.vmin.quantile(0.025))
            stats_agg['vmax'].append(df_r1_crt_var.vmax.quantile(1 - 0.025))
        df_stats_agg = pd.DataFrame(stats_agg)
        fp_out = Path(out_dir_crt_split) / 'stats_train_patches_agg.csv'
        df_stats_agg.to_csv(fp_out, index=False)
        print(f'Aggregated stats saved to {fp_out}')

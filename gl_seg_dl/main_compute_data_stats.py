import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_stats(fp):
    nc = xr.open_dataset(fp)
    band_data = nc.band_data.values
    data = np.concatenate([band_data, nc.dem.values[None, ...]], axis=0)

    stats = {
        'fp': str(fp),
    }

    # add the stats for the band data
    n = np.sum(np.sum(~np.isnan(data), axis=-1), axis=-1)
    s = np.nansum(np.nansum(data, axis=-1), axis=-1)
    ssq = np.nansum(np.nansum(data ** 2, axis=-1), axis=-1)
    vmin = np.nanmin(np.nanmin(data, axis=-1), axis=-1)
    vmax = np.nanmax(np.nanmax(data, axis=-1), axis=-1)
    stats['n'] = n
    stats['sum_1'] = list(s)
    stats['sum_2'] = list(ssq)
    stats['vmin'] = list(vmin)
    stats['vmax'] = list(vmax)
    stats['var_name'] = [f'band_{i}' for i in range(len(band_data))] + ['dem']

    return stats


if __name__ == '__main__':
    data_dir = '../data/s2_data/patches_orig_r_128_s_128'
    fp_list = list(Path(data_dir).glob('**/*.nc'))
    print(f'len(fp_list) = {len(fp_list)}')

    all_df = []
    for fp in tqdm(fp_list):
        stats = compute_stats(fp)
        df = pd.DataFrame(stats)
        all_df.append(df)

    df_all = pd.concat(all_df)
    fp = Path(data_dir) / 'stats_all.csv'
    df_all.to_csv(fp, index=False)
    print(f'Stats saved to {fp}')

    # compute mean and standard deviation based only on the region_1
    df_all['region'] = df_all.fp.apply(lambda x: x.split('/region_')[1].split('/')[0])
    stats_agg = {k: [] for k in ['var_name', 'mu', 'stddev', 'vmin', 'vmax']}
    df_r1 = df_all[df_all.region == 'r1']
    for var_name in df_all.var_name.unique():
        df_r1_crt_var = df_r1[df_r1.var_name == var_name]
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
    fp = Path(data_dir) / 'stats_agg.csv'
    df_stats_agg.to_csv(fp, index=False)
    print(f'Aggregated stats saved to {fp}')

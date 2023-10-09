import shapely
import shapely.geometry
import geopandas as gpd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xarray as xr


def get_patches_gdf(nc, patch_radius, sampling_step=None, add_center=False, add_centroid=False, add_extremes=False):
    """
    Given a xarray dataset for one glacier, it returns a geopandas dataframe with the contours of square patches
    extracted from the dataset. The patches are only generated if they have the center pixel on glacier.

    :param nc: xarray dataset containing the S2 data and the glacier masks
    :param patch_radius: patch radius (in px)
    :param sampling_step: sampling step applied both on x and y (in px);
                          if smaller than 2 * patch_radius, then patches will overlap
    :param add_center: whether to add one patch centered in the middle of the glacier's box
    :param add_centroid: whether to add one patch centered in the centroid of the glacier
    :param add_extremes: whether to add four patches centered in the corners of the glacier
    :return: a geopandas dataframe with the contours of the generated patches
    """

    if sampling_step is not None:
        assert add_center or add_centroid

    # build a mask containing all feasible patch centers
    nc_full_crt_g_mask_center_sel = (nc.mask_crt_g.data == 1)
    nc_full_crt_g_mask_center_sel[:patch_radius, :] = False
    nc_full_crt_g_mask_center_sel[:, -patch_radius:] = False
    nc_full_crt_g_mask_center_sel[:, :patch_radius] = False
    nc_full_crt_g_mask_center_sel[-patch_radius:, :] = False

    # get all feasible centers
    all_y_centers, all_x_centers = np.where(nc_full_crt_g_mask_center_sel)
    minx = all_x_centers.min()
    miny = all_y_centers.min()
    maxx = all_x_centers.max()
    maxy = all_y_centers.max()

    if sampling_step is not None:
        # sample the feasible centers uniformly; ensure that the first and last feasible centers are always included
        idx_x = np.asarray([p % sampling_step == 0 for p in all_x_centers])
        idx_y = np.asarray([p % sampling_step == 0 for p in all_y_centers])
        idx = idx_x & idx_y
        x_centers = all_x_centers[idx]
        y_centers = all_y_centers[idx]

        # on top of the previously sampled centers, add also the extreme corners
        if add_extremes:
            left = (minx, int(np.mean(all_y_centers[all_x_centers == minx])))
            top = (int(np.mean(all_x_centers[all_y_centers == miny])), miny)
            right = (maxx, int(np.mean(all_y_centers[all_x_centers == maxx])))
            bottom = (int(np.mean(all_x_centers[all_y_centers == maxy])), maxy)
            x_centers = np.concatenate([x_centers, [left[0], top[0], right[0], bottom[0]]])
            y_centers = np.concatenate([y_centers, [left[1], top[1], right[1], bottom[1]]])
    else:
        x_centers = []
        y_centers = []

    if add_centroid:
        x_centers = np.concatenate([x_centers, [int(np.mean(all_x_centers))]]).astype(int)
        y_centers = np.concatenate([y_centers, [int(np.mean(all_y_centers))]]).astype(int)

    if add_center:
        x_centers = np.concatenate([x_centers, [int((minx + maxx) / 2)]]).astype(int)
        y_centers = np.concatenate([y_centers, [int((miny + maxy) / 2)]]).astype(int)

    # build a geopandas dataframe with the sampled patches
    all_patches = {k: [] for k in ['x_center', 'y_center', 'bounds_px', 'bounds_m', 'gl_cvg']}
    for x_center, y_center in zip(x_centers, y_centers):
        minx_patch, maxx_patch = x_center - patch_radius, x_center + patch_radius
        miny_patch, maxy_patch = y_center - patch_radius, y_center + patch_radius
        nc_crt_patch = nc.isel(x=slice(minx_patch, maxx_patch), y=slice(miny_patch, maxy_patch))

        # compute the fraction of pixels that cover the glacier
        g_fraction = float(np.sum((nc_crt_patch.mask_crt_g == 1))) / nc_crt_patch.mask_crt_g.size

        all_patches['x_center'].append(x_center)
        all_patches['y_center'].append(y_center)
        all_patches['bounds_px'].append((minx_patch, miny_patch, maxx_patch, maxy_patch))
        all_patches['bounds_m'].append(nc_crt_patch.rio.bounds())
        all_patches['gl_cvg'].append(g_fraction)

    patches_df = gpd.GeoDataFrame(all_patches)
    patches_df = gpd.GeoDataFrame(patches_df, geometry=patches_df.bounds_m.apply(lambda x: shapely.geometry.box(*x)))
    patches_df = patches_df.set_crs(nc.rio.crs)

    return patches_df


def data_cv_split(sdf, num_folds, valid_fraction, outlines_split_dir):
    """
    :param sdf: geopandas dataframe with the glacier contours
    :param num_folds: how many CV folds to generate
    :param valid_fraction: the percentage of each training fold to be used as validation
    :param outlines_split_dir: output directory where the outlines split will be exported
    :return:
    """

    # regional split, assuming W to E direction
    sdf['bound_lim'] = sdf.bounds.maxx
    sdf = sdf.sort_values('bound_lim')
    sdf['Area'] = sdf.area / 1e6

    split_lims = np.linspace(0, 1, num_folds + 1)
    split_lims[-1] += 1e-4  # to include the last glacier

    for i_split in range(num_folds):
        # first extract the test fold and the combined train & valid fold
        test_lims = (split_lims[i_split], split_lims[i_split + 1])
        area_cumsumf = sdf.Area.cumsum() / sdf.Area.sum()
        idx_test = (test_lims[0] <= area_cumsumf) & (area_cumsumf < test_lims[1])
        s2_df_test = sdf[idx_test]
        s2_df_train_valid = sdf[~idx_test]

        # extract the valid for train & valid fold
        area_cumsumf = s2_df_train_valid.Area.cumsum() / s2_df_train_valid.Area.sum()
        train_lims = (0.0, 1.0 - valid_fraction)
        idx_train = (train_lims[0] <= area_cumsumf) & (area_cumsumf < train_lims[1])
        s2_df_train = s2_df_train_valid[idx_train]
        s2_df_valid = s2_df_train_valid[~idx_train]

        df_per_fold = {
            'train': s2_df_train,
            'valid': s2_df_valid,
            'test': s2_df_test,
        }

        for crt_fold, df_crt_fold in df_per_fold.items():
            print(f'Extracting outlines for split = {i_split + 1} / {num_folds}, fold = {crt_fold}')
            outlines_split_fp = Path(outlines_split_dir) / f'split_{i_split + 1}' / f'fold_{crt_fold}.shp'
            outlines_split_fp.parent.mkdir(parents=True, exist_ok=True)
            df_crt_fold.to_file(outlines_split_fp)
            print(
                f'Exported {len(df_crt_fold)} glaciers '
                f'out of {len(sdf)} ({len(df_crt_fold) / len(sdf) * 100:.2f}%);'
                f' actual area percentage = {df_crt_fold.Area.sum() / sdf.Area.sum() * 100:.2f}%'
                f' ({df_crt_fold.Area.sum():.2f} km^2 from a total of {sdf.Area.sum():.2f} km^2)')


def patchify_s2_data(rasters_dir, outlines_split_dir, num_folds, patches_dir, patch_radius, sampling_step):
    """
    Using the get_patches_gdf function, it exports patches to disk for each cross-validation split, with each split
    separated into training-validation-test.
    When generating the patches, add_centroid and add_extremes will be set to True (see get_patches_gdf), which means
    at least five patches will be generated per glacier.

    :param rasters_dir: directory containing the raster netcdf files
    :param outlines_split_dir: directory containing the cross-validation splits
    :param num_folds: number of cross-validation folds
    :param patches_dir: output directory where the extracted patches will be saved
    :param patch_radius: patch radius (in px)
    :param sampling_step: sampling step applied both on x and y (in px);
                          if smaller than 2 * patch_radius, then patches will overlap
    :return:
    """
    fp_list = sorted(list((Path(rasters_dir).glob('**/*.nc'))))
    assert len(fp_list) > 0, f'No netcdf files found in {rasters_dir}'

    for i_split in range(1, num_folds + 1):
        for crt_fold in ['train', 'valid', 'test']:
            outlines_split_fp = Path(outlines_split_dir) / f'split_{i_split}' / f'fold_{crt_fold}.shp'
            s2_df_crt_fold = gpd.read_file(outlines_split_fp)

            fp_list_crt_fold = sorted(
                list(filter(lambda f: f.parent.name in set(s2_df_crt_fold.GLACIER_NR.astype(str)), fp_list)))

            for g_fp in tqdm(fp_list_crt_fold, desc=f'split = {i_split} / {num_folds}; fold = {crt_fold}'):
                gl_num = g_fp.parent.name
                nc = xr.open_dataset(g_fp, decode_coords='all')

                # get the locations of the sampled patches
                patches_df = get_patches_gdf(
                    nc=nc,
                    sampling_step=sampling_step,
                    patch_radius=patch_radius,
                    add_center=False,
                    add_centroid=True,
                    add_extremes=True
                )

                # build the patches
                for i in tqdm(range(len(patches_df)), desc=f'Exporting patches for glacier {gl_num}'):
                    patch_shp = patches_df.iloc[i:i + 1]
                    crt_s2_data = nc.rio.clip(patch_shp.geometry)

                    r = patch_shp.iloc[0]
                    fn = f'{gl_num}_patch_{i}_xc_{r.x_center}_yc_{r.y_center}.nc'

                    patch_fp = Path(patches_dir) / f'split_{i_split}' / f'fold_{crt_fold}'
                    patch_fp /= Path(*g_fp.parts[-3:-1]) / fn
                    patch_fp.parent.mkdir(parents=True, exist_ok=True)

                    crt_s2_data.to_netcdf(patch_fp)

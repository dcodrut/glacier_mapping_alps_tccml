import shapely
import shapely.geometry
import geopandas as gpd
import numpy as np


def get_hdt_glacier_patches(hdt, patch_radius, sampling_step=None, add_center=False, add_centroid=False):
    if sampling_step is not None:
        assert add_center or add_centroid

    # build a mask containing all feasible patch centers
    hdt_full_crt_g_mask_center_sel = (hdt.mask_crt_g.data == 1)
    hdt_full_crt_g_mask_center_sel[:patch_radius, :] = False
    hdt_full_crt_g_mask_center_sel[:, -patch_radius:] = False
    hdt_full_crt_g_mask_center_sel[:, :patch_radius] = False
    hdt_full_crt_g_mask_center_sel[-patch_radius:, :] = False

    # get all feasible centers
    all_y_centers, all_x_centers = np.where(hdt_full_crt_g_mask_center_sel)
    minx = all_x_centers.min()
    miny = all_y_centers.min()
    maxx = all_x_centers.max()
    maxy = all_y_centers.max()

    if sampling_step is not None:
        # sample the feasible centers uniformly; ensure that the first and last feasible centers are always included
        idx_x = np.asarray([p % sampling_step == 0 for p in all_x_centers])
        idx_y = np.asarray([p % sampling_step == 0 for p in all_y_centers])
        idx = idx_x & idx_y
        num_patches = np.sum(idx)
        if num_patches == 0:
            # for very small glaciers in can happen that no patch was found; take a single one in the middle
            add_centroid = True

        # on top of the previously sampled centers, add also the extreme corners
        left = (minx, int(np.mean(all_y_centers[all_x_centers == minx])))
        top = (int(np.mean(all_x_centers[all_y_centers == miny])), miny)
        right = (maxx, int(np.mean(all_y_centers[all_x_centers == maxx])))
        bottom = (int(np.mean(all_x_centers[all_y_centers == maxy])), maxy)
        x_centers = np.concatenate([all_x_centers[idx], [left[0], top[0], right[0], bottom[0]]])
        y_centers = np.concatenate([all_y_centers[idx], [left[1], top[1], right[1], bottom[1]]])
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
        hdt_crt_patch = hdt.isel(x=slice(minx_patch, maxx_patch), y=slice(miny_patch, maxy_patch))

        # compute the fraction of pixels that cover the glacier
        g_fraction = float(np.sum((hdt_crt_patch.mask_crt_g == 1))) / hdt_crt_patch.mask_crt_g.size

        all_patches['x_center'].append(x_center)
        all_patches['y_center'].append(y_center)
        all_patches['bounds_px'].append((minx_patch, miny_patch, maxx_patch, maxy_patch))
        all_patches['bounds_m'].append(hdt_crt_patch.rio.bounds())
        all_patches['gl_cvg'].append(g_fraction)

    patches_df = gpd.GeoDataFrame(all_patches)
    patches_df['geometry'] = patches_df.bounds_m.apply(lambda x: shapely.geometry.box(*x))
    patches_df = patches_df.set_crs(hdt.rio.crs)

    return patches_df

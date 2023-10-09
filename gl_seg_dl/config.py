# data root directory
DATA_DIR = '../data/external'


# ######################################################################################################################
# ######################################################## S2 ##########################################################
# ######################################################################################################################
class S2:
    DIR = f'{DATA_DIR}/s2_data'
    DIR_AUX_DATA = f'{DIR}/aux_data'
    DIR_OUTLINES_SPLIT = f'{DIR}/outlines_split'
    NUM_CV_FOLDS = 5
    VALID_FRACTION = 0.15
    DIR_GL_RASTERS_INV = f'{DIR}/rasters_orig'
    MIN_GLACIER_AREA = 0.1  # km2
    DIR_GL_RASTERS_2023 = f'{DIR}/rasters_2023'
    DIRS_INFER = [DIR_GL_RASTERS_INV, DIR_GL_RASTERS_2023]  # directories on which to make glacier-wide inferences
    NUM_CORES_EVAL = 16

    # patch sampling settings
    PATCH_RADIUS = 128
    SAMPLING_STEP = 128
    DIR_GL_PATCHES = f'{DIR}/patches_orig_r_{PATCH_RADIUS}_s_{SAMPLING_STEP}'

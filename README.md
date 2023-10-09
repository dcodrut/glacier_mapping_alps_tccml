## Detailed Glacier Area Change Analysis in the European Alps with Deep Learning

![](./data/gif/one_glacier_per_country.gif "Our approach exemplified for one glacier in each country")

### Set up the python environment
```shell
    conda env create -f environment.yml
    conda activate glsegenv
    cd gl_seg_dl
```

### Data downloading
1. Download the glacier outlines: `bash ./scripts/download_outlines.sh`
2. Download the glacier rasters: `TODO` (An FTP link will be made available after publication).

### Reproduce the results
#### Data processing
1. Cross-validation splits & patch sampling: `python main_data_prep.py`
2. Compute the mean & stddev of the training patches: `python main_compute_data_stats.py`

#### Model training & testing
1. Train the five models: `bash scripts/train.sh` (by default it runs on a single GPU; check the bash script for running on multiple GPUs)
2. Apply model on each glacier both on the inventory images and the 2023 ones: `bash scripts/infer.sh`
## Detailed Glacier Area Change Analysis in the European Alps with Deep Learning

![](./data/gif/one_glacier_per_country.gif "Our approach exemplified for one glacier in each country")

### Set up the python environment
1. `conda env create -f environment.yml`  
2. `conda activate glsegenv`  
3. `cd gl_seg_dl`

### Data preparation

1. Download the glacier outlines: `bash ./scripts/download_outlines.sh`
2. Download the glacier rasters: `TODO` (An FTP link will be made available after publication.)
3. Data split & patch sampling: `python main_data_prep.py`

### Reproduce the results
1. Train the 5 five models: `bash scripts/train.sh` (by default it runs on a single GPU; check the bash script for running on multiple GPUs)

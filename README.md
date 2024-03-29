## Detailed Glacier Area Change Analysis in the European Alps with Deep Learning
#### @NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning
#### For the paper and presentation please check the [project page](https://www.climatechange.ai/papers/neurips2023/45).

![](./data/gif/one_glacier_per_country.gif "Our approach exemplified for one glacier in each country")

### Set up the python environment
```shell
    conda env create -f environment.yml
    conda activate glsegenv
    cd gl_seg_dl
```

### Reproduce the results
#### Data downloading:
1. Download the glacier outlines: `bash ./scripts/download_outlines.sh`
2. Download the glacier rasters (stored here: https://huggingface.co/datasets/dcodrut/glacier_mapping_alps): `bash ./scripts/download_data.sh`  
   The archived rasters have ~3Gb for each year (inventory one & 2023). After extracting the NetCDF rasters, we will need 20Gb for each year.

#### Data processing:
1. Cross-validation splits & patch sampling: `python main_data_prep.py`
2. Compute the mean & stddev of the training patches: `python main_compute_data_stats.py`

#### Model training, testing and area estimation:
1. Train the five models: `bash scripts/train.sh` (by default it runs on a single GPU; check the bash script for running on multiple GPUs)
2. Apply model on each glacier both on the inventory images and the 2023 ones: `bash scripts/infer.sh`
3. Estimate the areas based on the predictions from the previous step: `bash scripts/eval.sh` 
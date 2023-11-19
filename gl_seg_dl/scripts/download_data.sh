mkdir -p ../data/external/s2_data/
wget -N https://huggingface.co/datasets/dcodrut/glacier_mapping_alps/resolve/main/rasters_orig.tar.gz -P ../data/external/s2_data/
wget -N https://huggingface.co/datasets/dcodrut/glacier_mapping_alps/resolve/main/rasters_2023.tar.gz -P ../data/external/s2_data/
tar -xvzf ../data/external/s2_data/rasters_orig.tar.gz -C ../data/external/s2_data/
tar -xvzf ../data/external/s2_data/rasters_2023.tar.gz -C ../data/external/s2_data/
printf "\nFinished extracting the rasters to data/external/s2_data\n"

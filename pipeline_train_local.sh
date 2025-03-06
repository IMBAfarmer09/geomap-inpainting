#!/bin/bash
set -x

cd /home/zj/Programs/python_workspace/lama || { echo "directory not exist: /home/zj/Programs/python_workspace/lama"; exit 1; }

# env config
source /home/zj/miniconda/etc/profile.d/conda.sh
conda activate lama

export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# unzip data
mkdir geomap-hq-60k/
unzip data256x256.zip -d geomap-hq-60k/

# Reindex
for i in `echo {00001..61090}`
do
    mv 'geomap-hq-60k/data256x256/'$i'.jpg' 'geomap-hq-60k/data256x256/'$[10#$i - 1]'.jpg'
done

ls geomap-hq-60k/data256x256/ | shuf > geomap-hq-60k/temp_all_shuffled.flist

head -n 2048 geomap-hq-60k/temp_all_shuffled.flist > geomap-hq-60k/val_shuffled.flist
tail -n +2049 geomap-hq-60k/temp_all_shuffled.flist | head -n 2048 > geomap-hq-60k/visual_test_shuffled.flist
tail -n +4097 geomap-hq-60k/temp_all_shuffled.flist | head -n 55000 > geomap-hq-60k/train_shuffled.flist

mkdir geomap-hq-60k/train_256/
mkdir geomap-hq-60k/val_source_256/
mkdir geomap-hq-60k/visual_test_source_256/

cat geomap-hq-60k/train_shuffled.flist | xargs -I {} mv geomap-hq-60k/data256x256/{} geomap-hq-60k/train_256/
cat geomap-hq-60k/val_shuffled.flist | xargs -I {} mv geomap-hq-60k/data256x256/{} geomap-hq-60k/val_source_256/
cat geomap-hq-60k/visual_test_shuffled.flist | xargs -I {} mv geomap-hq-60k/data256x256/{} geomap-hq-60k/visual_test_source_256/

# create location config geomap.yaml
PWD=$(pwd)
DATASET=${PWD}/geomap-hq-60k
GEOMAP=${PWD}/configs/training/location/geomap.yaml

touch $GEOMAP
echo "# @package _group_" >> $GEOMAP
echo "data_root_dir: ${DATASET}/" >> $GEOMAP
echo "out_root_dir: ${PWD}/experiments/" >> $GEOMAP
echo "tb_dir: ${PWD}/tb_logs/" >> $GEOMAP
echo "pretrained_models: ${PWD}/" >> $GEOMAP

# generate mask
python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
geomap-hq-60k/val_source_256/ \
geomap-hq-60k/val_256/random_thick_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_256.yaml \
geomap-hq-60k/val_source_256/ \
geomap-hq-60k/val_256/random_thin_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_256.yaml \
geomap-hq-60k/val_source_256/ \
geomap-hq-60k/val_256/random_medium_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
geomap-hq-60k/visual_test_source_256/ \
geomap-hq-60k/visual_test_256/random_thick_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_256.yaml \
geomap-hq-60k/visual_test_source_256/ \
geomap-hq-60k/visual_test_256/random_thin_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_256.yaml \
geomap-hq-60k/visual_test_source_256/ \
geomap-hq-60k/visual_test_256/random_medium_256/

# start training
python -u bin/train.py -cn big-lama-geomap data.batch_size=15 +trainer.kwargs.resume_from_checkpoint=big-lama-finetune/big-lama-with-discr-remove-loss_segm_pl.ckpt

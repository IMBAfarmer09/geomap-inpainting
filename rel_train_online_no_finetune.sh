#!/bin/bash
set -x

cd /mnt/tqnas/zhijiang/home/yuansy/lama || { echo "directory not exist: /mnt/tqnas/zhijiang/home/yuansy/lama"; exit 1; }

#export TRAINING_PARENT_WORK_DIR=$(pwd)
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export USER=$(whoami)

# create location config geomap.yaml
PWD=$(pwd)
DATASET=${PWD}/geomap-hq-60k
GEOMAP=${PWD}/configs/training/location/geomap.yaml

if [ -f $GEOMAP ]; then
    rm $GEOMAP
fi

touch $GEOMAP
echo "# @package _group_" >> $GEOMAP
echo "data_root_dir: ${DATASET}/" >> $GEOMAP
echo "out_root_dir: ${PWD}/experiments/" >> $GEOMAP
echo "tb_dir: ${PWD}/tf_logs/" >> $GEOMAP
echo "pretrained_models: ${PWD}/" >> $GEOMAP

# start training
python -u bin/train.py -cn big-lama-geomap data.batch_size=30

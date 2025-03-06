#!/bin/bash
set -x

cd /root/lama || { echo "directory not exist: /root/lama"; exit 1; }

export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export USER=root

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
echo "tb_dir: ${PWD}/tb_logs/" >> $GEOMAP
echo "pretrained_models: ${PWD}/" >> $GEOMAP

# start training
python -u bin/train.py -cn big-lama-geomap data.batch_size=30 +trainer.kwargs.resume_from_checkpoint=big-lama-finetune/big-lama-with-discr-remove-loss_segm_pl.ckpt



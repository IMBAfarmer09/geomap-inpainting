#!/bin/bash
set -x
#cd /mnt/tqnas/zhijiang/home/yzw/geomap-inpainting || { echo "directory not exist: /mnt/tqnas/zhijiang/home/yuansy/lama"; exit 1; }
cd /home/zj/Music/git-geomap/geomap-inpainting || { echo "directory not exist: /mnt/tqnas/zhijiang/home/yuansy/lama"; exit 1; }

#export TRAINING_PARENT_WORK_DIR=$(pwd)
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export USER=$(whoami)

# create location config geomap.yaml
PWD=$(pwd)
DATASET=${PWD}/dataset_inpainting
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
python -u bin/train.py -cn big-lama-geomap data.batch_size=8 location.out_root_dir=/home/zj/Music/git-geomap/geomap-inpainting/experiments/testoutdir +trainer.kwargs.resume_from_checkpoint=/home/zj/Music/git-geomap/geomap-inpainting/big-lama-finetune/big-lama-with-discr-remove-loss_segm_pl.ckpt

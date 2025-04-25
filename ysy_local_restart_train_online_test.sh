#!/bin/bash
set -x
#cd /mnt/tqnas/zhijiang/home/yzw/geomap-inpainting || { echo "directory not exist: /mnt/tqnas/zhijiang/home/yuansy/lama"; exit 1; }
cd /home/zj/Music/git-ysy-geomap-inpainting/geomap-inpainting || { echo "directory not exist: /home/zj/Music/git-ysy-geomap-inpainting/geomap-inpainting"; exit 1; }

# env config
source /home/zj/miniconda/etc/profile.d/conda.sh
conda activate lama

#export TRAINING_PARENT_WORK_DIR=$(pwd)
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export USER=$(whoami)

# create location config geomap.yaml
PWD=$(pwd)
DATASET=${PWD}/dataset_inpainting
GEOMAP=${PWD}/configs/training/location/geomap.yaml
OUT_ROOT=${PWD}/experiments/20250424_1111_test  # <-- 提前定义输出目录变量

# 重新生成 geomap.yaml
if [ -f "$GEOMAP" ]; then
    rm "$GEOMAP"
fi
cat << EOF > "$GEOMAP"
# @package _group_
data_root_dir: ${DATASET}/
out_root_dir: ${OUT_ROOT}
tb_dir: ${PWD}/tb_logs/
pretrained_models: ${PWD}/
EOF

# —— 启动训练前的检查 —— #
if [ ! -d "$OUT_ROOT" ]; then
    echo "目录 $OUT_ROOT 不存在，首次训练，使用默认命令启动"
    python -u bin/train.py \
        -cn big-lama-geomap \
        data.batch_size=2 \
        hydra.run.dir=${OUT_ROOT} \
        losses.l1.weight_missing=10 \
        losses.l1.weight_known=10 \
        losses.adversarial.weight=10 \
        losses.feature_matching.weight=100 \
        losses.resnet_pl.weight=30 \
        losses.histogram.weight=50 \
        losses.histogram.n_bins=64 \
        losses.freeze_discriminator=true \
        +trainer.kwargs.resume_from_checkpoint=/home/zj/Music/git-ysy-geomap-inpainting/geomap-inpainting/big-lama-finetune/big-lama-with-discr-remove-loss_segm_pl.ckpt
else
    echo "目录 $OUT_ROOT 已存在，检测到上次训练残留，使用 last.ckpt 恢复训练"
    python -u bin/train.py \
        -cn big-lama-geomap \
        data.batch_size=2 \
        hydra.run.dir=${OUT_ROOT} \
        losses.l1.weight_missing=10 \
        losses.l1.weight_known=10 \
        losses.adversarial.weight=10 \
        losses.feature_matching.weight=100 \
        losses.resnet_pl.weight=30 \
        losses.histogram.weight=50 \
        losses.histogram.n_bins=64 \
        losses.freeze_discriminator=true \
        +trainer.kwargs.resume_from_checkpoint=${OUT_ROOT}/models/last.ckpt
fi

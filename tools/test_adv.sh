#!/usr/bin/env bash

# check parmas num
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 MODEL DATASET"
    exit 1
fi

MODEL=$1
DATASET=$2

# setting path
CFG_PATH="setting.json"

# get settings by jq
MODEL_CFG_PREFIX=$(jq -r ".FILE_CFG.MODEL_CFG_PREFIX.${MODEL}" "$CFG_PATH")
DATASET_SUFFIX=$(jq -r ".FILE_CFG.DATASET_SUFFIX.${DATASET}" "$CFG_PATH")
CKPT_FILE_PREFIX=$(jq -r ".FILE_CFG.CKPT_FILE_PREFIX.${MODEL}" "$CFG_PATH")

model_config_path="${MODEL_CFG_PREFIX}${DATASET_SUFFIX}.py"
model_ckpt_path="${CKPT_FILE_PREFIX}${DATASET_SUFFIX}.pth"

# echo "Model Config Path: $model_config_path"
# echo "Model Checkpoint Path: $model_ckpt_path"

# excute test
python $(dirname "$0")/test.py \
    $CONFIG_PATH \
    $CKPT_PATH \

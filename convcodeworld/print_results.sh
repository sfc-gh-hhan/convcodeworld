#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ConvCodeWorld

OPTION=$1
MODEL_NAME=$2
SAVE_DIR=$3

if [ -z "${SAVE_DIR}" ]; then
  SAVE_DIR="results"
fi

python print_results.py --option $OPTION --model_name $MODEL_NAME --save_dir $SAVE_DIR

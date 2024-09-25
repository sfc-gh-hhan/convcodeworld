#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ConvCodeWorld

OPTION=$1
MODEL_NAME=$2
SAVE_DIR=$3
REF_MODEL_NAME=$4

if [ -z "${SAVE_DIR}" ]; then
  SAVE_DIR="results"
fi
if [ -z "${REF_MODEL_NAME}" ]; then
  REF_MODEL_NAME="codellama/CodeLlama-7b-Instruct-hf"
fi


python print_results.py --option $OPTION --model_name $MODEL_NAME --save_dir $SAVE_DIR --ref_model_name $REF_MODEL_NAME

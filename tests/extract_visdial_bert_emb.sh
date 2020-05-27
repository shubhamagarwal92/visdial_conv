#!/usr/bin/env bash

source activate visdialch

export CURRENT_DIR=${PWD} # data
export PARENT_DIR="$(dirname "$CURRENT_DIR")" # visdialch/visdialch
export CODE_DIR="$(dirname "$PARENT_DIR")" # visdialch
export PROJECT_DIR="$(dirname "$CODE_DIR")" # visdial --> data
export DATA_DIR=$PROJECT_DIR/data
export SAVE_DATA_DIR=$PROJECT_DIR/data



export GPU_ID="1 2 3"
echo "Running on: " $GPU_ID

# Try local
python extract_visdial_bert_emb.py \
-d $DATA_DIR \
-s $SAVE_DATA_DIR \
-g $GPU_ID

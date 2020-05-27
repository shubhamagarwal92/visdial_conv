#!/usr/bin/env bash

source activate visdialch

# Common paths
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR
export CODE_DIR=$PARENT_DIR
export CONFIG_DIR=$CODE_DIR/configs
export PROJECT_DIR="$(dirname "$PARENT_DIR")"

## SA: todo if config also from
export CONFIG_YML=$CONFIG_DIR/rva.yml
export DATA_DIR=$PROJECT_DIR/data

echo $DATA_DIR
PYTHONPATH=. python visdialch/data/init_glove.py \
--config-yml $CONFIG_YML \
--pretrained-txt  $DATA_DIR/glove.6B.300d.txt \
--data_dir $DATA_DIR \
--save-npy $DATA_DIR/glove.npy


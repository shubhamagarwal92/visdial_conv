#!/usr/bin/env bash

source activate visdialch

# Common paths
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR
export CODE_DIR=$PARENT_DIR
export CONFIG_DIR=$CODE_DIR/configs
export PROJECT_DIR="$(dirname "$PARENT_DIR")"

export DATA_DIR=$PROJECT_DIR/data/


python data/augment_dense_annotations.py \
-d $DATA_DIR/visdial_1.0_train.json \
-a $DATA_DIR/visdial_1.0_train_dense_annotations.json \
-s $DATA_DIR/visdial_1.0_train_augmented_dense_annotations.json

echo "Saved in: " $DATA_DIR

#!/usr/bin/env bash

source activate visdialch

# Common paths
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR
export CODE_DIR=$PARENT_DIR
export CONFIG_DIR=$CODE_DIR/configs
export PROJECT_DIR="$(dirname "$PARENT_DIR")"


export DATA_DIR=$PROJECT_DIR/data
export SAVE_DATA_DIR=$PROJECT_DIR/data/data_analysis

mkdir -p $SAVE_DATA_DIR

echo "Saving data in: " $SAVE_DATA_DIR

python data_analysis/validate_heurisitc_on_amt.py \
-d $SAVE_DATA_DIR \
-s $SAVE_DATA_DIR

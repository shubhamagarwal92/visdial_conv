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

echo "Saving data in: " $DATA_DIR
read -p "What type of new annotations: (1: gt_1 (default), 0: uniform): " ANN_TYPE
ANN_TYPE=${ANN_TYPE:-1}


if [ $ANN_TYPE == 1 ]; then
    ANN_TYPE="gt_1"
else
    ANN_TYPE="uniform"
fi


python new_annotations/fix_dense_annotations.py \
-d $DATA_DIR \
-s $DATA_DIR \
-t $ANN_TYPE

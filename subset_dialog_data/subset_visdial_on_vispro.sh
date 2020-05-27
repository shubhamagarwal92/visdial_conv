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
export SAVE_DATA_DIR=$PROJECT_DIR/data/vispro

mkdir -p $SAVE_DATA_DIR

echo "Saving vispro image ids data in: " $SAVE_DATA_DIR

cd $SAVE_DATA_DIR
 wget https://raw.githubusercontent.com/HKUST-KnowComp/Visual_PCR/master/data/val.vispro.1.1.jsonlines
 wget https://raw.githubusercontent.com/HKUST-KnowComp/Visual_PCR/master/data/train.vispro.1.1.jsonlines
 wget https://raw.githubusercontent.com/HKUST-KnowComp/Visual_PCR/master/data/test.vispro.1.1.jsonlines

cd $PARENT_DIR
python subset_dialog_data/get_vispro_image_ids.py \
-d $SAVE_DATA_DIR \
-s $SAVE_DATA_DIR


python subset_dialog_data/subset_visdial_on_image_ids.py \
-d $DATA_DIR \
-s $SAVE_DATA_DIR \
-i $SAVE_DATA_DIR/vispro_image_ids.txt \
-t "vispro"

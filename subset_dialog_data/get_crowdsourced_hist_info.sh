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
export SAVE_DATA_DIR=$PROJECT_DIR/data/crowdsourced

mkdir -p $SAVE_DATA_DIR
echo "Saving crowdsourced image ids data in: " $SAVE_DATA_DIR

cd $SAVE_DATA_DIR
wget https://www.dropbox.com/s/omgxbxio2726iag/visdial_img_ids_hist_info_batch_1_5.txt?dl=0 -O crowdsourced_image_ids.txt


cd $PARENT_DIR
python subset_dialog_data/subset_visdial_on_image_ids.py \
-d $DATA_DIR \
-s $SAVE_DATA_DIR \
-i $SAVE_DATA_DIR/crowdsourced_image_ids.txt \
-t "crowdsourced"

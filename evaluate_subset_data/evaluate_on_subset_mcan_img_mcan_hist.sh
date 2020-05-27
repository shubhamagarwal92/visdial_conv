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
export CONFIG_YML=$CONFIG_DIR/mcan_img_mcan_hist.yml

export DATA_DIR=$PROJECT_DIR/data
export MODEL_DIR=$PROJECT_DIR/models/hwu_new_models/
read -p "Enter the GPU id (as 0/1/2):  " GPU_ID
read -p "Enter the model name:  " MODEL_NAME
export MODEL_NAME=${MODEL_NAME:-mcan_img_mcan_hist}


export SAVE_MODEL_DIR=$MODEL_DIR/$MODEL_NAME
echo "Model saved in: " $SAVE_MODEL_DIR

GPU_ID=${GPU_ID:-"0 1 2 3"}
echo "Running on gpus : " $GPU_ID


# SA: TODO check the name of directory in subset_dialog_data/get_crowdsourced_hist_info.sh

read -p "Enter subset type as 0/1 (0: vispro ; 1: crowdsourced (default) ): " SUBSET_TYPE
SUBSET_TYPE=${SUBSET_TYPE:-1}

if [ $SUBSET_TYPE == 0 ]; then
export SUBSET_TYPE="vispro"
else
export SUBSET_TYPE="crowdsourced"
fi



export SUBSET_DATA_DIR=$PROJECT_DIR/data/$SUBSET_TYPE
export SUBSET_DATA_TYPE=_$SUBSET_TYPE
echo "Evaluating on subset: " $SUBSET_DATA_TYPE


read -p "Enter the test checkpoint number: " CHECKPOINT_TEST_NUM
CHECKPOINT_TEST_NUM=${CHECKPOINT_TEST_NUM:-best_ndcg}
export CHECKPOINT_TEST_PATH=$SAVE_MODEL_DIR/checkpoint_${CHECKPOINT_TEST_NUM}.pth

read -p "Enter split type as (val (default) or test): " SPLIT
SPLIT=${SPLIT:-"val"}

export RANKS_PATH=$SAVE_MODEL_DIR/ranks_${SPLIT}_${CHECKPOINT_TEST_NUM}${SUBSET_DATA_TYPE}.json
export LOG_PATH=$SAVE_MODEL_DIR/evaluate_${SPLIT}_${CHECKPOINT_TEST_NUM}${SUBSET_DATA_TYPE}.log

CURRENT_DATE=$(date)
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME $CURRENT_DATE"

python evaluate.py \
--val-json $SUBSET_DATA_DIR/visdial_1.0_val${SUBSET_DATA_TYPE}.json \
--val-dense-json $SUBSET_DATA_DIR/visdial_1.0_val_dense_annotations${SUBSET_DATA_TYPE}.json \
--test-json $DATA_DIR/visdial_1.0_test.json \
--config-yml $CONFIG_YML \
--load-pthpath $CHECKPOINT_TEST_PATH \
--split $SPLIT \
--save-ranks-path $RANKS_PATH \
--save-dirpath $SAVE_MODEL_DIR \
--data_dir $DATA_DIR \
--gpu-ids $GPU_ID >> $LOG_PATH


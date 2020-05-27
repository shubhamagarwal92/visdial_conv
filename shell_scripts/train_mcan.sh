#!/usr/bin/env bash

# Common paths
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR
export CODE_DIR=$PARENT_DIR
export CONFIG_DIR=$CODE_DIR/configs
export PROJECT_DIR="$(dirname "$PARENT_DIR")"

# Sensei
if [ $1 == 2 ];
then
    export DATA_DIR=$2
    export MODEL_DIR=$3
#    export MODEL_NAME=$4
    export GPU_ID=$5
    SERVER=2
    OUT_STRING="sensei"

# Condor
elif [ $1 == 1 ];
then
    export DATA_DIR=$PROJECT_DIR/data
    #export DATA_DIR=/mnt/ilcompfad0/user/shubhama/projects/visdial_adobe/data
    export MODEL_DIR=$PROJECT_DIR/models
    export GPU_ID="1 2 3"
#    export MODEL_NAME=vqa
    SERVER=1
    OUT_STRING="ilcomp1"

# ilcomp0/local
else

    export DATA_DIR=$PROJECT_DIR/data
    #export DATA_DIR=/mnt/ilcompfad0/user/shubhama/projects/visdial_adobe/data
    export MODEL_DIR=$PROJECT_DIR/models/final_models
#    read -p "Enter the model type used for directory (such as date):  " MODEL_NAME
    read -p "Enter the GPU id (as 0/1/2):  " GPU_ID

    SERVER=0
    OUT_STRING="ilcompm0"
#    export GPU_ID="1 2 3"
#    conda activate visdialch
    source activate visdialch

fi


# Is Using history?
#export USE_HIST=$1 # Take input from shell
read -p "Using history?? 1:Yes (Default 0):  " USE_HIST
USE_HIST=${USE_HIST:-0} # Default not using history
if [ $USE_HIST == 1 ];
then
    echo "Using history"
    export CONFIG_YML=$CONFIG_DIR/mcan.yml
    export MODEL_NAME=mcan_using_hist
else
    echo "Using history is set to false"
    export CONFIG_YML=$CONFIG_DIR/mcan_img_only.yml
    export MODEL_NAME=mcan_img_only
fi


echo "Running on server: " $OUT_STRING

export SAVE_MODEL_DIR=$MODEL_DIR/$MODEL_NAME
mkdir -p $SAVE_MODEL_DIR
echo "Model saved in: " $SAVE_MODEL_DIR

echo "Running on gpus : " $GPU_ID


## SA: todo checkpointing for all
read -p "Enter the checkpoint number: " CHECKPOINT_NUM
CHECKPOINT_NUM=${CHECKPOINT_NUM:-4}
export CHECKPOINT_PATH=$SAVE_MODEL_DIR/checkpoint_${CHECKPOINT_NUM}.pth


if [ $SERVER == 2 ] || [ $SERVER == 1 ] ;
then
## sensei or condor
echo "Training on sensei/condor"
/opt/conda/envs/visdialch/bin/python train.py \
--train-json $DATA_DIR/visdial_1.0_train.json \
--val-json $DATA_DIR/visdial_1.0_val.json \
--val-dense-json $DATA_DIR/visdial_1.0_val_dense_annotations.json \
--save-dirpath $SAVE_MODEL_DIR \
--config-yml $CONFIG_YML \
--validate \
--data_dir $DATA_DIR \
--in-memory \
--cpu-workers 8 \
--load-pthpath $CHECKPOINT_PATH \
--gpu-ids $GPU_ID >> $SAVE_MODEL_DIR/train_logs.txt # provide more ids for multi-GPU execution other args...

echo "Not in memory"

else
# local or ilcompm0
echo "Training on local"
echo "In memory"


CHECKPOINT_NUM=${CHECKPOINT_NUM:-4}
export CHECKPOINT_PATH=$SAVE_MODEL_DIR/checkpoint_${CHECKPOINT_NUM}.pth


echo "Loading checkpoint..please remove"
python train.py \
--train-json $DATA_DIR/visdial_1.0_train.json \
--val-json $DATA_DIR/visdial_1.0_val.json \
--val-dense-json $DATA_DIR/visdial_1.0_val_dense_annotations.json \
--save-dirpath $SAVE_MODEL_DIR \
--config-yml $CONFIG_YML \
--validate \
--data_dir $DATA_DIR \
--in-memory \
--cpu-workers 8 \
--load-pthpath $CHECKPOINT_PATH \
--gpu-ids $GPU_ID >> $SAVE_MODEL_DIR/train_logs.txt # provide more ids for multi-GPU execution other args...

fi




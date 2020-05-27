#!/usr/bin/env bash

source activate visdialch

export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
export PROJECT_DIR="$(dirname "$PARENT_DIR")"
export DATA_DIR=$PROJECT_DIR/data
export SAVE_DATA_DIR=$DATA_DIR/data_dump

mkdir -p $SAVE_DATA_DIR
echo "Data saved in: " $SAVE_DATA_DIR

export LOGS_FILE=$SAVE_DATA_DIR/data_analysis_logs.txt
rm $LOGS_FILE


# Try local
python extract_text_data.py \
-d $DATA_DIR \
-s $SAVE_DATA_DIR >> $LOGS_FILE

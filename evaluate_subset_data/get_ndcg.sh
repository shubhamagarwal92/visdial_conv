#!/usr/bin/env bash


source activate visdialch

# Common paths
export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR

export PROJECT_DIR="$(dirname "$PARENT_DIR")"

export DATA_DIR=$PROJECT_DIR/data
export MODEL_PRED_ROOT=$PROJECT_DIR/models/acl_val_set/

echo "All models in: " $MODEL_PRED_ROOT

CURRENT_DATE=$(date)
CURRENT_TIME=$(date +"%T")
echo "Current time : $CURRENT_TIME $CURRENT_DATE"


read -p "Is complement 0: Not complement, 1:Yes ? " IS_COMPLEMENT

if [ $IS_COMPLEMENT == 0 ]
then

    read -p "Enter subset type as 0/1 (0: none (whole set); 1: crowdsourced (default); 2: vispro): " SUBSET_TYPE
    SUBSET_TYPE=${SUBSET_TYPE:-1}

    if [ $SUBSET_TYPE == 0 ]
    then
        export SUBSET_TYPE="" # whole val set
    elif [ $SUBSET_TYPE == 1 ]
    then
        export SUBSET_TYPE="_crowdsourced"
    else
        export SUBSET_TYPE="_vispro"
    fi


    export SUBSET_DATA_STR=${SUBSET_TYPE#"_"}
    echo $SUBSET_DATA_STR
    export SUBSET_DATA_DIR=$DATA_DIR/$SUBSET_DATA_STR
    echo "Evaluating on subset: " $SUBSET_TYPE


    export ANN_JSON=$SUBSET_DATA_DIR/visdial_1.0_val_dense_annotations${SUBSET_TYPE}.json
    echo "Using annotations: " $ANN_JSON


    # SA: TODO find better way
    # avoid error: argument -s/--subset_type: expected one argument
    # https://unix.stackexchange.com/questions/496843/passing-generated-empty-strings-as-command-line-arguments
    if [ -z "$SUBSET_TYPE" ]
    then
    # whole val set
        PYTHONPATH=. python evaluate_subset_data/get_ndcg.py \
        -m $MODEL_PRED_ROOT \
        -a $ANN_JSON

        PYTHONPATH=. python evaluate_subset_data/significance_test.py \
        -m $MODEL_PRED_ROOT

    else
        PYTHONPATH=. python evaluate_subset_data/get_ndcg.py \
        -m $MODEL_PRED_ROOT \
        -a $ANN_JSON \
        -s $(SUBSET_TYPE)

        PYTHONPATH=. python evaluate_subset_data/significance_test.py \
        -m $MODEL_PRED_ROOT \
        -s $SUBSET_TYPE
    fi

else
    export SUBSET_TYPE="_complement"
    PYTHONPATH=. python evaluate_subset_data/ndcg_complement_ranks.py \
    -m $MODEL_PRED_ROOT \
    -a $DATA_DIR/visdial_1.0_val_dense_annotations.json \
    -s $SUBSET_TYPE

    PYTHONPATH=. python evaluate_subset_data/significance_test.py \
    -m $MODEL_PRED_ROOT \
    -s $SUBSET_TYPE
fi

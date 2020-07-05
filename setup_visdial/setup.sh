#!/usr/bin/env bash

export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
cd $PARENT_DIR
export CODE_DIR=$PARENT_DIR
export CONFIG_DIR=$CODE_DIR/configs
export PROJECT_DIR="$(dirname "$PARENT_DIR")"
export DATA_DIR=$PROJECT_DIR/data
export MODEL_DIR=$PROJECT_DIR/models
mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR

echo "Data directory created at: " "${DATA_DIR}"
echo "Model directory created at: " "${MODEL_DIR}"
# Conda env and setup
 conda create -n visdialch python=3.6
 conda activate visdialch
 pip install -r requirements.txt
 python -c "import nltk; nltk.download('punkt')";
## If issues related to RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED for rnn
## pip install -U torch==1.0.1 -f https://download.pytorch.org/whl/cu100/stable
## or
## conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
## pip install -U numpy


cd $DATA_DIR

## Uncomment to download visdial2019 data
## Dialogue data
wget https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip
unzip visdial_1.0_train.zip && rm visdial_1.0_train.zip
wget https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip
unzip visdial_1.0_val.zip && rm visdial_1.0_val.zip
wget https://www.dropbox.com/s/o7mucbre2zm7i5n/visdial_1.0_test.zip
unzip visdial_1.0_test.zip && rm visdial_1.0_test.zip
wget https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json
# vocab
wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json

# Faster RCNN features
wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5
wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5
wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5

# Glove embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.100d.txt glove.6B.200d.txt glove.6B.50d.txt glove.6B.zip

# Dense annotation on training set
wget https://www.dropbox.com/s/1ajjfpepzyt3q4m/visdial_1.0_train_dense_sample.json?dl=0 -O visdial_1.0_train_dense_annotations.json

#
## VGG FC7 features
#wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_train.h5
#wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_val.h5
#wget https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_test.h5
#
#echo "Downloading pool5 features"
## Pool5 features
# wget https://computing.ece.vt.edu/~abhshkdz/visdial/data/v1.0/data_img_vgg16_pool5_train.h5
# wget https://computing.ece.vt.edu/~abhshkdz/visdial/data/v1.0/data_img_vgg16_pool5_val.h5
# wget https://computing.ece.vt.edu/~abhshkdz/visdial/data/v1.0/data_img_vgg16_pool5_test.h5



# Downloading raw images:
# Train:
# wget http://images.cocodataset.org/zips/train2014.zip
# wget http://images.cocodataset.org/zips/val2014.zip

# # Valid:
# wget https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=0 -O VisualDialog_val2018.zip
# # SA: try this if it doesnt work
# #wget https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=1

# # Test:
# wget https://www.dropbox.com/s/mwlrg31hx0430mt/VisualDialog_test2018.zip?dl=0 -O VisualDialog_test2018.zip




# Vispro dataset
# wget https://raw.githubusercontent.com/HKUST-KnowComp/Visual_PCR/master/data/val.vispro.1.1.jsonlines
# wget https://raw.githubusercontent.com/HKUST-KnowComp/Visual_PCR/master/data/train.vispro.1.1.jsonlines
# wget https://raw.githubusercontent.com/HKUST-KnowComp/Visual_PCR/master/data/test.vispro.1.1.jsonlines

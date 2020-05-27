#!/usr/bin/env bash

export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
echo "File is in: " $CURRENT_DIR


export FileNameWithExt=${0##*/}
export FILE_NAME=${FileNameWithExt%.*}

echo $FILE_NAME


# https://stackoverflow.com/questions/965053/extract-filename-and-extension-in-bash/965072
filename=$(basename -- "$fullfile")
extension="${filename##*.}"
filename="${filename%.*}"


# SA: todo split using underscore and forget the "train" part

# This give extension
my_name=$(basename -- "$0")
echo $my_name



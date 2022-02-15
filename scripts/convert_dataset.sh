#! /bin/bash

#######################################################################
#Description    : Converts between different dataset formats using datumaro
#Args           : Experiment directory, input-format, export-format
#Author         : Joshua Hoogstraat
#Date           : 14.01.2021
#######################################################################

## IMPORTANT ##
# 1. Requires python venv to be active (source .venv/bin/activate).
# 2. Input/Export format is used as directories names aswell!
# VOC: A custom labels file is required as the labels are usually not the Pascal Visual Object Classes.
# VoTT-json: Path entries in the json-file should be relative, not absolute.
###############

# Some input/output formats (See datumaro docs for more):
# - vott_json (https://openvinotoolkit.github.io/datumaro/docs/formats/vott_json/)
# - coco_instances (https://openvinotoolkit.github.io/datumaro/docs/formats/coco/)
# - voc_detection (https://openvinotoolkit.github.io/datumaro/docs/formats/pascal_voc/)
# - kitti_detection (https://openvinotoolkit.github.io/datumaro/docs/formats/kitti/)

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 PROJECT INPUT_FORMAT OUTPUT_FORMAT" >&2
  exit 1
fi

# Check if input dataset exists
if ! [ -d "projects/$1/data/$2" ]; then
    echo "projects/$1/data/$2 not a directory" >&2
    exit 1
fi

datum convert -i projects/$1/data/$2/ \
              -if $2 \
              -o projects/$1/data/$3/ \
              -f $3 \
              --overwrite \
              -- --save-images

if [ "$3" = kitti_detection ]; then
    IMG_COUNT=$(ls projects/$1/data/$3/train/image_2 | wc -l)
    LBL_COUNT=$(ls projects/$1/data/$3/train/label_2 | wc -l)
    echo "Images: $IMG_COUNT; Labels: $LBL_COUNT"
fi
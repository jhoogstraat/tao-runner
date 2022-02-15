#! /bin/bash

#######################################################################
#Description    : Converts between different dataset formats using datumaro
#Args           : Experiment directory, input-format, export-format
#Author         : Joshua Hoogstraat
#Date           : 01.02.2021
#######################################################################

## IMPORTANT ##
# 1. Requires python venv to be active (source .venv/bin/activate).
# 2. Input/Export format is used as directories names aswell!
# VOC: A custom labels file is required as the labels are usually not the Pascal Visual Object Classes.
# VoTT-json: Path entries in the json-file should be relative, not absolute.
###############

## EXAMPLES ##
# ./scripts/resize_dataset.sh project kitti_detection/train kitti_detection kitti_detection_1000x1000
# ./scripts/resize_dataset.sh project voc_detection voc_detection voc_detection_1000x1000
###############

# Some input/output formats (See datumaro docs for more):
# - vott_json (https://openvinotoolkit.github.io/datumaro/docs/formats/vott_json/)
# - coco_instances (https://openvinotoolkit.github.io/datumaro/docs/formats/coco/)
# - voc_detection (https://openvinotoolkit.github.io/datumaro/docs/formats/pascal_voc/)
# - kitti_detection (https://openvinotoolkit.github.io/datumaro/docs/formats/kitti/)

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 PROJECT INPUT_DATASET INPUT_FORMAT OUTPUT_DIRECTORY" >&2
  exit 1
fi

# Check if input dataset exists
if ! [ -d "projects/$1/data/$2" ]
  then
    echo "projects/$1/data/$2 not a directory" >&2
    exit 1
fi

datum transform -t resize \
                -o projects/$1/data/$4/ \
                --overwrite \
                projects/$1/data/$2:$3 \
                -- -dw 1000 -dh 1000 \
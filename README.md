# YOLO-V1-TF1
This repository contains an implementation of YOLO v1 (You Only Look Once version 1) using TensorFlow 1. YOLO is a popular object detection algorithm known for its speed and accuracy using Darknet-19 as backbone.

## Features
- Implementation of the YOLO v1 algorithm
- Training and inference scripts
- Pre-trained weights for quick start
- Evaluation metrics for model performance

## Installation
Clone YOLO-V1-TF1 repository

'''
git clone https://github.com/strcoder4007/YOLO-V1-TF1.git
cd YOLO-V1-TF1
'''

- Run the download script to Download Pascal VOC dataset, and create correct directories
'''
./download_data.sh
'''
- Download YOLO_small weight file and put it in data/weight

## Configuring Hyperparameters 
Modify configuration in yolo/config.py

## Training
'''
python train.py
'''

## Test
'''
python test.py
'''

#!/bin/bash

#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=75:10:00

source activate yad2k
KERAS_BACKEND=tensorflow
export HDF5_USE_FILE_LOCKING=FALSE

dataset_name='MusicInArt'
annotations_path='/home/msabatelli/art_detector/MusicInArt/yolo_splits/final_training_set.txt'
log_dir='../logs/from_bottleneck/'
classes_path='/home/msabatelli/art_detector/MusicInArt/yolo_splits/list_of_instruments.txt'
anchors_path='/home/msabatelli/art_detector/MusicInArt/Anchors/MIA_full_yolo_full_yolo_yolo_anchors.txt'
bayesian=false

python train.py --dataset_name $dataset_name --annotations_path $annotations_path --log_dir $log_dir --classes_path $classes_path --anchors_path $anchors_path --bayesian $bayesian

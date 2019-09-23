#!/bin/bash

#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=75:10:00

source activate yad2k
KERAS_BACKEND=tensorflow
export HDF5_USE_FILE_LOCKING=FALSE

dataset_name='TinyMusicInArt'
annotations_path='../annotated_datasets/CSV/MusicInArt/dataset_splits/Alan/tiny_dataset_Alan_training_set.txt'
log_dir='../logs/'
classes_path='../annotated_datasets/CSV/MusicInArt/instruments_list/tiny_version_list_of_instruments.txt'
anchors_path='../anchors/tiny_version_full_yolo_yolo_anchors.txt'
bayesian=false

python train.py --dataset_name $dataset_name --annotations_path $annotations_path --log_dir $log_dir --classes_path $classes_path --anchors_path $anchors_path --bayesian $bayesian

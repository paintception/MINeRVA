#!/bin/bash

dataset_name='minerva'
training_annotations_path='/path/to/my/training_set.txt'
validation_annotations_path='/path/to/my/validation_set.txt'
log_dir='./weights/'
classes_path='/path/to/my/list_of_instruments.txt'
anchors_path='../anchors/minerva_anchors.txt'

python train.py --dataset_name $dataset_name --training_annotations_path $training_annotations_path --validation_annotations_path $validation_annotations_path --log_dir $log_dir --classes_path $classes_path --anchors_path $anchors_path

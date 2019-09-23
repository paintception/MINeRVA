import pandas as pd
import os

ALAN_PATH = '../annotated_datasets/CSV/MusicInArt/dataset_splits/Alan/'

alan_destination = '/scratch/msabatelli/TinyMusicInArt/Images/'

if not os.path.exists(ALAN_PATH):
    os.makedirs(ALAN_PATH)

training_set = pd.read_csv('../annotated_datasets/CSV/MusicInArt/dataset_splits/tiny_dataset_training_set.txt', sep=' ')
testing_set = pd.read_csv('../annotated_datasets/CSV/MusicInArt/dataset_splits/tiny_dataset_testing_set.txt', sep = ' ')

training_set['Image Filename'] = training_set['Image Filename'].apply(lambda x: "{}{}".format(alan_destination, x))
testing_set['Image Filename'] = testing_set['Image Filename'].apply(lambda x: "{}{}".format(alan_destination, x))

training_set.to_csv(ALAN_PATH + 'tiny_dataset_Alan_' + 'training_set.txt', index=False, sep=' ')
testing_set.to_csv(ALAN_PATH + 'tiny_dataset_Alan_' + 'testing_set.txt', index=False, sep=' ')
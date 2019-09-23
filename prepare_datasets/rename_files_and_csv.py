import pandas as pd

import os
import shutil

IMAGES = '/home/matthia/Documents/Datasets/MusicInArt/'
DESTINATION = '/home/matthia/Documents/Datasets/MusicInArt/'

df = pd.read_csv('/home/matthia/Documents/art_detector/annotated_datasets/CSV/FullDataset/full_dataset_exploration.csv')

filenames = df['Image Filename']
image_list = os.listdir(IMAGES)
cnt = 0

for filename in set(filenames):
    f = os.path.basename(filename)
    extension = os.path.splitext(filename)[1]
    for image_name in image_list:
        if f == image_name:
            cnt += 1
            df = df.replace(filename, 'Image_' + str(cnt) + extension)  # replace old file name with new one in df
            new_image_name = 'Image_' + str(cnt) + extension # rename crappy image with new name

            shutil.copy(IMAGES + image_name, DESTINATION + new_image_name) # copy in the final folder

df.to_csv('/home/matthia/Documents/art_detector/annotated_datasets/CSV/FullDataset/MusicArtFullDataset.csv') # store clean csv
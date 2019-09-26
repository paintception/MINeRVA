import pandas as pd
import os

ground_truth = pd.read_csv('../annotated_datasets/CSV/MusicInArt/dataset_splits/tiny_dataset_testing_set.csv')
ground_truth_path = '../test/ground_truth/tiny_dataset/'

if not os.path.exists(ground_truth_path):
    os.makedirs(ground_truth_path)

for index, row in ground_truth.iterrows():
    filename = os.path.splitext(row['Image Filename'])[0]
    label = row['Term'].split(' ', 1)[0]
    bounding_box = row['coordinates'].replace(',',' ')

    with open(ground_truth_path + filename + '.txt', 'w') as f:
        f.write(label + ' ')
        f.write(bounding_box)
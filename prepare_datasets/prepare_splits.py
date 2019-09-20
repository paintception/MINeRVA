import pandas as pd
import os
import kmeans

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

annotation_datasets = '../annotated_datasets/CSV/'
columns =['Image Filename','coordinates','Term']
dst = '/path/to/your/dataset'

def get_full_dataset():
    flickr = pd.read_csv(annotation_datasets + 'Flickr/full_dataset_Flickr.csv', error_bad_lines=False)
    kmkg = pd.read_csv(annotation_datasets + 'KMKG/full_dataset_KMKG.csv')
    kmsk = pd.read_csv(annotation_datasets + 'KMSK/full_dataset_KMSK.csv')
    odile = pd.read_csv(annotation_datasets + 'Odile/full_dataset_Odile.csv')
    ridim_1 = pd.read_csv(annotation_datasets + 'RIDIM1/full_dataset_RIDIM1.csv')
    ridim_2 = pd.read_csv(annotation_datasets + 'RIDIM2/full_dataset_RIDIM2.csv')

    df = pd.concat([flickr, kmkg, kmsk, odile, ridim_1,ridim_2], axis=0)
    df = df[df.duplicated(subset=["Term"], keep=False)] # Remove instruments which occur only once
    df.drop_duplicates() # Remove potential duplicates
    df.to_csv(annotation_datasets + 'FullDataset/' + 'full_dataset_exploration.csv')

    return df

def explore_original_dataset(df):
    df['Term'].value_counts().plot('barh').invert_yaxis()
    plt.title('Distribution of number of bounding boxes on a subset of instruments')
    plt.xlabel('Bounding Box Occurences')
    plt.show()

def explore_training_and_testing_splits(training_df, testing_df):
    training_df['Term'].value_counts().plot('barh').invert_yaxis()
    plt.title('Distribution of number of bounding boxes on a subset of instruments')
    plt.xlabel('Bounding Box Occurences')

    testing_df['Term'].value_counts().plot('barh').invert_yaxis()
    plt.title('Distribution of number of bounding boxes on a subset of instruments')
    plt.xlabel('Bounding Box Occurences')
    plt.show()

def store_list_of_all_instruments(df, dataset_version):
    instruments = set(df['Term'].tolist())
    with open(annotation_datasets + 'FullDataset/' + 'instruments_list/' + dataset_version + '_list_of_instruments.txt', 'w') as f:
        for item in instruments:
            f.write("%s\n" % item)

def format_csv_to_txt(df):
    coordinates = df['coordinates'].tolist()
    terms = df['Encoded-Term'].tolist()
    coordinates = [str(i) for i in coordinates]
    terms = [str(i) for i in terms]
    full = [i + j for i, j in zip(coordinates, terms)]

    del df['Term']
    del df['Encoded-Term']
    del df['coordinates']
    df['coordinates_and_label'] = full

    return df

def prepare_ground_truth_files(df):
    ground_truth_path = annotation_datasets + 'FullDataset/GroundTruth/'
    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)

def make_splits_tiny_dataset(df):
    tiny = df[df['Term'].isin(df['Term'].value_counts()[df['Term'].value_counts() > 25].index)]
    tiny['Image Filename'] = tiny['Image Filename'].apply(lambda x: x.replace(os.path.dirname(x), dst))
    store_list_of_all_instruments(tiny, 'tiny_version')

    le = LabelEncoder()
    tiny['Encoded-Term'] = le.fit_transform(tiny.Term.values.astype(str))

    training_set, testing_set = train_test_split(tiny, test_size=0.2)

    #explore_training_and_testing_splits(training_set, testing_set)

    training_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/'+  'tiny_dataset_' + 'training_set.csv', index=False)
    testing_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/'+ 'tiny_dataset_' + 'testing_set.csv', index=False)

    entire_dataset = format_csv_to_txt(tiny)
    training_set = format_csv_to_txt(training_set)
    testing_set = format_csv_to_txt(testing_set)

    entire_dataset.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'tiny_dataset_' + 'complete_set.txt', index=False, sep=' ')
    training_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'tiny_dataset_' + 'training_set.txt', index=False, sep=' ')
    testing_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'tiny_dataset_' + 'testing_set.txt', index=False, sep=' ')

full_dataset = get_full_dataset()
#explore_original_dataset(full_dataset)
#make_splits_full_dataset(full_dataset)
make_splits_tiny_dataset(full_dataset)
kmeans.prepare_anchors('tiny_version', 9, annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'tiny_dataset_' + 'complete_set.txt')

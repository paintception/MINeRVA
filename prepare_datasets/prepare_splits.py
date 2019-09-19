import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

annotation_datasets = '../annotated_datasets/CSV/'
columns =['Image Filename','coordinates','Term']
dst = '/home/matthia/Documents/Datasets/InstrumentsFullCollection'

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

def store_list_of_all_instruments(df, dataset_version):
    instruments = set(df['Term'].tolist())
    with open(annotation_datasets + 'FullDataset/' + 'instruments_list/' + dataset_version + '_list_of_instruments.txt', 'w') as f:
        for item in instruments:
            f.write("%s\n" % item)

def make_splits_full_dataset(df):
    le = LabelEncoder()  # We need a numerical encoding for each instrument
    df['Term'] = le.fit_transform(df.Term.values.astype(str))
    df['Image Filename'] = df['Image Filename'].apply(lambda x: x.replace(os.path.dirname(x), dst))  # change paths
    coordinates = df['coordinates'].tolist()
    terms = df['Term'].tolist()
    coordinates = [str(i) for i in coordinates]
    terms = [str(i) for i in terms]
    full = [i + j for i, j in zip(coordinates, terms)]

    del df['Term']
    del df['coordinates']
    df['full'] = full

    df.to_csv(annotation_datasets + 'FullDataset/' + 'full_dataset.csv', index=False, sep=' ')
    df.to_csv(annotation_datasets + 'FullDataset/' + 'full_dataset.txt', index=False, sep=' ')

    training_set, testing_set = train_test_split(df, test_size=0.2)
    training_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_' + 'training_set.csv', index=False)
    testing_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_' + 'testing_set.csv', index=False)

    training_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_' + 'training_set.txt', index=False, sep=' ')
    testing_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_' + 'testing_set.txt', index=False, sep=' ')

def make_splits_tiny_dataset(df):
    tiny = df[df['Term'].isin(df['Term'].value_counts()[df['Term'].value_counts() > 25].index)]
    store_list_of_all_instruments(tiny, 'tiny_version')

    tiny['Term'].value_counts().plot('barh').invert_yaxis()
    plt.title('Distribution of number of bounding boxes on a subset of instruments')
    plt.xlabel('Bounding Box Occurences')
    plt.show()

    le = LabelEncoder()
    tiny['Term'] = le.fit_transform(tiny.Term.values.astype(str))
    tiny['Image Filename'] = tiny['Image Filename'].apply(lambda x: x.replace(os.path.dirname(x), dst))  # change paths
    coordinates = tiny['coordinates'].tolist()
    terms = tiny['Term'].tolist()
    coordinates = [str(i) for i in coordinates]
    terms = [str(i) for i in terms]
    full = [i + j for i, j in zip(coordinates, terms)]

    del tiny['Term']
    del tiny['coordinates']
    tiny['full'] = full

    tiny.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'full_tiny_dataset.csv', index=False, sep=' ')
    tiny.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'full_tiny_dataset.txt', index=False, sep=' ')

    training_set, testing_set = train_test_split(tiny, test_size=0.2)
    training_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/'+  'tiny_dataset_' + 'training_set.csv', index=False)
    testing_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/'+ 'tiny_dataset_' + 'testing_set.csv', index=False)

    training_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'tiny_dataset_' + 'training_set.txt', index=False, sep=' ')
    testing_set.to_csv(annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'tiny_dataset_' + 'testing_set.txt', index=False, sep=' ')


full_dataset = get_full_dataset()
#explore_original_dataset(full_dataset)
#make_splits_full_dataset(full_dataset)
make_splits_tiny_dataset(full_dataset)
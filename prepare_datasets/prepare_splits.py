import pandas as pd
import os
import kmeans
import shutil

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

annotation_datasets = '../annotated_datasets/CSV/'
columns =['Image Filename','coordinates','Term']

def get_full_dataset():
    """
    We load all the separate dataframes which have been retrieved from cytomine with: get_annotations_cytomine.py
    Since each tiny-dataset has been labelled within its own cytomine project we have to re-gather everything separately
    --> blame Eva for this.

    :return: a dataframe with all the merged annotations which have been collected from cytomine.
            we remove potential duplicates and the annotations which have been made for an instrument only once since
            they can't be part of our training-testing splits.
    """

    if not os.path.isfile(annotation_datasets + 'FullDataset/' + 'full_dataset_exploration.csv'):
        flickr = pd.read_csv(annotation_datasets + 'Flickr/full_dataset_Flickr.csv', error_bad_lines=False)
        kmkg = pd.read_csv(annotation_datasets + 'KMKG/full_dataset_KMKG.csv')
        kmsk = pd.read_csv(annotation_datasets + 'KMSK/full_dataset_KMSK.csv')
        odile = pd.read_csv(annotation_datasets + 'Odile/full_dataset_Odile.csv')
        ridim_1 = pd.read_csv(annotation_datasets + 'RIDIM1/full_dataset_RIDIM1.csv')
        ridim_2 = pd.read_csv(annotation_datasets + 'RIDIM2/full_dataset_RIDIM2.csv')

        df = pd.concat([flickr, kmkg, kmsk, odile, ridim_1,ridim_2], axis=0)
        df = df[df.duplicated(subset=["Term"], keep=False)] # Remove instruments which occur only once
        df = df.drop_duplicates() # Remove potential duplicates
        df.to_csv(annotation_datasets + 'FullDataset/' + 'full_dataset_exploration.csv')

        print('Stored a full dirty version of the dataset')

    if not os.path.isfile(annotation_datasets + '/FullDataset/MusicArtFullDataset.csv'):
        IMAGES = '/home/matthia/Documents/Datasets/InstrumentsFullCollection/'
        DESTINATION = annotation_datasets + 'FullDataset/MusicInArt/'
        path_regex = '^(/[^/ ]*)+/?$'   # used for deleting files which are on my laptop but have not been annotated

        if not os.path.exists(DESTINATION):
            os.makedirs(DESTINATION)

        df = pd.read_csv(annotation_datasets + 'FullDataset/' + 'full_dataset_exploration.csv')
        filenames = df['Image Filename']
        image_list = os.listdir(IMAGES)
        cnt = 0

        print('Moving and Storing Images')

        for filename in set(filenames):
            f = os.path.basename(filename)
            extension = os.path.splitext(filename)[1]
            for image_name in image_list:
                if f == image_name:
                    cnt += 1
                    df = df.replace(filename,
                                    'Image_' + str(cnt) + extension)  # replace old file name with new one in df
                    new_image_name = 'Image_' + str(cnt) + extension  # rename crappy image with a better name

                    shutil.copy(IMAGES + image_name, DESTINATION + new_image_name)  # copy in the final folder

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        filter = df['Image Filename'].str.contains(path_regex)
        df = df[~filter]

        df.to_csv(annotation_datasets + 'FullDataset/'+'/MusicArtFullDataset.csv')
        return df

    else:
        print('Dataset and Images have already been created')
        df = pd.read_csv(annotation_datasets + 'FullDataset/'+'/MusicArtFullDataset.csv')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df

def explore_training_and_testing_splits(training_df, testing_df):
    """
    Some exploratory plots which show us the distribution of the amount of bounding boxes that have been made fro each
        instrument class. We check that the distribution matches between training and testing
    :param training_df: the training-df
    :param testing_df: the testing-df
    :return: None
    """

    training_df['Term'].value_counts().plot('barh').invert_yaxis()
    plt.title('Distribution of number of bounding boxes on a subset of instruments')
    plt.xlabel('Bounding Box Occurences')

    testing_df['Term'].value_counts().plot('barh').invert_yaxis()
    plt.title('Distribution of number of bounding boxes on a subset of instruments')
    plt.xlabel('Bounding Box Occurences')
    plt.show()

def store_list_of_all_instruments(df, dataset_version):
    """
    YOLO and FastRCNN require a set of all instruments that need to be detected. We therefore extract this information
    from the retrieved dataset and save it in a different file
    :param df: the full dataset
    :param dataset_version: a string which indicates which version of the dataset we are working with (full vs tiny)
    :return: None
    """

    instruments = set(df['Term'].tolist())
    with open(annotation_datasets + 'FullDataset/' + 'instruments_list/' + dataset_version + '_list_of_instruments.txt', 'w') as f:
        for item in instruments:
            f.write("%s\n" % item)

def format_csv_to_txt(df):
    """
    Given a dataframe we need some processing in order to have it in a way which can be used by YOLO and FastRCNN.
    This script performs some dummy transformations on the dataframe and returns a new version of it which matches with
    what is required for training.

    :param df: a dataframe
    :return: a slighlty modified version of the same df
    """
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

def prepare_ground_truth_files(testing_df):
    ground_truth_path = annotation_datasets + 'FullDataset/GroundTruth/'
    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)

def make_splits_tiny_dataset(df):
    """
    We create the official splits for our dataset, we reseve 80% for training while 20% for testing.
    Here we deal with the 'tiny' version of our dataset, where we only keep the instruments which have been annotated
    more than 25 times. The idea is to provide two different datasets, an 'easy' one and a 'harder' one.
    We also store everything in a csv and txt file.

    :param df: the full dataframe
    :return: None
    """

    tiny = df[df['Term'].isin(df['Term'].value_counts()[df['Term'].value_counts() > 25].index)]
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
kmeans.prepare_anchors('tiny_version', 6, annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'tiny_dataset_' + 'complete_set.txt') # get anchors for tiny-yolo
kmeans.prepare_anchors('tiny_version', 9, annotation_datasets + 'FullDataset/' + 'dataset_splits/' + 'tiny_dataset_' + 'complete_set.txt') # get anchors for YOLOV3
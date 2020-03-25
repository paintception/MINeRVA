import pandas as pd
from sklearn.preprocessing import LabelEncoder

STORING_PATH = ''

splits_path = ''
mode = ''

training_set = pd.read_csv(splits_path + 'train.txt', index_col=[0], sep=',')
training_set['image_filename'] = training_set['image_filename'].replace(' ', '', regex=True)
training_set['instrument_name'] = training_set['instrument_name'].replace(' ', '-', regex=True)

dev_set = pd.read_csv(splits_path + 'dev.txt', index_col=[0], sep=',')
dev_set['image_filename'] = dev_set['image_filename'].replace(' ', '', regex=True)
dev_set['instrument_name'] = dev_set['instrument_name'].replace(' ', '-', regex=True)

testing_set = pd.read_csv(splits_path + 'test.txt', index_col=[0], sep=',')
testing_set['image_filename'] = testing_set['image_filename'].replace(' ', '', regex=True)
testing_set['instrument_name'] = testing_set['instrument_name'].replace(' ', '-', regex=True)

instruments = set(training_set['instrument_name'].tolist())
with open(splits_path + 'list_of_instruments.txt','w') as f:
    for item in instruments:
        f.write("%s\n" % item)

le = LabelEncoder()

training_set['instrument_label'] = le.fit_transform(training_set.instrument_name.values.astype(str))
dev_set['instrument_label'] = le.fit_transform(dev_set.instrument_name.values.astype(str))
testing_set['instrument_label'] = le.fit_transform(testing_set.instrument_name.values.astype(str))

if mode == 'hypernyms':
    instruments = set(training_set['lev1'].tolist())
    with open(splits_path + 'hypernyms.txt', 'w') as f:
        for item in instruments:
            f.write("%s\n" % item)

    training_set.columns.str.match('Unnamed')
    training_set = training_set.loc[:, ~training_set.columns.str.match('Unnamed')]
    dev_set.columns.str.match('Unnamed')
    dev_set = dev_set.loc[:, ~dev_set.columns.str.match('Unnamed')]
    testing_set.columns.str.match('Unnamed')
    testing_set = testing_set.loc[:, ~testing_set.columns.str.match('Unnamed')]

    training_set['lev1'] = le.fit_transform(training_set.lev1.values.astype(str))
    dev_set['lev1'] = le.fit_transform(dev_set.lev1.values.astype(str))
    testing_set['lev1'] = le.fit_transform(testing_set.lev1.values.astype(str))

    training_set['bounding_box_coordinates'] = training_set['bounding_box_coordinates'].astype(str) + ','
    training_set['yolo_information'] = training_set['bounding_box_coordinates'] + training_set['lev1'].map(str)
    dev_set['bounding_box_coordinates'] = dev_set['bounding_box_coordinates'].astype(str) + ','
    dev_set['yolo_information'] = dev_set['bounding_box_coordinates'] + dev_set['lev1'].map(str)
    testing_set['bounding_box_coordinates'] = testing_set['bounding_box_coordinates'].astype(str) + ','
    testing_set['yolo_information'] = testing_set['bounding_box_coordinates'] + testing_set['lev1'].map(str)

    del training_set['id']
    del training_set['instrument_name']
    del training_set['bounding_box_coordinates']
    del training_set['area_of_bounding_box']
    del training_set['instrument_label']
    del training_set['lev1']
    del training_set['lev2']

    del dev_set['id']
    del dev_set['instrument_name']
    del dev_set['bounding_box_coordinates']
    del dev_set['area_of_bounding_box']
    del dev_set['instrument_label']
    del dev_set['lev1']
    del dev_set['lev2']

    del testing_set['id']
    del testing_set['instrument_name']
    del testing_set['bounding_box_coordinates']
    del testing_set['area_of_bounding_box']
    del testing_set['instrument_label']
    del testing_set['lev1']
    del testing_set['lev2']

    training_set['image_filename'] = training_set['image_filename'].apply(lambda x: "{}{}".format(STORING_PATH, x))
    training_set['yolo_information'] = training_set['yolo_information'].replace(' ', '_', regex=True)
    training_set.to_csv(splits_path + mode + '_training_set.txt', index = False, sep = ' ')

    dev_set['image_filename'] = dev_set['image_filename'].apply(lambda x: "{}{}".format(STORING_PATH, x))
    dev_set['yolo_information'] = training_set['yolo_information'].replace(' ', '_', regex=True)

    dev_set.to_csv(splits_path + mode + '_dev_set.txt', index=False, sep=' ')

    testing_set['image_filename'] = testing_set['image_filename'].apply(lambda x: "{}{}".format(STORING_PATH, x))
    testing_set['yolo_information'] = testing_set['yolo_information'].replace(' ', '_', regex=True)

    testing_set.to_csv(splits_path + mode + '_testing_set.txt', index=False, sep=' ')

if mode == 'full':
    training_set.instrument_label.loc[(training_set['instrument_label'] >= 0)] = 0
    dev_set.instrument_label.loc[(dev_set['instrument_label'] >= 0)] = 0
    testing_set.instrument_label.loc[(testing_set['instrument_label'] >= 0)] = 0

    print(training_set.head(10))

if mode != 'granular':
    training_set['bounding_box_coordinates'] = training_set['bounding_box_coordinates'].astype(str) + ','
    training_set['yolo_information'] = training_set['bounding_box_coordinates']+ training_set['instrument_label'].map(str)

    dev_set['bounding_box_coordinates'] = dev_set['bounding_box_coordinates'].astype(str) + ','
    dev_set['yolo_information'] = dev_set['bounding_box_coordinates']+ dev_set['instrument_label'].map(str)

    testing_set['bounding_box_coordinates'] = testing_set['bounding_box_coordinates'].astype(str) + ','
    testing_set['yolo_information'] = testing_set['bounding_box_coordinates']+ testing_set['instrument_label'].map(str)

    training_set['image_filename'] = training_set['image_filename'].apply(lambda x: "{}{}".format(STORING_PATH, x))
    dev_set['image_filename'] = dev_set['image_filename'].apply(lambda x: "{}{}".format(STORING_PATH, x))
    testing_set['image_filename'] = testing_set['image_filename'].apply(lambda x: "{}{}".format(STORING_PATH, x))

    del training_set['id']
    del training_set['instrument_name']
    del training_set['bounding_box_coordinates']
    del training_set['area_of_bounding_box']
    del training_set['instrument_label']
    
    del dev_set['id']
    del dev_set['instrument_name']
    del dev_set['bounding_box_coordinates']
    del dev_set['area_of_bounding_box']
    del dev_set['instrument_label']
    
    del testing_set['id']
    del testing_set['instrument_name']
    del testing_set['bounding_box_coordinates']
    del testing_set['area_of_bounding_box']
    del testing_set['instrument_label']
    
    training_set.to_csv(splits_path + mode + '_training_set.txt', index = False, sep = ' ')
    dev_set.to_csv(splits_path + mode + '_dev_set.txt', index = False, sep = ' ')
    testing_set.to_csv(splits_path + mode + '_testing_set.txt', index = False, sep = ' ')

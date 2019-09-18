from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pprint import pprint

import logging
import sys
import re
import csv
from argparse import ArgumentParser
from shapely import geometry
import os
import pandas as pd
from cytomine import Cytomine
from cytomine.models import AnnotationCollection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

IMAGES_PATH = '/home/matthia/Documents/Datasets/KMSK/'

annotations_file = '../annotation_files/KMSKAnnotations.csv'
pat = re.compile(r'''(-*\d+\.\d+ -*\d+\.\d+);*''')

df = pd.read_csv(annotations_file, sep=';')
cols = [1,2,3,7,9,10]
df.drop(df.columns[cols],axis=1,inplace=True)
df['coordinates'] = ''

final_df = pd.DataFrame(columns=['Id', 'Image Id', 'Image Filename', 'Term', 'coordinates'])

if __name__ == '__main__':

    id_project = '121964493'
    id_name = 'KMSK'

    annotated_datasets_csv = '../annotated_datasets/CSV/' + id_name + '/'
    annotated_datasets_txt = '../annotated_datasets/TXT/' + id_name + '/'

    parser = ArgumentParser(prog="Cytomine Python client example")

    # Cytomine
    parser.add_argument('--cytomine_host', dest='host',
                        default='demo.cytomine.be', help="The Cytomine host")
    parser.add_argument('--cytomine_public_key', dest='public_key',
                        default='ee8335d9-ae0a-4368-b61b-719d33543523',
                        help="The Cytomine public key")
    parser.add_argument('--cytomine_private_key', dest='private_key',
                        default='442903ae-4cee-4546-8ab7-4ed3404f8b94',
                        help="The Cytomine private key")
    parser.add_argument('--cytomine_id_project', dest='id_project',
                        default=id_project,
                        help="The project from which we want the crop")
    parser.add_argument('--download_path', default='/home/matthia/Desktop/annotations/', required=False,
                        help="Where to store images")
    params, other = parser.parse_known_args(sys.argv[1:])

    with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key,
                  verbose=logging.INFO) as cytomine:
        annotations = AnnotationCollection()
        annotations.project = params.id_project
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showGIS = True
        annotations.fetch()

        for annotation in annotations:
            n = df.loc[df['Image Id'] == annotation.image]
            n['coordinates'] = ''
            matches = pat.findall(annotation.location)

            if matches:
                try:
                    lst = [tuple(map(float, m.split())) for m in matches]

                    poly = geometry.Polygon(lst)
                    info = geometry.mapping(poly)
                    coordinates = info['coordinates']

                    x_min, y_min = coordinates[0][0]
                    x_max, y_max = coordinates[0][2]
                    coordinates = str([int(x_min),int(y_min),int(x_max),int(y_max)])
                    coordinates = coordinates.replace(' ','')
                    n['coordinates'] = coordinates[1:-1]+','
                    final_df = final_df.append(n)
                except:
                    pass

        le = LabelEncoder()
        instruments = set(final_df['Term'].tolist())

        with open(annotated_datasets_txt + 'project_' + id_name + '_instruments.txt', 'w') as f:
            for item in instruments:
                f.write("%s\n" % item)

        final_df['Term'] = le.fit_transform(final_df.Term.values.astype(str))
        del final_df['Id']
        del final_df['Image Id']
        del final_df['Y']
        final_df['Image Filename'] = IMAGES_PATH + final_df['Image Filename'].astype(str)
        col1 = 'Term'
        col2 = 'coordinates'
        final_df = final_df[[col1 if col == col2 else col2 if col == col1 else col for col in final_df.columns]]

        final_df["merged"] = final_df["coordinates"].map(str) + final_df["Term"].map(str)
        del final_df['coordinates']
        del final_df['Term']

        final_df['Image Filename'] = final_df['Image Filename'].str.replace(" ","")

        final_df.to_csv(annotated_datasets_csv + 'full_dataset_' + id_name + '.csv', index=False)
        final_df.to_csv(annotated_datasets_txt + 'tmp_dataset_' + id_name + '.txt', index=False, sep=' ')

        training_set, testing_set = train_test_split(final_df, test_size=0.2)
        training_set.to_csv(annotated_datasets_csv + 'dataset_' + id_name + '_training_set.csv', index=False)
        testing_set.to_csv(annotated_datasets_csv + 'dataset_' + id_name + '_testing_set.csv', index=False)

        training_set.to_csv(annotated_datasets_txt + 'dataset_' + id_name + '_training_set.txt', index=False, sep=' ')
        testing_set.to_csv(annotated_datasets_txt + 'dataset_' + id_name + '_testing_set.txt', index=False, sep = ' ')

        with open(annotated_datasets_txt + 'tmp_dataset_' + id_name + '.txt', 'r') as f, open(annotated_datasets_txt +
                                                                                              '/full_dataset_' + id_name + '.txt', 'w') as fo:
            for line in f:
                fo.write(line.replace('"', '').replace("'", ""))
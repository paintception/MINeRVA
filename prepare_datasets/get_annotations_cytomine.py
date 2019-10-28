from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pprint import pprint
from matplotlib import pyplot as plt

import logging
import sys
import re
import time
import urllib
import cv2
from argparse import ArgumentParser
from shapely import geometry
import os
import pandas as pd
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstanceCollection, TermCollection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

STORING_PATH = "/home/matthia/Desktop/MusicInArt/"
pat = re.compile(r'''(-*\d+\.\d+ -*\d+\.\d+);*''')

d = {}

if __name__ == '__main__':

    id_project = '125386343'
    parser = ArgumentParser(prog="Cytomine Python client example")

    # Cytomine
    parser.add_argument('--cytomine_host', dest='host',
                        default='research.cytomine.be', help="The Cytomine host")
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

        images = ImageInstanceCollection().fetch_with_filter("project", id_project)
        terms = TermCollection().fetch_with_filter("project", id_project)

        for term in terms:
            d[term.id] = term.name

        for image in images:
            print('Analyzing Image: ', image.filename)
            #image.download(os.path.join(STORING_PATH, str(params.id_project), "{originalFilename}"))

            annotations = AnnotationCollection()
            annotations.image = image.id
            annotations.project = params.id_project
            annotations.showWKT = True
            annotations.showMeta = True
            annotations.showGIS = True
            annotations.showTerm = True
            annotations.fetch()

            for annotation in annotations:
                #print(annotation.term[0])
                matches = pat.findall(annotation.location)
    
                if matches:
                    lst = [tuple(map(float, m.split())) for m in matches]
    
                    poly = geometry.Polygon(lst)
                    info = geometry.mapping(poly)
                    coordinates = info['coordinates']
                    x_min, y_min = coordinates[0][0]
                    x_max, y_max = coordinates[0][2]
                    coordinates = str([int(x_min),int(y_min),int(x_max),int(y_max)])
                    coordinates = coordinates.replace(' ','')
                    print(coordinates)
                    time.sleep(0.2)
                    #n['coordinates'] = coordinates[1:-1]+','
                    #final_df = final_df.append(n)
            print('------------------')

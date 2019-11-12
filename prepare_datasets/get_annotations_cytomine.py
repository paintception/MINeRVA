from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pprint import pprint

import logging
import sys
import re
import time
import urllib
from shapely import geometry

import os
import pandas as pd
from cytomine import Cytomine

from cytomine.models import AnnotationCollection, ImageInstanceCollection, TermCollection
from sklearn.preprocessing import LabelEncoder

STORING_PATH = "/home/matthia/Desktop/MusicInArt/"  # the annotated images coming from cytomine
CROPS_PATH = "/home/matthia/Desktop/MusicInArtCrops/"   # the crops which can be used for the classification experiments
pat = re.compile(r'''(-*\d+\.\d+ -*\d+\.\d+);*''')

if __name__ == '__main__':

    # Nikolay you can append everything in here if you want
    df = pd.DataFrame(columns = ['image_filename', 'bounding_box_coordinates',
                                 'instrument_name', 'area_of_bounding_box'])
    d = {}

    host = 'research.cytomine.be'

    id_project = '125386343' #'130917744', '119921990', '121964493', '122386653', '105442790' # run this script for each of these ids separately
    public_key = 'ee8335d9-ae0a-4368-b61b-719d33543523'
    private_key = '442903ae-4cee-4546-8ab7-4ed3404f8b94'

    with Cytomine(host=host, public_key=public_key, private_key=private_key,
                  verbose=logging.INFO) as cytomine:

        images = ImageInstanceCollection().fetch_with_filter("project", id_project)
        terms = TermCollection().fetch_with_filter("project", id_project)

        for term in terms[:218]: # we do not want the coming labels which do not correspond to any musical instruments
            d[term.id] = term.name

        for image in images:
            filename = image.originalFilename

            try:
                print('Dumping the Original Image from Cytomine!')
                image.download(os.path.join(STORING_PATH, str(id_project), "{originalFilename}"))

                annotations = AnnotationCollection()
                annotations.image = image.id
                annotations.project = id_project
                annotations.showWKT = True
                annotations.showMeta = True
                annotations.showGIS = True
                annotations.showTerm = True
                annotations.fetch()

                for annotation in annotations:
                    matches = pat.findall(annotation.location)
                    bounding_box_area = annotation.area

                    print('Dumping the Crop of the Instrument')
                    annotation.dump(dest_pattern=os.path.join(CROPS_PATH, "{project}", "crop", "{id}.jpg"))

                    if matches:
                        instrument_name = d[annotation.term[0]]

                        lst = [tuple(map(float, m.split())) for m in matches]

                        poly = geometry.Polygon(lst)
                        info = geometry.mapping(poly)
                        coordinates = info['coordinates']
                        x_min, y_min = coordinates[0][0]
                        x_max, y_max = coordinates[0][2]
                        coordinates = str([int(x_min),int(y_min),int(x_max),int(y_max)])
                        coordinates = coordinates.replace(' ','')
                        print('Bounding Box coordinates:', coordinates[1:-1]+','+ instrument_name)
                        print('Area of the Bounding Box: ', bounding_box_area)
                        time.sleep(0.2)
                print('------------------')

            except:
                """ 
                This is an error coming from Cytomine which we can pass
                """
                pass


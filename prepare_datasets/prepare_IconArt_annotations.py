import glob
import os
import xml.etree.ElementTree as ET

ANNOTATIONS_PATH = '../IconArt_v1/Annotations/'
TRAINING_SET = '../IconArt_v1/ImageSets/train.txt'
ALAN_IMAGES_PATH = '/scratch/msabatelli/datasets/IconArt_v1/JPEGImages/'

annotations = glob.glob(ANNOTATIONS_PATH+"*.xml")

annotations_ = {'angel' : 0,
               'Child_Jesus' : 1,
               'crucifixion_of_Jesus' : 2,
               'Mary' : 3,
               'nudity' : 4,
               'ruins': 5,
               'Saint_Sebastien' : 6,
               }
d = {}

for annotation in annotations:
    print('-------------------------------')
    print('Parsing annotation {}'.format(annotation))
    tree = ET.parse(annotation)
    root = tree.getroot()
    for item in root.findall('object'):
        print('Found Bounding Box')
        info = []
        for child in item.findall('bndbox'):
            for coordinate in child.findall('xmin'):
                print('X-min coordinate {}'.format(coordinate.text))
                info.append(coordinate.text)
            for coordinate in child.findall('ymin'):
                print('Y-min coordinate {}'.format(coordinate.text))
                info.append(coordinate.text)
            for coordinate in child.findall('xmax'):
                print('X-max coordinate {}'.format(coordinate.text))
                info.append(coordinate.text)
            for coordinate in child.findall('ymax'):
                print('Y-max coordinate {}'.format(coordinate.text))
                info.append(coordinate.text)
        for child in item.findall('name'):
            print('Label {}'.format(child.text))
            info.append(annotations_[child.text])

        d[annotation] = info


fo = open('../annotated_datasets/IconArt.txt', "w")

for k, v in d.items():
    annotation =  str(v).replace(' ','')
    annotation = annotation.replace('\'', '')
    annotation = ' ' + annotation[1:-1]
    image_name = os.path.basename(str(k))
    fo.write(os.path.splitext(ALAN_IMAGES_PATH + image_name)[0] + '.jpg'+ annotation+ '\n')

fo.close()
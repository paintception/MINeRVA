"""
Script which validates the performance of a YOLO model on a separate testing-set.
We create for each image in the dataset a file in which for each image we store the predictions given by the model in
the following format:

    - detected_class_name x_min y_min x_max y_max

This is important for computing the mAP once a similar file with the ground truth is created as presented in
https://github.com/Cartucho/mAP and done in the pascal-VOCO-2012 challenge
"""

import os
import argparse
from yolo import YOLO
from PIL import Image

DATASET_NAME = 'IconArt'

IMAGES = './testing_set_tiny_MIA/'

DETECTION_RESULTS ='./yolo_detections/' + DATASET_NAME + '/'
DETECTED_IMAGES = './Images/' + DATASET_NAME + '/'

if not os.path.exists(DETECTED_IMAGES):
    os.makedirs(DETECTED_IMAGES)

if not os.path.exists(DETECTION_RESULTS):
    os.makedirs(DETECTION_RESULTS)

def detect_img(yolo):
    for img in os.listdir(IMAGES):
        base = os.path.basename(img)
        tmp_filename = os.path.splitext(base)[0]

        image = Image.open(IMAGES + img)
        results = yolo.detect_image(image)
        filename = DETECTION_RESULTS + tmp_filename + '.txt'

        with open(filename, 'w') as f:
            for class_name, min_coord, max_coord in zip(*results[:3]):
                f.write("%s " % class_name)
                f.write("%s " % str(min_coord)[1:-1].replace(',',''))
                f.write("%s \n" % str(max_coord)[1:-1].replace(',',''))

        results[3].save(DETECTED_IMAGES + 'detections_' + tmp_filename + '.jpg')

    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--test', default=False, action="store_true",
        help='Running model on the testing set'
    )

    FLAGS = parser.parse_args()

    if FLAGS.test:
        detect_img(YOLO(**vars(FLAGS)))
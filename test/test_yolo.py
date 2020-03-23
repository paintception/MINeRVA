"""
Script which validates the performance of a trained YOLO model on a separate testing-set.
We create for each image in the dataset a txt file where we store the predictions that are given by the model in
the following format:

    - detected_class_name x_min y_min x_max y_max coordinates

This is important for computing the mAP once a similar file with the ground truth is created as presented in
the ../prepare_datasets/ folder.

When ran this script will automatically create two folders, one where the txt files will be stored and another one which
will save the detected bounding boxes on top of the original image as presented in the README.md.
"""

import os
import argparse
from yolo import YOLO
from PIL import Image

DATASET_NAME = 'example_dataset'

IMAGES = '../testing_images/' #path to the images you would like to detect an instrument in

DETECTION_RESULTS ='./detected_files/' #path where to store the detected bounding-boxes
DETECTED_IMAGES = './detected_images/' #path where to store an image with the detected bounding-box

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
                if results is not None:
                    for class_name, min_coord, max_coord in zip(*results[:3]):
                        f.write("%s " % class_name)
                        f.write("%s " % str(min_coord)[1:-1].replace(',',''))
                        f.write("%s \n" % str(max_coord)[1:-1].replace(',',''))
                    assert isinstance(results, object)
                    results[3].save(DETECTED_IMAGES + 'detections_' + tmp_filename + '.jpg')
                else:
                    # it might be that there are no detections so we just store and empty txt file
                    continue

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

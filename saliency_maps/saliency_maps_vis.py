import os
# import tensorflow as tf
# import keras.backend as K
# from saliency_maps import visualize_filters

# from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing import image as Image
from visual_backprop import VisualBackprop
import numpy as np
from matplotlib import pyplot as plt
# import glob

from keras.models import load_model

import argparse


def arguments():
    parser = argparse.ArgumentParser(description='CharNMT arguments')

    parser.add_argument('-image', type=str, help='path to image', default=None)
    parser.add_argument('-model', type=str, help='model', default=None)
    parser.add_argument('-save', type=str, help='path to the saliency map', default=None)

    return parser.parse_args()


def show_image(image, grayscale=False, ax=None, title=''):
    if ax is None:
        plt.figure()
    plt.axis('off')

    if len(image.shape) == 2 or grayscale == False:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)

        vmax = np.percentile(image, 99)
        vmin = np.min(image)

        plt.imshow(image, vmin=vmin, vmax=vmax)
        plt.title(title)
    else:
        image = image + 127.5
        image = image.astype('uint8')

        plt.imshow(image)
        plt.title(title)


def get_trained_model(model_path='./ECCVModels/'):
    """
    Choose one of the different pre-trained models from the ECCV paper
    :return: one of the models + weights
    """
    pre_trained_model = load_model(os.path.join(model_path, 'model.h5'))
    pre_trained_model.load_weights(os.path.join(model_path, 'weights.h5'))

    return pre_trained_model


# Change the model and weights with the ones that are obtained after the classification training

def get_image(image_path='./images/example_image.jpg'):
    img = Image.load_img(image_path, target_size=(224, 224))
    img = np.asarray(img)
    return img


# Loads the image in a format which is liked by a keras model

if __name__ == '__main__':
    args = arguments()
    model = load_model(os.path.join(args.model, 'model.h5'))
    model.load_weights(os.path.join(args.model, 'weights.h5'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    file_name = os.path.basename(args.image)
    image = get_image(args.image)

    show_image(image, ax=plt.subplot('120'), grayscale=False, title='Original Image')

    x = np.expand_dims(image, axis=0)
    visual_bprop = VisualBackprop(model)
    mask = visual_bprop.get_mask(x[0])
    show_image(mask, ax=plt.subplot('121'), title='Saliency Map from a trained model')

    # p = os.path.join("./viz_results/granular/VGG19/", inst)
    #
    # if not os.path.exists(p):
    #     os.makedirs(p)

    # plt.show()
    plt.savefig(os.path.join(args.save, file_name))

    # for im_path in glob.glob('./data/granular/test/*/*.jpg'):
    #     file_name = os.path.basename(im_path)
    #     inst = os.path.basename(os.path.dirname(im_path))
    #     # print(im_path)
    #     image = get_image(im_path)
    #
    #     show_image(image, ax=plt.subplot('120'), grayscale=False, title='Original Image')
    #
    #     x = np.expand_dims(image, axis=0)
    #
    #     visual_bprop = VisualBackprop(model)
    #
    #     #
    #     # visualize_filters.visualize_layer(model, 'block1_conv2')
    #
    #     mask = visual_bprop.get_mask(x[0])
    #     show_image(mask, ax=plt.subplot('121'), title='Saliency Map from a trained model')
    #
    #     p = os.path.join("./viz_results/granular/VGG19/", inst)
    #
    #     if not os.path.exists(p):
    #         os.makedirs(p)
    #
    #     # plt.show()
    #     plt.savefig(os.path.join(p, file_name))

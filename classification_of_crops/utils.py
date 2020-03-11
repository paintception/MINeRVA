from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import numpy as np
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical


class DataGenerator:
    def __init__(self, images_path='./data/top5/', batch_size=2):
        self.data_generator = ImageDataGenerator()
        self.images_path = images_path
        self.batch_size = batch_size

    def get_training_generator(self):
        train_generator = self.data_generator.flow_from_directory(
            directory=self.images_path + "./train/",
            target_size=(224, 224),
            color_mode="rgb",
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )

        return train_generator

    def get_validation_generator(self):
        valid_generator = self.data_generator.flow_from_directory(
            directory=self.images_path + "./dev/",
            target_size=(224, 224),
            color_mode="rgb",
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )

        return valid_generator

    def get_testing_generator(self):
        testing_generator = self.data_generator.flow_from_directory(
            directory=self.images_path + "./test/",
            target_size=(224, 224),
            color_mode="rgb",
            batch_size=1,
            class_mode="categorical",
            shuffle=False,
            seed=42
        )

        return testing_generator


class BalancedGenerator(DataGenerator):
    def __init__(self, images_path='./data/top5/', batch_size=2):
        super().__init__(images_path, batch_size)

    def get_training_generator(self):
        train_generator = self.data_generator.flow_from_directory(
            directory=self.images_path + "./train/",
            target_size=(224, 224),
            color_mode="rgb",
            batch_size=1,
            class_mode="sparse",
            shuffle=True,
            seed=42)

        X = []
        Y = []

        for x, y in train_generator:
            X.append(x)
            Y.append(y)

            if len(X) >= train_generator.n:
                break

        X = np.concatenate(X)
        # print(X.shape)
        X = X.reshape((train_generator.n, 224 * 224 * 3))
        Y = np.concatenate(Y)

        # print(X.shape, Y.shape)

        # generator, steps = balanced_batch_generator(X, Y, sampler=RandomOverSampler)

        ros = RandomOverSampler(random_state=42)
        # print(Y)
        X_res, y_res = ros.fit_resample(X, Y)
        # print(X_res.shape, y_res.shape)
        y_res = to_categorical(y_res)
        X_res = X_res.reshape((X_res.shape[0], 224, 224, 3))
        # print(X_res.shape, y_res.shape)

        # print(train_generator.class_indices)

        return self.data_generator.flow(X_res, y_res)


def get_pre_trained_model(model_path='./ECCVModels/', net_name='ResNet'):
    """
    Choose one of the different pre-trained models from the ECCV paper
    :return: one of the models + weights
    """
    pre_trained_model = load_model(os.path.join(model_path, f'creator_{net_name}_model.h5'))
    pre_trained_model.load_weights(os.path.join(model_path, f'creator_{net_name}_weights.h5'))
    pre_trained_model.layers.pop()
    pre_trained_model.layers.pop()

    return pre_trained_model

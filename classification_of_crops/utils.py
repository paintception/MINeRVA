from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

MODELS_PATH = './ECCVModels/' # ECCV models
IMAGES_PATH = './data/' # Toy data to illustrate how to run the script, put the crops of the instruments in the same data-structure

data_generator = ImageDataGenerator()

def get_training_generator():
    train_generator = data_generator.flow_from_directory(
        directory= IMAGES_PATH + "./train/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=2,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    return train_generator

def get_validation_generator():
    valid_generator = data_generator.flow_from_directory(
        directory= IMAGES_PATH + "./val/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=2,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    return valid_generator

def get_testing_generator():
    testing_generator = data_generator.flow_from_directory(
        directory= IMAGES_PATH + "./test/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=2,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    return testing_generator

def get_pre_trained_model():
    """
    Choose one of the different pre-trained models from the ECCV paper
    :return: one of the models + weights
    """

    pre_trained_model = load_model(MODELS_PATH + 'creator_ResNet_model.h5')
    pre_trained_model.load_weights(MODELS_PATH + 'creator_ResNet_weights.h5')
    pre_trained_model.layers.pop()
    pre_trained_model.layers.pop()

    return pre_trained_model

get_training_generator()
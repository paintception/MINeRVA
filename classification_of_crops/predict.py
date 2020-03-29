from keras.models import load_model
import utils
import os
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import argparse


def arguments():
    parser = argparse.ArgumentParser(description='CharNMT arguments')

    parser.add_argument('-data', type=str, help='path to data', default=None)
    parser.add_argument('-model', type=str, help='model', default=None)

    return parser.parse_args()


def predict(data_path, model_path):
    model = load_model(os.path.join(model_path, 'model.h5'))
    model.load_weights(os.path.join(model_path, 'weights.h5'))
    data_generator = utils.BalancedGenerator(images_path=data_path, batch_size=32)
    test_generator = data_generator.get_testing_generator()

    output = model.predict_generator(generator=test_generator, steps=test_generator.n // test_generator.batch_size,
                                     pickle_safe=True)
    print(output)
    output = np.argmax(output, axis=1)
    accuracy = accuracy_score(test_generator.classes, output)
    f1_macro = f1_score(test_generator.classes, output, average='macro')
    f1_micro = f1_score(test_generator.classes, output, average='micro')
    error_matrix = confusion_matrix(test_generator.classes, output)

    results = {'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'error_matrix': error_matrix,
               'class_indices': test_generator.class_indices}

    print(results)


if __name__ == '__main__':
    args = arguments()
    predict(data_path=args.data, model_path=args.model)

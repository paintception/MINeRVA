import utils
import os
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pickle
from keras.optimizers import Adam
import argparse
import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description='CharNMT arguments')

    parser.add_argument('-data', type=str, help='path to data', default=None)
    parser.add_argument('-model_path', type=str, help='path where the model is', default=None)
    parser.add_argument('-net', type=str, help='ResNet or V3 or VGG19', default=None)
    parser.add_argument('-save', type=str, help='path to save results', default=None)
    parser.add_argument('-lr', type=float, help='learning rate', default=None)
    parser.add_argument("--balanced", default=False, action="store_true", help="oversampling")

    return parser.parse_args()


def run_experiment(model_path, net_name, data_path, results_path, lr=0.0001, balanced=True):
    nb_epochs = 200  # adjust the number of epochs
    # results_path = f'./results_up/{data_path}/{net_name}/'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if balanced:
        data_generator = utils.BalancedGenerator(images_path=data_path, batch_size=32)
    else:
        data_generator = utils.DataGenerator(images_path=data_path, batch_size=32)

    csv_logger_callback = CSVLogger(os.path.join(results_path, 'results_file.csv'), append=True, separator=';')
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
                                            restore_best_weights=True
                                            )

    train_generator = data_generator.get_training_generator()
    valid_generator = data_generator.get_validation_generator()
    test_generator = data_generator.get_testing_generator()

    print("Classes:", test_generator.class_indices)
    print("Number of examples:", train_generator.n)

    pre_trained_model = utils.get_pre_trained_model(model_path, net_name)
    pre_trained_output = pre_trained_model.output

    predictions = Dense(len(test_generator.class_indices), activation=tf.nn.softmax, name='final_output')(
        pre_trained_output)
    model = Model(input=pre_trained_model.input, output=predictions)
    adam = Adam(lr)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.n // train_generator.batch_size,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.n // valid_generator.batch_size,
                        epochs=nb_epochs,
                        callbacks=[csv_logger_callback, early_stopping_callback]
                        )

    output = model.predict_generator(generator=test_generator, steps=test_generator.n // test_generator.batch_size,
                                     pickle_safe=True)

    output = np.argmax(output, axis=1)
    accuracy = accuracy_score(test_generator.classes, output)
    f1_macro = f1_score(test_generator.classes, output, average='macro')
    f1_micro = f1_score(test_generator.classes, output, average='micro')
    error_matrix = confusion_matrix(test_generator.classes, output)

    results = {'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'error_matrix': error_matrix,
               'class_indices': test_generator.class_indices}

    with open(os.path.join(results_path, 'results.pickle'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(os.path.join(results_path, 'model.h5'))
    model.save_weights(os.path.join(results_path, 'weights.h5'))


if __name__ == "__main__":
    args = arguments()
    run_experiment(model_path=args.model_path, net_name=args.net,
                   data_path=args.data, results_path=args.save,
                   lr=args.lr, balanced=args.balanced)

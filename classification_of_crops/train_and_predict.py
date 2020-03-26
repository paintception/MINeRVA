import utils
import os
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pickle
from keras.optimizers import Adam

import numpy as np


def run_experiment(net_name, dataset):
    nb_epochs = 200  # adjust the number of epochs
    results_path = f'./results_up/{dataset}/{net_name}/'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    data_generator = utils.BalancedGenerator(images_path=f'./data/{dataset}/', batch_size=32)

    csv_logger_callback = CSVLogger(results_path + 'results_file.csv', append=True, separator=';')
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
                                            restore_best_weights=True
                                            )

    train_generator = data_generator.get_training_generator()
    valid_generator = data_generator.get_validation_generator()
    test_generator = data_generator.get_testing_generator()

    print("Classes:", test_generator.class_indices)
    print("Number of examples:", train_generator.n)

    pre_trained_model = utils.get_pre_trained_model(net_name=net_name)
    pre_trained_output = pre_trained_model.output

    predictions = Dense(len(test_generator.class_indices), activation=tf.nn.softmax, name='final_output')(
        pre_trained_output)
    model = Model(input=pre_trained_model.input, output=predictions)
    adam = Adam(0.0001)

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

    with open(results_path + 'results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(results_path + 'model.h5')
    model.save_weights(results_path + 'weights.h5')


if __name__ == "__main__":

    for net in ['ResNet', 'V3', 'VGG19']:
        for data_name in ['top5', 'top10', 'top20', 'all', 'granular', 'source']:
            run_experiment(net, data_name)

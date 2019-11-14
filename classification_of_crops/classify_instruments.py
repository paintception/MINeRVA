import utils
import os
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping

def run_experiment():
    nb_epochs = 1   # adjust the number of epochs
    results_path = './results/'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    csv_logger_callback = CSVLogger(results_path + 'results_file.csv', append=True, separator=';')
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    train_generator = utils.get_training_generator()
    valid_generator = utils.get_validation_generator()
    test_generator = utils.get_testing_generator()

    pre_trained_model = utils.get_pre_trained_model()
    pre_trained_output = pre_trained_model.output

    predictions = Dense(len(train_generator.class_indices), activation=tf.nn.softmax, name='final_output')(pre_trained_output)
    model = Model(input=pre_trained_model.input, output=predictions)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.n//train_generator.batch_size,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.n//valid_generator.batch_size,
                        epochs=nb_epochs,
                        callbacks=[csv_logger_callback, early_stopping_callback]
                        )

    final_accuracies = model.evaluate_generator(generator=test_generator, steps=test_generator.n//test_generator.batch_size,
                                                pickle_safe=True)
    print("Final Acuracies ---->", final_accuracies[1])

    model.save(results_path + 'model.h5')
    model.save_weights(results_path + 'weights.h5')

if __name__ == "__main__":
    run_experiment()
import gc
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras

from keras_models import KerasModels
from create_spectograms import MelSpectrogram
from audiomanip.audioutils import AudioUtils

# Disable TF warnings about speed up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(exec_times=10, epochs=100, optimizer='adam', loss_function='categorical_crossentropy',
         results_folder='results/', conv_layers=[16,32,64]):
    # Configuration
    folder = 'gtzan'  # 'garageband'#
    create_npy_array = False
    batch_size = 32

    # results_folder = 'results_opts2/'
    # epochs = 100
    # exec_times = 10
    # optimizer = 'adam'  # 'sgd'

    model_path = folder + '_exec_' + str(exec_times) + '_epochs_' + str(epochs) + '_opt_' + str(
        optimizer) + '_loss_' + str(loss_function) + '.h5'

    if create_npy_array:
        create_npy(folder)

    songs = np.load(folder + '/' + 'songs.npy')
    genres = np.load(folder + '/' + 'genres.npy')

    # print("Original songs array shape: {0}".format(songs.shape))
    # print("Original genre array shape: {0}".format(genres.shape))

    input_shape = (128, 128)

    # Train multiple times and get mean score
    val_acc = []
    test_history = []
    test_acc = []
    test_acc_mvs = []
    output_text = ''

    for x in range(exec_times):
        # Split the dataset into training and test
        x_train, x_test, y_train, y_test = train_test_split(
            songs, genres, test_size=0.1, stratify=genres)

        # Split training set into training and validation
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=1 / 6, stratify=y_train)

        # split the train, test and validation data in size 128x128
        x_val, y_val = AudioUtils().splitsongs_melspect(x_val, y_val, '1D')
        x_test, y_test = AudioUtils().splitsongs_melspect(x_test, y_test, '1D')
        x_train, y_train = AudioUtils().splitsongs_melspect(x_train, y_train, '1D')

        cnn, output_text = KerasModels.cnn_melspect(input_shape, conv_layers)

        print(output_text)

        # print("\nTrain shape: {0}".format(x_train.shape))
        # print("Validation shape: {0}".format(x_val.shape))
        # print("Test shape: {0}\n".format(x_test.shape))
        # print("Size of the CNN: %s\n" % cnn.count_params())

        # Compiler for the model
        cnn.compile(loss=create_loss_function(loss_function),
                    optimizer=create_optimizer(optimizer),
                    metrics=['accuracy'])

        # Early stop
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0,
                                                   patience=2,
                                                   verbose=0,
                                                   mode='auto')

        # Fit the model
        history = cnn.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_val, y_val),
                          callbacks=[early_stop])

        score = cnn.evaluate(x_test, y_test, verbose=0)
        score_val = cnn.evaluate(x_val, y_val, verbose=0)

        # Majority Voting System
        pred_values = np.argmax(cnn.predict(x_test), axis=1)
        mvs_truth, mvs_res = AudioUtils().voting(np.argmax(y_test, axis=1), pred_values)
        acc_mvs = accuracy_score(mvs_truth, mvs_res)

        # Save metrics
        val_acc.append(score_val[1])
        test_acc.append(score[1])
        test_history.append(history)
        test_acc_mvs.append(acc_mvs)

        # Print metrics
        print('Test accuracy:', score[1])
        print('Test accuracy for Majority Voting System:', acc_mvs)

        # Print the confusion matrix for Voting System
        cm = confusion_matrix(mvs_truth, mvs_res)
        print(cm)

    # Print the statistics
    print("Validation accuracy - mean: %s, std: %s" % (np.mean(val_acc), np.std(val_acc)))
    print("Test accuracy - mean: %s, std: %s" % (np.mean(test_acc), np.std(test_acc)))
    print("Test accuracy MVS - mean: %s, std: %s" % (np.mean(test_acc_mvs), np.std(test_acc_mvs)))

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    # plt.title('model accuracy exec: ' + str(exec_times)
    #           + ' - epochs: ' + str(epochs)
    #           + ' - opt: ' + str(optimizer)
    #           + ' - loss: ' + str(loss_function))
    plt.title('model accuracy: '+ str(output_text))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(results_folder + folder + '_model_accuracy_exec_' + str(exec_times) + '_epochs_' +
                str(epochs) + '_opt_' + optimizer + '_loss_' + str(loss_function) + '.png')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.title('model loss exec: ' + str(exec_times)
    #           + ' - epochs: ' + str(epochs)
    #           + ' - opt: ' + str(optimizer)
    #           + ' - loss: ' + str(loss_function))
    plt.title('model loss:'+ str(output_text))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(results_folder + folder + '_model_loss_exec_' + str(exec_times) + '_epochs_' +
                str(epochs) + '_opt_' + optimizer + '_loss_' + str(loss_function) + '.png')

    plt.close()

    # Save the model
    cnn.save(results_folder + model_path)
    h5f = h5py.File(results_folder + folder + '_exec_' + str(exec_times) + '_epochs_' + str(epochs) + '_opt_' + str(
        optimizer) + '_loss_' + str(loss_function) + '.h5', 'w')
    h5f.create_dataset('train_acc', data=history.history['acc'])
    h5f.create_dataset('test_acc', data=history.history['val_acc'])
    h5f.create_dataset('train_loss', data=history.history['loss'])
    h5f.create_dataset('test_loss', data=history.history['val_loss'])
    h5f.close()

    # Free memory
    del songs
    del genres
    del cnn
    del history

    gc.collect()


def create_npy(folder):
    song_rep = MelSpectrogram(folder + '/')
    songs, genres = song_rep.audio_to_array()

    np.save(folder + '/' + 'songs.npy', songs)
    np.save(folder + '/' + 'genres.npy', genres)

    return True


def create_optimizer(optimizer):
    if optimizer == 'sgd':
        return keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
    elif optimizer == 'adam':
        return keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
    elif optimizer == 'rmsprop':
        return keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    elif optimizer == 'adagrad':
        return keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    elif optimizer == 'adadelta':
        return keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    elif optimizer == 'adamax':
        return keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif optimizer == 'nadam':
        return keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    else:
        return keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)


def create_loss_function(loss_function):
    if loss_function == 'mean_squared_error':
        return keras.losses.mean_squared_error
    elif loss_function == 'mean_absolute_error':
        return keras.losses.mean_absolute_error
    elif loss_function == 'mean_absolute_percentage_error':
        return keras.losses.mean_absolute_percentage_error
    elif loss_function == 'mean_squared_logarithmic_error':
        return keras.losses.mean_squared_logarithmic_error
    elif loss_function == 'squared_hinge':
        return keras.losses.squared_hinge
    elif loss_function == 'hinge':
        return keras.losses.hinge
    elif loss_function == 'categorical_hinge':
        return keras.losses.categorical_hinge
    elif loss_function == 'logcosh':
        return keras.losses.logcosh
    elif loss_function == 'categorical_crossentropy':
        return keras.losses.categorical_crossentropy
    elif loss_function == 'binary_crossentropy':
        return keras.losses.binary_crossentropy
    elif loss_function == 'kullback_leibler_divergence':
        return keras.losses.kullback_leibler_divergence
    elif loss_function == 'poisson':
        return keras.losses.poisson
    elif loss_function == 'cosine_proximity':
        return keras.losses.cosine_proximity
    else:
        return keras.losses.categorical_crossentropy

def pre_def_main(conv_layers):
    results_folder = 'results_' + str(len(conv_layers))+'_'
    for index, conv_layer in enumerate(conv_layers, start=1):
        results_folder += str(conv_layer)
        if (index < len(conv_layers)):
            results_folder += '_'
    results_folder += '/'
    main(10, 200, 'adagrad', 'categorical_crossentropy', results_folder, conv_layers)

# if __name__ == '__main__':
#     main()

exs = [10]
eps = [200]
ops = ['adagrad']  # ['adadelta', 'adagrad', 'adam', 'adamax', 'nadam', 'rmsprop', 'sgd'] #
los = ['categorical_crossentropy'
    # 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
    # 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge',
    # 'logcosh', 'categorical_crossentropy',
    # 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity'
]

# mods = [[8, 16, 32, 16], [16, 32, 16], [16, 32, 64], [16, 32, 64, 32], [16, 32, 64, 128], [16, 64, 256], [16, 64, 256, 512]]

mods = [[16, 64, 256]]

resulting_folder = 'results_4/'

# for exec_time in exs:
#     for epoch in eps:
#         for opt in ops:
#             for loss in los:
#                 print('Run with exec_time: ' + str(exec_time) +
#                       ' - epochs: ' + str(epoch) +
#                       ' - opt: ' + str(opt) +
#                       ' - loss: ' + str(loss))
#                 main(exec_time, epoch, opt, loss, resulting_folder)

for mod in mods:
    # print(mod)
    pre_def_main(mod)

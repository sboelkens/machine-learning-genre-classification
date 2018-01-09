import gc
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras

from audiomanip.audiostruct import AudioStruct
from audiomanip.audiomodels import ModelZoo
from audiomanip.audioutils import AudioUtils

# Disable TF warnings about speed up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(exec_times = 10, epochs = 100, optimizer = 'adam'):
    # Configuration
    folder = 'gtzan'  # 'garageband'#
    results_folder = 'results_opts/'
    save_npy = True
    batch_size = 32

    epochs = 100
    exec_times = 10
    optimizer = 'adam'  # 'sgd'

    model_path = folder + '_exec_' + str(exec_times) + '_epochs_' + str(epochs) + '_opt_' + str(optimizer) + '.h5'

    # Read data
    data_type = 'NPY'  # 'AUDIO_FILES' #
    input_shape = (128, 128)
    print("data_type: %s" % data_type)

    if data_type == 'AUDIO_FILES':
        song_rep = AudioStruct(folder + '/')
        songs, genres = song_rep.getdata()

        # Save the audio files as npy files to read faster next time
        if save_npy:
            np.save(folder + '/' + 'songs.npy', songs)
            np.save(folder + '/' + 'genres.npy', genres)

    elif data_type == 'NPY':
        songs = np.load(folder + '/' + 'songs.npy')
        genres = np.load(folder + '/' + 'genres.npy')

    # Not valid datatype
    else:
        raise ValueError('Argument Invalid: The options are AUDIO_FILES or NPY for data_type')

    print("Original songs array shape: {0}".format(songs.shape))
    print("Original genre array shape: {0}".format(genres.shape))

    # Train multiple times and get mean score
    val_acc = []
    test_history = []
    test_acc = []
    test_acc_mvs = []

    for x in range(exec_times):
        # Split the dataset into training and test
        X_train, X_test, y_train, y_test = train_test_split(
            songs, genres, test_size=0.1, stratify=genres)

        # Split training set into training and validation
        X_train, X_Val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=1 / 6, stratify=y_train)

        print("X_Val: ", len(X_Val))
        print("y_val: ", len(y_val))
        # split the train, test and validation data in size 128x128
        X_Val, y_val = AudioUtils().splitsongs_melspect(X_Val, y_val, '1D')
        X_test, y_test = AudioUtils().splitsongs_melspect(X_test, y_test, '1D')
        X_train, y_train = AudioUtils().splitsongs_melspect(X_train, y_train, '1D')

        cnn = ModelZoo.cnn_melspect_1D(input_shape)

        print("\nTrain shape: {0}".format(X_train.shape))
        print("Validation shape: {0}".format(X_Val.shape))
        print("Test shape: {0}\n".format(X_test.shape))
        print("Size of the CNN: %s\n" % cnn.count_params())

        # Optimizers
        if optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
        elif optimizer == 'adam':
            opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        elif optimizer == 'adagrad':
            opt = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        elif optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        elif optimizer == 'adamax':
            opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        elif optimizer == 'nadam':
            opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        elif optimizer == 'tfoptimizer':
            opt = keras.optimizers.TFOptimizer(optimizer)
        else:
            opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)

        # Compiler for the model
        cnn.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=opt,
                    metrics=['accuracy'])

        # Early stop
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0,
                                                  patience=2,
                                                  verbose=0,
                                                  mode='auto')

        # Fit the model
        history = cnn.fit(X_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(X_Val, y_val),
                          callbacks=[earlystop])

        score = cnn.evaluate(X_test, y_test, verbose=0)
        score_val = cnn.evaluate(X_Val, y_val, verbose=0)

        # Majority Voting System
        pred_values = np.argmax(cnn.predict(X_test), axis=1)
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

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy exec: ' + str(exec_times) + ' - epochs: ' + str(epochs) + ' - opt: ' + str(optimizer))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(results_folder + folder + '_model_accuracy_exec_' + str(exec_times) + '_epochs_' + str(epochs) + '_opt_' + optimizer + '.png')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss exec: ' + str(exec_times) + ' - epochs: ' + str(epochs) + ' - opt: ' + str(optimizer))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(results_folder + folder + '_model_loss_exec_' + str(exec_times) + '_epochs_' + str(epochs) + '_opt_' + optimizer + '.png')

    # Save the model
    cnn.save(model_path)

    # Free memory
    del songs
    del genres
    gc.collect()

exec_times = [10, 20]
epochs = [100, 200]
opts = ['rmsprop', 'adagrad', 'adadelta', 'adam', 'sgd', 'adamax', 'nadam', 'tfoptimizer']

for exec_time in exec_times:
    for epoch in epochs:
        for opt in opts:
            main(exec_time, epoch, opt)

# if __name__ == '__main__':
#     main()
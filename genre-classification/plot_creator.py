import numpy as np
import h5py
import matplotlib.pyplot as plt


def make_plot(folder, h5py_files, names):

    plt.figure(1)
    count = 0
    for h5py_file in h5py_files:
        h5f = h5py.File(folder + h5py_file + '.h5', 'r')
        train_acc = h5f['train_acc'][:]
        test_acc = h5f['test_acc'][:]
        h5f.close()
        plt.plot(train_acc, color='C'+str(count))
        plt.plot(test_acc, ls='--', color='C'+str(count))
        count += 1

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    legend_names = []
    for name in names:
        legend_names.append('train_' + name)
        legend_names.append('test_' + name)
    plt.legend(legend_names, loc='upper left')
    # plt.savefig('results_combined/accuracy.png')

    plt.figure(2)
    count = 0
    for h5py_file in h5py_files:
        h5f = h5py.File(folder + h5py_file + '.h5', 'r')
        train_loss = h5f['train_loss'][:]
        test_loss = h5f['test_loss'][:]
        h5f.close()
        plt.plot(train_loss, color='C'+str(count))
        plt.plot(test_loss, ls='--', color='C'+str(count))
        count += 1

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    legend_names = []
    for name in names:
        legend_names.append('train_' + name)
        legend_names.append('test_' + name)
    plt.legend(legend_names, loc='upper left')
    # plt.savefig('results_combined/loss.png')

    plt.show()
    plt.close()


# h5_files = [
#     'gtzan_exec_10_epochs_100_opt_adam_loss_mean_absolute_error',
#     'gtzan_exec_10_epochs_100_opt_adam_loss_mean_absolute_percentage_error',
#     'gtzan_exec_10_epochs_100_opt_adam_loss_mean_squared_error',
#     'gtzan_exec_10_epochs_100_opt_adam_loss_mean_squared_logarithmic_error',
#     'gtzan_exec_10_epochs_100_opt_adam_loss_categorical_crossentropy',
#     'gtzan_exec_10_epochs_100_opt_adam_loss_categorical_hinge',
#     'gtzan_exec_10_epochs_100_opt_adam_loss_hinge',
#     'gtzan_exec_10_epochs_100_opt_adam_loss_logcosh'
# ]
# file_names = [
#     'mean_abs_error',
#     'mean_abs_perc_error',
#     'mean_squared_error',
#     'mean_squared_log_error',
#     'categorical_crossentropy',
#     'categorical_hinge',
#     'hinge',
#     'logcosh'
# ]
#
# make_plot('results_losses/', h5_files, file_names)



# h5_files = [
#     'gtzan_exec_10_epochs_100_opt_adadelta_loss_categorical_crossentropy',
#     'gtzan_exec_10_epochs_100_opt_adagrad_loss_categorical_crossentropy',
#     'gtzan_exec_10_epochs_100_opt_adam_loss_categorical_crossentropy',
#     'gtzan_exec_10_epochs_100_opt_adamax_loss_categorical_crossentropy',
#     'gtzan_exec_10_epochs_100_opt_nadam_loss_categorical_crossentropy',
#     'gtzan_exec_10_epochs_100_opt_rmsprop_loss_categorical_crossentropy',
#     'gtzan_exec_10_epochs_100_opt_sgd_loss_categorical_crossentropy'
# ]
# file_names = [
#     'adadelta',
#     'adagrad',
#     'adam',
#     'adamax',
#     'nadam',
#     'rmsprop',
#     'sgd'
# ]

# make_plot('results_opts/', h5_files, file_names)

h5_files = [
    'results_3_16_32_16/gtzan_exec_10_epochs_200_opt_adagrad_loss_categorical_crossentropy',
    'results_3_16_32_64/gtzan_exec_10_epochs_200_opt_adagrad_loss_categorical_crossentropy',
    'results_3_16_64_256/gtzan_exec_10_epochs_200_opt_adagrad_loss_categorical_crossentropy',
    'results_3_32_64_128/gtzan_exec_10_epochs_200_opt_adagrad_loss_categorical_crossentropy',
    'results_4_8_16_32_16/gtzan_exec_10_epochs_200_opt_adagrad_loss_categorical_crossentropy',
    'results_4_16_32_64_32/gtzan_exec_10_epochs_200_opt_adagrad_loss_categorical_crossentropy',
    'results_4_16_32_64_128/gtzan_exec_10_epochs_200_opt_adagrad_loss_categorical_crossentropy',
    'results_4_16_64_256_512/gtzan_exec_10_epochs_200_opt_adagrad_loss_categorical_crossentropy',
    'results_4_32_64_128_256/gtzan_exec_10_epochs_200_opt_adagrad_loss_categorical_crossentropy'
]
file_names = [
    'K 3: 16-32-16',
    'K 3: 16-32-64',
    'K 3: 16-64-256',
    'K 3: 32-64-128',
    'K 3: 64-128-256',
    'K 4: 8-16-32-16',
    'K 4: 16-32-64-32',
    'K 4: 16-32-64-128',
    'K 4: 16-64-256-512',
    'K 4: 32-64-128-256',
]

make_plot('', h5_files, file_names)
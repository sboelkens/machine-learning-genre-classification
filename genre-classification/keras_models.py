from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization


# @Class: KerasModels
# @Description: Set of models to use to solve the classification problem.
class KerasModels(object):
    @staticmethod
    def cnn_melspect(input_shape, conv_layers):
        kernel_size = len(conv_layers)
        print('conv_layers: ', conv_layers)
        print('kernel_size: ', kernel_size)
        # activation_func = LeakyReLU()
        activation_func = Activation('relu')
        inputs = Input(input_shape)
        pool = inputs
        bn = None

        for conv_layer in conv_layers:
            conv = Conv1D(conv_layer, kernel_size)(pool)
            act = activation_func(conv)
            bn = BatchNormalization()(act)
            pool = MaxPooling1D(pool_size=2, strides=2)(bn)

        # Global Layers
        gmaxpl = GlobalMaxPooling1D()(bn)
        gmeanpl = GlobalAveragePooling1D()(bn)
        mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

        # Regular MLP
        dense1 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(mergedlayer)
        actmlp = activation_func(dense1)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(reg)
        actmlp = activation_func(dense2)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(10, activation='softmax')(reg)

        model = Model(inputs=[inputs], outputs=[dense2])

        display_output = 'Kernel size '+str(len(conv_layers))+' - '
        for index, conv_layer in enumerate(conv_layers, start=1):
            display_output += str(conv_layer)
            if(index < len(conv_layers)):
                display_output += '/'

        return model, display_output

    @staticmethod
    def cnn_melspect_1(input_shape):
        kernel_size = 3
        # activation_func = LeakyReLU()
        activation_func = Activation('relu')
        inputs = Input(input_shape)

        # Convolutional block_1
        conv1 = Conv1D(32, kernel_size)(inputs)
        act1 = activation_func(conv1)
        bn1 = BatchNormalization()(act1)
        pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

        # Convolutional block_2
        conv2 = Conv1D(64, kernel_size)(pool1)
        act2 = activation_func(conv2)
        bn2 = BatchNormalization()(act2)
        pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)

        # Convolutional block_3
        conv3 = Conv1D(128, kernel_size)(pool2)
        act3 = activation_func(conv3)
        bn3 = BatchNormalization()(act3)

        # Global Layers
        gmaxpl = GlobalMaxPooling1D()(bn3)
        gmeanpl = GlobalAveragePooling1D()(bn3)
        mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

        # Regular MLP
        dense1 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(mergedlayer)
        actmlp = activation_func(dense1)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(reg)
        actmlp = activation_func(dense2)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(10, activation='softmax')(reg)

        model = Model(inputs=[inputs], outputs=[dense2])
        return model

    @staticmethod
    def cnn_melspect_2(input_shape):
        kernel_size = 3
        # activation_func = LeakyReLU()
        activation_func = Activation('relu')
        inputs = Input(input_shape)

        # Convolutional block_1
        conv1 = Conv1D(64, kernel_size)(inputs)
        act1 = activation_func(conv1)
        bn1 = BatchNormalization()(act1)
        pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

        # Convolutional block_2
        conv2 = Conv1D(128, kernel_size)(pool1)
        act2 = activation_func(conv2)
        bn2 = BatchNormalization()(act2)
        pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)

        # Convolutional block_3
        conv3 = Conv1D(256, kernel_size)(pool2)
        act3 = activation_func(conv3)
        bn3 = BatchNormalization()(act3)

        # Global Layers
        gmaxpl = GlobalMaxPooling1D()(bn3)
        gmeanpl = GlobalAveragePooling1D()(bn3)
        mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

        # Regular MLP
        dense1 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(mergedlayer)
        actmlp = activation_func(dense1)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(reg)
        actmlp = activation_func(dense2)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(10, activation='softmax')(reg)

        model = Model(inputs=[inputs], outputs=[dense2])
        return model

    @staticmethod
    def cnn_melspect_3(input_shape):
        kernel_size = 4
        # activation_func = LeakyReLU()
        activation_func = Activation('relu')
        inputs = Input(input_shape)

        # Convolutional block_1
        conv1 = Conv1D(32, kernel_size)(inputs)
        act1 = activation_func(conv1)
        bn1 = BatchNormalization()(act1)
        pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

        # Convolutional block_2
        conv2 = Conv1D(64, kernel_size)(pool1)
        act2 = activation_func(conv2)
        bn2 = BatchNormalization()(act2)
        pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)

        # Convolutional block_3
        conv3 = Conv1D(128, kernel_size)(pool2)
        act3 = activation_func(conv3)
        pool3 = BatchNormalization()(act3)

        # Convolutional block_4
        conv4 = Conv1D(256, kernel_size)(pool3)
        act4 = activation_func(conv4)
        bn4 = BatchNormalization()(act4)

        # Global Layers
        gmaxpl = GlobalMaxPooling1D()(bn4)
        gmeanpl = GlobalAveragePooling1D()(bn4)
        mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

        # Regular MLP
        dense1 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(mergedlayer)
        actmlp = activation_func(dense1)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(reg)
        actmlp = activation_func(dense2)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(10, activation='softmax')(reg)

        model = Model(inputs=[inputs], outputs=[dense2])
        return model

    @staticmethod
    def cnn_melspect_4(input_shape):
        kernel_size = 4
        # activation_func = LeakyReLU()
        activation_func = Activation('relu')
        inputs = Input(input_shape)

        # Convolutional block_1
        conv1 = Conv1D(8, kernel_size)(inputs)
        act1 = activation_func(conv1)
        bn1 = BatchNormalization()(act1)
        pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

        # Convolutional block_2
        conv2 = Conv1D(16, kernel_size)(pool1)
        act2 = activation_func(conv2)
        bn2 = BatchNormalization()(act2)
        pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)

        # Convolutional block_3
        conv3 = Conv1D(32, kernel_size)(pool2)
        act3 = activation_func(conv3)
        pool3 = BatchNormalization()(act3)

        # Convolutional block_4
        conv4 = Conv1D(64, kernel_size)(pool3)
        act4 = activation_func(conv4)
        bn4 = BatchNormalization()(act4)

        # Global Layers
        gmaxpl = GlobalMaxPooling1D()(bn4)
        gmeanpl = GlobalAveragePooling1D()(bn4)
        mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

        # Regular MLP
        dense1 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(mergedlayer)
        actmlp = activation_func(dense1)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(reg)
        actmlp = activation_func(dense2)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(10, activation='softmax')(reg)

        model = Model(inputs=[inputs], outputs=[dense2])
        return model, 'Kernel size 4 - 8/16/32/64'

    @staticmethod
    def cnn_melspect_urban_sound(input_shape):
        kernel_size = 4
        # activation_func = LeakyReLU()
        activation_func = Activation('relu')
        inputs = Input(input_shape)

        # Convolutional block_1
        conv1 = Conv1D(8, kernel_size)(inputs)
        act1 = activation_func(conv1)
        bn1 = BatchNormalization()(act1)
        pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

        # Convolutional block_2
        conv2 = Conv1D(16, kernel_size)(pool1)
        act2 = activation_func(conv2)
        bn2 = BatchNormalization()(act2)
        pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)

        # Convolutional block_3
        conv3 = Conv1D(32, kernel_size)(pool2)
        act3 = activation_func(conv3)
        pool3 = BatchNormalization()(act3)

        # Convolutional block_4
        conv4 = Conv1D(64, kernel_size)(pool3)
        act4 = activation_func(conv4)
        bn4 = BatchNormalization()(act4)

        # Global Layers
        gmaxpl = GlobalMaxPooling1D()(bn4)
        gmeanpl = GlobalAveragePooling1D()(bn4)
        mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

        # Regular MLP
        dense1 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(mergedlayer)
        actmlp = activation_func(dense1)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(512,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal')(reg)
        actmlp = activation_func(dense2)
        reg = Dropout(0.5)(actmlp)

        dense2 = Dense(10, activation='softmax')(reg)

        model = Model(inputs=[inputs], outputs=[dense2])
        return model, 'Kernel size 4 - 8/16/32/64'

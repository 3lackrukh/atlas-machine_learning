#!/usr/bin/env python3
""" Module defines the inception_network method """
from tensorflow import keras as K


def inception_network():
    """
    Builds the inception network as described in
    Going Deeper with Convolutions (2014)
    """
    inception_block = __import__('0-inception_block').inception_block

    init = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))

    convolution_1 = K.layers.Conv2D(filters=64,
                                    kernel_size=(7, 7),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=init)(inputs)

    max_pool_1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=(2, 2),
                                       padding='same')(convolution_1)

    convolution_2a = K.layers.Conv2D(filters=64,
                                     kernel_size=(1, 1),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer=init)(max_pool_1)

    convolution_2b = K.layers.Conv2D(filters=192,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer=init)(convolution_2a)

    max_pool_2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       gstrides=(2, 2),
                                       padding='same')(convolution_2b)

    inception_3a = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])
    max_pool_3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=(2, 2),
                                       padding='same')(inception_3b)

    inception_4a = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])
    max_pool_4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=(2, 2),
                                       padding='same')(inception_4e)

    inception_5a = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])
    avg_pool_5 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                           strides=(1, 1),
                                           padding='valid')(inception_5b)

    dropout_6 = K.layers.Dropout(rate=0.4)(avg_pool_5)

    softmax_7 = K.layers.Dense(units=1000,
                               activation='softmax',
                               kernel_initializer=init)(dropout_6)

    return K.Model(inputs=inputs, outputs=softmax_7)

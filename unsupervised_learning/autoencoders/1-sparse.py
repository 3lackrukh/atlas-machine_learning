#!/usr/bin/env python3
""" Module defines the autoencoder method """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ Creates a sparse autoencoder
    Parameters:
        input_dims: int dimensions of the model input
        hidden_layers: list of int nodes for each hidden layer
        latent_dims: int dimensions of the latent space layer
        lambtha: L1 regularization parameter
    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for i in hidden_layers:
        x = keras.layers.Dense(i,'relu')(x)
    encoder_outputs = keras.layers.Dense(latent_dims, 'relu',
                                         keras.regularizers.l1(lambtha))(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for i in reversed(hidden_layers):
        x = keras.layers.Dense(i, 'relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, 'sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    # Autoencoder
    autoencoder_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(encoder_inputs, autoencoder_outputs)

    auto.compile('adam', 'binary_crossentropy')

    return encoder, decoder, auto

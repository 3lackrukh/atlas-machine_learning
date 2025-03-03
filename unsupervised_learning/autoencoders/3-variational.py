#!/usr/bin/env python3
""" Module defines the autoencoder method """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    Parameters:
        input_dims: int dimensions of the model input
        hidden_layers: list of int nodes for each hidden layer
        latent_dims: int dimensions of the latent space layer
    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims)(x)  # No activation (None)
    z_log_var = keras.layers.Dense(latent_dims)(x)  # No activation (None)

    # Sampling layer
    def sampling(args):
        """ Sampling function """
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dims = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dims))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon
    
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    
    # Create encoder model
    encoder = keras.Model(encoder_inputs, [z, z_mean, z_log_var], name='encoder')
    
    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    # Autoencoder
    auto_inputs = keras.Input(shape=(input_dims,))
    _, z_mean, z_log_var = encoder(auto_inputs)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    outputs = decoder(z)
    auto = keras.Model(auto_inputs, outputs, name='vae')
    
    # Add KL divergence loss as a separate loss component
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var),
        axis=-1
    )
    kl_loss = keras.backend.mean(kl_loss)
    auto.add_loss(kl_loss)
    
    # Compile with binary_crossentropy as the main loss function
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
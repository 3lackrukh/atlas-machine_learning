#!/usr/bin/env python3
"""Module defines the gensim_to_keras method"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer

    Parameters:
        model: a trained gensim word2vec model

    Returns:
        the model's keras Embedding layer
    """
    # Create a keras Embedding layer with the same weights
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=model.wv.vectors.shape[0],
        output_dim=model.wv.vectors.shape[1],
        weights=[model.wv.vectors],
        trainable=True
    )

    return embedding_layer

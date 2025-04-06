#!/usr/bin/env python3
"""Module defines the 2-word2vec method"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim word2vec model

    Parameters:
        sentences: Lost of sentences to be trained on
        vector_size: Dimensionality of the embedding layer
        min_count: minimum number of occurences of a word for use in training
        window: maximum distance between current and predicted word
        negative: siz of negative sampling
        cbow: bool determining training type
            True: CBOW
            False: Skip-gram
        epochs: number of epochs to train over
        seed: seed for the random number generator
        workers: number of worker threads to train the model

    Returns:
        model: trained gensim word2vec model
    """
    model = gensim.models.Word2Vec(sentences, vector_size=vector_size,
                                   min_count=min_count, window=window,
                                   negative=negative, seed=seed,
                                   workers=workers, sg=not cbow)

    model.build_vocab(sentences)

    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model

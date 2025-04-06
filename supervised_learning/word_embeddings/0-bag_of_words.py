#!/usr/bin/env python3
"""Module defines the bag_of_words method"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix out of a list of sentences

    Parameters:
        sentences: List of sentences to analyze
        vocab: List of vocabulary words to use for analysis
            if None, all words in sentences will be used

    Returns:
        embeddings: numpy ndarray of shape(s, f) containing the embeddings
            s: number of sentences
            f: number of features analyzed
        features: List of features used for embeddings
    """
    if vocab is None:
        vocab = []
        for sentence in sentences:
            sentence = strip(sentence)
            for word in sentence.split():
                if word not in vocab:
                    vocab.append(word)
    vocab.sort()

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, sentence in enumerate(sentences):
        sentence = strip(sentence)
        for word in sentence.split():
            embeddings[i, vocab.index(word)] += 1

    return embeddings, vocab


def strip(sentence):
    """
    Strips possession and punctuation from a string

    Parameters:
        sentence: String to strip

    Returns:
        Stripped string
    """
    import string
    translator = str.maketrans('', '', string.punctuation)
    sentence = sentence.replace("'s", "")
    return sentence.lower().translate(translator)

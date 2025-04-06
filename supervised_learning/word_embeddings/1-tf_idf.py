#!/usr/bin/env python3
"""Module defines the tf_idf method"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix out of a list of sentences

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
    sentences = [strip(sentence) for sentence in sentences]

    if vocab is None:
        vocab = []
        for sentence in sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab.append(word)
        vocab.sort()

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=np.float32)

    # Bag of words embeddings (Term Frequency)
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1 / len(sentence.split())

    # Calculate document frequency from Term Frequency embeddings
    df = np.count_nonzero(embeddings > 0, axis=0)

    # Calculate inverse document frequency values
    idf = np.log((len(sentences) + 1) / (df + 1)) + 1

    # Calculate TF-IDF embeddings
    embeddings = embeddings * idf

    # Scale each embedding to unit length
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(embeddings, norm, where=norm != 0)

    return embeddings, np.array(vocab)


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

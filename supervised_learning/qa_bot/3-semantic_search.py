#!/usr/bin/env python3
"""Module defines the 3-semantic_search method"""
import os
import regex as re
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Parameters
        corpus_path: (str) the path to the corpus file
        sentence: (str) the sentence to search for
    Returns
        (str) the reference text of the document most similar to the sentence
    """
    # Load sentence encoder model
    # note: large version optimized for greater-than-word length text
    model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    s_encoder = hub.load(model_url)

    # Encode the query sentence
    q_embedding = s_encoder([sentence])

    max_similarity = -1
    most_similar_doc = ""

    # Encode and compare markdown files to query
    # note: model implements best-effort preprocessing
    for doc in os.listdir(corpus_path):
        if doc.endswith('.md'):
            doc_path = os.path.join(corpus_path, doc)
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_text = f.read()
                d_embedding = s_encoder([doc_text])

                # note: model normalizes embeddings to unit length
                # cosine similarity is inner product of normalized vectors
                similarity = np.inner(q_embedding, d_embedding)

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_doc = doc_text

    return most_similar_doc

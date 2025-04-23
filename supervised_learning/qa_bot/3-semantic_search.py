#!/usr/bin/env python3
"""Module defines the 3-semantic_search method"""
import os
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
    s_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Encode the query sentence
    s_embedding = s_encoder([sentence])
    
    max_similarity = -1
    most_similar_doc = ""
    
    # Read and encode corpus one at a time
    for doc in os.listdir(corpus_path):
        doc_path = os.path.join(corpus_path, doc)\
            
        # skip non-md files
        if not doc.endswith('.md'):
            continue

        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
            doc_embedding = s_encoder([doc_text])[0]

            # Calculate cosine similarity   
            similarity = np.dot(s_embedding, doc_embedding) / (
                np.linalg.norm(s_embedding) * np.linalg.norm(doc_embedding)
            )

            # Update most similar document if similarity is higher
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_doc = doc_text
                
    return most_similar_doc

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
    s_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Encode the query sentence
    q_embedding = s_encoder([sentence])
    
    max_similarity = -1
    most_similar_doc = ""
    
    # Read and encode corpus one at a time
    for doc in os.listdir(corpus_path):
        doc_path = os.path.join(corpus_path, doc)\
            
        # skip non-md files
        if not doc.endswith('.md'):
            continue
        
        print(f"Processing {doc}")
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
            
            # Split document into sentences with regex pattern
            s_pattern = r'(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
            sentences = re.split(s_pattern, doc_text)
            
            # Filter sentences shorter than 10 characters
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            print(f"Found {len(sentences)} sentences")
            if not sentences:
                continue
            
            s_embeddings = s_encoder(sentences)
            
            # Calculate cosinse similarity for each sentence in the doc
            s_similarities = []
            for s_embedding in s_embeddings:
                similarity = np.dot(q_embedding, s_embedding) / (
                    np.linalg.norm(q_embedding) * np.linalg.norm(s_embedding)
                )
                s_similarities.append(float(similarity))
                
            # Get mean similarity for the document
            similarity = np.mean(s_similarities)
            print(f"Similarity to query: {similarity}")
            # Update most similar document if similarity is higher
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_doc = doc_text
                print(f"New most similar doc: {doc}")
    
    print(f" Most similar document: {most_similar_doc}")
                
    return most_similar_doc

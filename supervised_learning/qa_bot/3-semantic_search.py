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
    # large version optimized for greater-than-word length text
    s_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    
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
            s_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
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
                
            # Weighted similarity scoring
            # best match (60% weight)
            # top 3 average (30% weight)
            # total average (10% weight)
            sorted_similarities = sorted(s_similarities, reverse=True)
            
            best_match = sorted_similarities[0]
            top_3_avg = np.mean(sorted_similarities[:3]) if len(sorted_similarities) >= 3 else best_match
            total_avg = np.mean(sorted_similarities)
            similarity = (0.6 * best_match) + (0.3 * top_3_avg) + (0.1 * total_avg)
            print(f"Weighted similarity to query: {similarity}")
            
            # Update most similar document if similarity is higher
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_doc = doc_text
                print(f"New most similar doc: {doc}")
                
    return most_similar_doc
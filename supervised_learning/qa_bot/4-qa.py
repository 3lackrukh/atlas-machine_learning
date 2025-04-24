#!/usr/bin/env python3
"""Module defines the question_answer and answer_loop methods"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

# Global variables
_s_encoder = None
_bert_model = None
_tokenizer = None
_doc_embeddings = None
_doc_texts = []
_embeddings_loaded = False


def load_models():
    """
    Loads the required models into global scope
    for semantic_search and document_search.
    """
    global _s_encoder, _bert_model, _tokenizer
    
    
    _s_encoder = hub.load("https://tfhub.dev/google/"
                          "universal-sentence-encoder-large/5")
    
    _bert_model = hub.load("https://www.kaggle.com/models/seesee/bert/"
                          "TensorFlow2/uncased-tf2-qa/1")
    
    _tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-"
                                              "masking-finetuned-squad")


def load_embeddings(corpus_path):
    """
    Loads document embeddings extracted from the corpus into global scope.

    Parameters
        corpus_path: (str) the path to the corpus file
    """
    global _doc_embeddings, _doc_texts, _embeddings_loaded
    
    if _embeddings_loaded:
        return

    cache_dir = os.path.join(corpus_path, ".cache")
    doc_embeddings_cache = os.path.join(cache_dir, "doc_embeddings.npy")
    doc_texts_cache = os.path.join(cache_dir, "doc_texts.npy")
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    if os.path.exists(doc_embeddings_cache) and os.path.exists(doc_texts_cache):
        # Load embeddings from cache
        _doc_embeddings = np.load(doc_embeddings_cache, allow_pickle=True)
        _doc_texts = np.load(doc_texts_cache, allow_pickle=True).tolist()
    else: # Extract embeddings from corpus
        doc_texts = []
        for doc in os.listdir(corpus_path):
            if doc.endswith(".md"):
                with open(os.path.join(corpus_path, doc), "r", encoding="utf-8") as f:
                    doc_texts.append(f.read())
                    
    # Encode documents in batches
    batch_size = 16
    doc_embeddings = []
        
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i:i + batch_size]
        batch_embeddings = _s_encoder(batch)
        doc_embeddings.append(batch_embeddings.numpy())
        
    # Concatenate all batch embeddings
    _doc_embeddings = np.vstack(doc_embeddings)
    _doc_texts = doc_texts
    
    # Save embeddings to cache
    np.save(doc_embeddings_cache, _doc_embeddings)
    np.save(doc_texts_cache, np.array(_doc_texts, dtype=object))
    
    _embeddings_loaded = True


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Parameters
        corpus_path: (str) the path to the corpus file
        sentence: (str) the sentence to search for
    Returns
        (str) the reference text of the document most similar to the sentence
    """
    global _s_encoder, _doc_embeddings, _doc_texts
    
    # Load models if not loaded yet
    if _s_encoder is None:
        load_models()
        
    # Load or create document embeddings
    load_embeddings(corpus_path)
    
    # Encode the query sentence
    q_embedding = _s_encoder([sentence]).numpy()
    
    # Compute similarities (vectorized)
    similarities = np.inner(q_embedding, _doc_embeddings)[0]
    
    # Find the document with highest similarity
    max_idx = np.argmax(similarities)
    
    return _doc_texts[max_idx]


def document_search(question, reference):
    """
    Finds a relevant snippet from reference text to answer a question.

    Parameters
        question: (str) the question to answer
        reference: (str) the text to search for the answer
    Returns
        answer: (str) the answer to the question
    """
    # Load the pre-trained BERT model and tokenizer
    global _bert_model, _tokenizer
    
    if _bert_model is None:
        load_models()

    # Preprocess the input and reference
    q_tokens = _tokenizer.tokenize(question)
    r_tokens = _tokenizer.tokenize(reference)

    # Build full token sequence with special tokens
    tokens = ["[CLS]"] + q_tokens + ["[SEP]"] + r_tokens + ["[SEP]"]

    # Convert tokens to IDs
    input_word_ids = _tokenizer.convert_tokens_to_ids(tokens)

    # Create input mask (1 for real tokens, 0 for padding)
    input_mask = [1] * len(input_word_ids)

    # Create token type IDs (0 for question, 1 for reference)
    input_type_ids = [0] * (len(q_tokens) + 2) + [1] * (len(r_tokens) + 1)

    # Convert to tensors and add batch dimension
    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids)
    )

    # Get the model's output
    outputs = _bert_model([input_word_ids, input_mask, input_type_ids])

    # Find start and end positions
    short_start = tf.argmax(outputs[0][0, 1:]) + 1
    short_end = tf.argmax(outputs[1][0, 1:]) + 1

    # Handle case where no answer is found
    if short_start >= short_end:
        return None

    # Extract answer tokens and convert to string
    answer_tokens = tokens[short_start: short_end + 1]
    answer = _tokenizer.convert_tokens_to_string(answer_tokens)

    return answer


def question_answer(corpus_path):
    """
    Leverages document_search to answer questions from the user.

    Parameters
        corpus_path: (str) the path to the corpus directory
    """
    # Load models if not loaded yet
    if _s_encoder is None:
        load_models()

    # Preload embeddings for faster first response
    load_embeddings(corpus_path)

    while True:
        question = input("Q: ")
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        
        # Measure performance
        import time
        start_time = time.time()
        
        # Find most relevant document and extract answer
        document = semantic_search(corpus_path, question)
        answer = document_search(question, document)
        
        # Print performance metrics
        total_time = time.time() - start_time
        print("Query time:", total_time, "seconds")
        
        if answer is not None:
            print("A:", answer)

        else:
            print("A: Sorry, I do not understand your question.")

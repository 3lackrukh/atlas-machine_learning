#!/usr/bin/env python3
"""Module defines the question_answer method"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np


def question_answer(question, reference):
    """
    Finds a snippet of text containing the answer to a question.
    
    Parameters
        question: (str) the question to answer
        reference: (str) the text to search for the answer
    Returns
        answer: (str) the answer to the question 
    """
    # Load the pre-trained BERT model and tokenizer
    bert_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    
    # Preprocess the input and reference
    inputs = tokenizer(question, reference, add_special_tokens=True, return_tensors="tf")

    # Get the input IDs, attention mask, and token type IDs
    # Note: The tokenizer returns a dictionary with input_ids, attention_mask, and token_type_ids
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    
    # Pass inputs through the BERT model
    outputs = bert_model([input_ids, attention_mask, token_type_ids])
    start_logits = outputs[0]
    end_logits = outputs[1]
    
    # Find the highest probability start and end indices
    start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
    end_idx = tf.argmax(end_logits, axis=1).numpy()[0]
    
    # Get tokens from these indices
    tokens = input_ids[0]
    answer_tokens = tokens[start_idx:end_idx + 1]
    
    # Convert tokens to string
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    print(type(answer))
    print(answer)
    return answer

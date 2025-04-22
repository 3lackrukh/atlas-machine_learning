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
    bert_model = hub.load("https://www.kaggle.com/models/seesee/bert/TensorFlow2/uncased-tf2-qa/1")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    
    # Preprocess the input and reference
    inputs = tokenizer.encode_plus(question, reference, add_special_tokens=True, return_tensors="tf")

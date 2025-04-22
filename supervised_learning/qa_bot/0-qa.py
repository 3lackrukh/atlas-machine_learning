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
    bert_model = hub.load("https://www.kaggle.com/models/seesee/bert/"
                          "TensorFlow2/uncased-tf2-qa/1")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-"
                                              "masking-finetuned-squad")

    # Preprocess the input and reference
    q_tokens = tokenizer.tokenize(question)
    r_tokens = tokenizer.tokenize(reference)

    # Build full token sequence with special tokens
    tokens = ["[CLS]"] + q_tokens + ["[SEP]"] + r_tokens + ["[SEP]"]

    # Convert tokens to IDs
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

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
    outputs = bert_model([input_word_ids, input_mask, input_type_ids])

    # Find start and end positions
    # don't forget the [CLS] token
    short_start = tf.argmax(outputs[0][0, 1:]) + 1
    short_end = tf.argmax(outputs[1][0, 1:]) + 1

    # Handle case where no answer is found
    if short_start >= short_end:
        print("Sorry, I couldn't find an answer to your question.")
        return None

    # Extract answer tokens and convert to string
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer

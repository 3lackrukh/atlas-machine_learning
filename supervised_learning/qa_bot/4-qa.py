#!/usr/bin/env python3
"""Module defines the question_answer and answer_loop methods"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
semantic_search = __import__('3-semantic_search').semantic_search


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
        return None

    # Extract answer tokens and convert to string
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer


def question_answer(corpus_path):
    """
    Leverages the question_answer method to answer questions from the user.

    Parameters
        reference: (str) the text to search for the answers
    """
    while True:
        question = input("Q: ")
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        document = semantic_search(corpus_path, question)
        answer = document_search(question, document)
        if answer is not None:
            print("A:", answer)

        else:
            print("A: Sorry, I do not understand your question.")

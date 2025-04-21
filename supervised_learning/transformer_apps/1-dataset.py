#!/usr/bin/env python3
"""Module defines the Dataset class"""
import transformers
import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """class Dataset to load and preps a dataset for machine translation"""

    def __init__(self):
        """class consrtuctor"""
        self.vocab_size = 2**13
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)

        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates subword tokens for our dataset

        Parameters:
            data: tf.data.Dataset whose examples are formatted as tuple(pt, en)
                pt: tf.Tensor containing the Portuguese sentence
                en: tf.Tensor containing the corresponding English sentence
        Returns:
            tokenizer_pt: the tokenizer used for the Portuguese sentences
            tokenizer_en: the tokenizer used for the English sentences
        """
        # Extract and decode sentences into lists
        pt_sentences = [pt.numpy().decode('utf-8') for pt, _ in data]
        en_sentences = [en.numpy().decode('utf-8') for _, en in data]

        # Initialize and train pretrained tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
            ).train_new_from_iterator(pt_sentences, vocab_size=self.vocab_size)

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
            ).train_new_from_iterator(en_sentences, vocab_size=self.vocab_size)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens

        Parameters:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence
        Returns:
            pt_tokens: np.ndarray containing the Portuguese tokens
            en_tokens: np.ndarray containing the English tokens
        """
        # Decode tensors into strings
        pt = pt.numpy().decode('utf-8')
        en = en.numpy().decode('utf-8')

        # Encode sentences into tokens
        pt_tokens = self.tokenizer_pt.encode(pt, return_tensors='np').squeeze()
        en_tokens = self.tokenizer_en.encode(en, return_tensors='np').squeeze()

        # Replace the start and end tokens
        pt_tokens[0] = self.vocab_size
        en_tokens[0] = self.vocab_size
        pt_tokens[-1] = self.vocab_size + 1
        en_tokens[-1] = self.vocab_size + 1
        
        return pt_tokens, en_tokens

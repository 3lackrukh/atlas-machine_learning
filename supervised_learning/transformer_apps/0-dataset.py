#!/usr/bin/env python3
"""Module defines the Dataset class"""
import transformers
import tensorflow_datasets as tfds


class Dataset:
    """class Dataset to load and preps a dataset for machine translation"""

    def __init__(self):
        """class consrtuctor"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)

        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True)

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
        # Initialize pretrained tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            vocab_size=2**15,
            language="pt")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            vocab_size=2**15,
            language="en")

        return tokenizer_pt, tokenizer_en

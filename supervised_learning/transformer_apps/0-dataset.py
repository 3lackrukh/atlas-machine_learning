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
        
        # Set vocab size according to specified parameters
        vocab_size = 2**13
        
        # Initialize pretrained tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
            ).train_new_from_iterator(pt_sentences, vocab_size=vocab_size)
        
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
            ).train_new_from_iterator(en_sentences, vocab_size=vocab_size)

        return tokenizer_pt, tokenizer_en

#!/usr/bin/env python3
"""Module defines the uni_bleu method"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    
    Parametrs:
        references: list of reference translations
            each reference translation is a list of the words in the translation
        sentence: list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    # Get total number of candidate words
    len_sen = len(sentence)


    # Initialize a dictionary to store the count of each word in the sentence
    sentence_dict = {}

    # Iterate over each word in the sentence
    for word in sentence:
        # Add each new word to the dictionary
        if word not in sentence_dict.keys():
            sentence_dict[word] = 0
        # Increment the count of the word
        sentence_dict[word] += 1

    print(f'sentence_dict: {sentence_dict}')
    
    # Dictionary to store max count of any word in any reference
    refs_dict = {}

    # Iterate over each word in each reference
    for reference in references:
        # Dictionary to store count of each word in reference
        ref_dict = {}
        for word in reference:
            if word not in ref_dict.keys():
                ref_dict[word] = 0
            # Update the maximum count of the word
            ref_dict[word] += 1
        # Update the maximum count of the word in refs_dict
        for word, count in ref_dict.items():
            refs_dict[word] = max(refs_dict.get(word, 0), count)
        print(f'ref_dict: {ref_dict}')
        print(f'refs_dict: {refs_dict}')



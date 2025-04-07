#!/usr/bin/env python3
"""Module defines the uni_bleu method"""
import numpy as np
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence

    Parametrs:
        references: list of reference translations
            each reference translation is a list of words in the translation
        sentence: list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    # Get candidate length and reference lengths
    len_sen = len(sentence)
    ref_lens = [len(reference) for reference in references]

    # Find effective reference length
    # The closest reference length to the candidate length
    closest_ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - len_sen))

    # Calculate brevity penalty
    if len_sen >= closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - (closest_ref_len / len_sen))

    # Count words in candidate sentence``
    sentence_counts = Counter(sentence)

    # Dictionary to store max count of any word in any reference
    ref_max_counts = {}

    # Get max word counts across references
    for reference in references:
        ref_counts = Counter(reference)
        for word, count in ref_counts.items():
            ref_max_counts[word] = max(ref_max_counts.get(word, 0), count)

    # Calculate and store clipped counts
    clipped_count = sum(min(count, ref_max_counts.get(word, 0))
                        for word, count in sentence_counts.items())

    # Calculate precision
    precision = clipped_count / len_sen

    return brevity_penalty * precision

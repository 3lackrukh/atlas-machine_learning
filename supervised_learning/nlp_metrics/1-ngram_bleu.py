#!/usr/bin/env python3
"""Module defines the ngram_bleu method"""
import numpy as np
from collections import Counter


def get_ngrams(sentence, n):
    """
    Generates n-grams from a sentence

    Parameters:
        sentence: list of words in the sentence
        n: size of n-grams to generate

    Returns:
        list of n-grams
    """
    ngrams = []
    for i in range(len(sentence) - n + 1):
        ngrams.append(tuple(sentence[i:i + n]))
    return ngrams
    
    
def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    Parametrs:
        references: list of reference translations
            each reference translation is a list of words in the translation
        sentence: list containing the model proposed sentence
        n: size of n-gram to use for evaluation

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

    # Count n-grams in candidate sentence and references    
    sentence_ngrams = get_ngrams(sentence, n)
    ref_ngrams = [get_ngrams(reference, n) for reference in references]
    print(f'sentence_ngrams: {sentence_ngrams}')
    print(f'ref_ngrams: {ref_ngrams}')
    
    # Count n-grams in candidate sentence
    sentence_counts = Counter(sentence_ngrams)
    print(f'sentence_counts: {sentence_counts}')

    # Dictionary to store max count of any word in any reference
    ref_max_counts = {}

    # Get max word counts across references
    for reference in ref_ngrams:
        ref_counts = Counter(reference)
        print(f'ref_counts: {ref_counts}')
        for ngram, count in ref_counts.items():
            ref_max_counts[ngram] = max(ref_max_counts.get(ngram, 0), count)
    
    print(f'ref_max_counts: {ref_max_counts}')

    # Calculate and store clipped counts
    clipped_count = sum(min(count, ref_max_counts.get(ngram, 0))
                        for ngram, count in sentence_counts.items())
    
    print(f'clipped_count: {clipped_count}')

    # Calculate precision
    precision = clipped_count / len(sentence_ngrams) if len(sentence_ngrams) > 0 else 0

    return brevity_penalty * precision

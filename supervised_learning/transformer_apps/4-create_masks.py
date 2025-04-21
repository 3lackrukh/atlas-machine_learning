#!/usr/bin/env python3
"""Module defines the create_masks method"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation

    Parameters:
        inputs: tf.Tensor of shape (batch_size, seq_len_in)
            containing the input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out)
            containing the target sentence

    Returns:
        encoder_mask: tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in)

        combined_mask: tf.Tensor of shape
            (batch_size, 1, seq_len_out, seq_len_out)
            the maximum between lookahaed_mask and target padding mask

        decoder_mask: tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in)
            the cross attention mask for the decoder

    """
    # Get sequence output length
    seq_len_out = tf.shape(target)[1]

    # Create encoder padding mask
    # expand dimensions for multi-head attention
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Create look ahead mask for second attention block
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones(
        (seq_len_out, seq_len_out)), -1, 0)

    # Create target padding mask
    target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_mask = target_mask[:, tf.newaxis, tf.newaxis, :]

    # Combine the masks
    combined_mask = tf.maximum(target_mask, look_ahead_mask)

    # note - decoder mask is the same as encoder mask
    # as both mask padding tokens in the input sequence
    return encoder_mask, combined_mask, encoder_mask

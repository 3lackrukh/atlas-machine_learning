#!/usr/bin/env python3
""" Module defines the train method """
import tensorflow.compat.v1 as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, y_train, X_valid, y_valid, layer_sizes, activations, alpha,
          iterations, save_path='./model.ckpt'):
    """
    Builds, trains, and saves a neural network classifier.

    Parameters:
    X_train: Training input data.
    y_train: Training target data.
    X_valid: Validation input data.
    y_valid: Validation target data.
    layer_sizes: List of integers representing number of nodes in each layer.
    activations: List of strings representing activation function for ea layer.
    alpha: Learning rate for the optimization algorithm.
    iterations: Number of iterations to train the model.
    save_path: Path to save the trained model.
    """
    # Create placeholders for input data and labels
    x, y = create_placeholders(X_train.shape[1], y_train.shape[1])

    # Build the neural network model
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create training operation
    train_op = create_train_op(loss, alpha)

    # Add operations to the graph's collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize variables and create a saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Training loop
        for i in range(iterations + 1):
            # Calculate training cost and accuracy
            t_cost, t_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_train, y: y_train})

            # Calculate validation cost and accuracy
            v_cost, v_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_valid, y: y_valid})

            # Print progress at specified intervals
            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {t_cost}")
                print(f"\tTraining Accuracy: {t_accuracy}")
                print(f"\tValidation Cost: {v_cost}")
                print(f"\tValidation Accuracy: {v_accuracy}")

            # Perform training step if not at the last iteration
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: y_train})

        # Save the model and return the save path
        return saver.save(sess, save_path)

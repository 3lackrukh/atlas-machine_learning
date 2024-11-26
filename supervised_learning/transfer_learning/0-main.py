#!/usr/bin/env python3

from tensorflow import keras as K
import tensorflow as tf
preprocess_data = __import__('0-transfer').preprocess_data
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

# Define the custom LinearDecay class that was used in training
class LinearDecay(LearningRateSchedule):
    def __init__(self, initial_lr, final_lr, decay_steps):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_steps = decay_steps

    def __call__(self, step):
        lr = self.initial_lr - ((self.initial_lr - self.final_lr)
                                / K.backend.cast(self.decay_steps, 'float32'))\
                * K.backend.cast(step, 'float32')
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "final_lr": self.final_lr,
            "decay_steps": self.decay_steps
        }

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)

# Load the model with custom_objects
with K.utils.custom_object_scope({'LinearDecay': LinearDecay}):
    model = K.models.load_model('cifar10.h5')

model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_loader import get_test_image
change_contrast = __import__('3-contrast').change_contrast

tf.random.set_seed(0)

# Get test image with seed 0 (matches original test)
image = get_test_image(seed=0)
plt.imshow(change_contrast(image, 0.5, 3))
plt.show()
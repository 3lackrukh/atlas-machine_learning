#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_loader import get_test_image
rotate_image = __import__('2-rotate').rotate_image

# get test image with seed 2 (matches original test)
image = get_test_image(seed=2)
plt.imshow(rotate_image(image))
plt.show()
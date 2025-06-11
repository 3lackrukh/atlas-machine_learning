#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_loader import get_test_image
crop_image = __import__('1-crop').crop_image

# Get test image with seed 1 (matches original test)
image = get_test_image(seed=1)
plt.imshow(crop_image(image, (200, 200, 3)))
plt.show()

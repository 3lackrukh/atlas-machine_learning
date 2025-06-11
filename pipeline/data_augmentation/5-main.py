#!/usr/bin/env python3
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_loader import get_test_image
change_hue = __import__('5-hue').change_hue

image = get_test_image(seed=5)
plt.imshow(change_hue(image, 0.5))
plt.show()
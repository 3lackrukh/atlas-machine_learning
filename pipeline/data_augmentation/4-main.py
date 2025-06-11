#!/usr/bin/env python3
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_loader import get_test_image
change_brightness = __import__('4-brightness').change_brightness

image = get_test_image(seed=4)
plt.imshow(change_brightness(image, 0.3))
plt.show()
#!/usr/bin/env python3

import matplotlib.pyplot as plt
from dataset_loader import get_test_image
flip_image = __import__('0-flip').flip_image

# Get test image with seed 0 (matches original test)
image = get_test_image(seed=0)
plt.imshow(flip_image(image))
plt.show()

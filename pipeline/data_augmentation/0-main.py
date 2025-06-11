#!/usr/bin/env python3

import matplotlib.pyplot as plt
from dataset_loader import get_test_image
flip_image = __import__('0-flip').flip_image

# Get test image (dataset loaded only once across all main files)
image = get_test_image()
plt.imshow(flip_image(image))
plt.show()

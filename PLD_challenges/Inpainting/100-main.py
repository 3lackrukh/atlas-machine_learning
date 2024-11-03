#!/usr/bin/env python3
from inpaint import damage_mask, mask_to_image, visualize_result, interpolation_repair, telea_repair, navier_strokes_repair, biharmonic_repair
import matplotlib.pyplot as plt
import cv2
image_path = 'cat_damaged.png'

# Create mask
image, mask, neighbor_count = damage_mask(
    image_path,
    patch_size=(3, 3),
    threshold=3
)

# Visualize results
visualize_result(image, mask, neighbor_count)

# Apply mask to image
result = mask_to_image(image, mask)

# Display final result
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Image with Damage Highlighted')
plt.axis('off')
plt.show()

# Repair image using different techniques
repaired_image_interpolation = interpolation_repair(image, mask)
repaired_image_telea = telea_repair(image, mask)
repaired_image_navier_strokes = navier_strokes_repair(image, mask)
# repaired_image_kriging = kriging_repair(image, mask)
repaired_image_biharmonic = biharmonic_repair(image, mask)
plt.figure(figsize=(15, 5))
plt.subplot(231)
plt.imshow(cv2.cvtColor(repaired_image_interpolation, cv2.COLOR_BGR2RGB))
plt.title('Interpolation Repair')
plt.axis('off')
plt.subplot(232)
plt.imshow(cv2.cvtColor(repaired_image_telea, cv2.COLOR_BGR2RGB))
plt.title('Telea Repair')
plt.axis('off')
plt.subplot(233)
plt.imshow(cv2.cvtColor(repaired_image_navier_strokes, cv2.COLOR_BGR2RGB))
plt.title('Navier Strokes Repair')
plt.axis('off')
# plt.subplot(234)
# plt.imshow(cv2.cvtColor(repaired_image_kriging, cv2.COLOR_BGR2RGB))
# plt.title('Kriging Repair')
# plt.axis('off')
plt.subplot(235)
plt.imshow(cv2.cvtColor(repaired_image_biharmonic, cv2.COLOR_BGR2RGB))
plt.title('Biharmonic Repair')
plt.axis('off')
plt.show()
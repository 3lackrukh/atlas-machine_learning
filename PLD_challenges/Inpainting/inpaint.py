#!/usr/bin/env python3

import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import cv2
from scipy.ndimage import convolve
from scipy.interpolate import griddata
from scipy.sparse import linalg, csr_matrix
import matplotlib.pyplot as plt

def damage_mask(img_path, patch_size=(3,3), threshold=3):
    """
    Generate a damage mask for the given image using a simple thresholding approach.

    Args:
        img_path (str): Path to the input image.
        patch_size (tuple, optional): Size of the patches for feature extraction. Defaults to (3, 3).
        threshold (int, optional): Threshold value for determining damaged regions. Defaults to 4.

    Returns:
        numpy.ndarray: Damage mask for the input image.
    """
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image path: {}".format(img_path))

    # Convert the image to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create initial mask for black pixels
    black_mask = gray == 0
    
    # Create kernel for counting neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_ct = convolve(black_mask.astype(np.uint8), kernel)

    # Create damage mask based on threshold
    damage_mask = black_mask & (neighbor_ct >= threshold)
 
    damage_mask_unit8 = (damage_mask).astype(np.uint8) * 255
    
    return img, damage_mask_unit8, neighbor_ct

def visualize_result(image, mask, neighbor_count):
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Damage Mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(neighbor_count, cmap='hot')
    plt.title('Neighbor Count Heatmap')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def mask_to_image(image, mask):
    bool_mask = mask.astype(bool)
    result= image
    
    if len(image.shape) == 3:
        result[bool_mask] = [0, 0, 255]
    else:
        result[bool_mask] = 255
        
    return result

def interpolation_repair(image, mask):
    # Get coordinates of non-maked pixels
    non_masked_coords = np.argwhere(~mask)
    known_coords = np.column_stack((non_masked_coords[:, 0], non_masked_coords[:, 1]))
    known_values = image[known_coords[:, 0], known_coords[:, 1]]
    
    # Get coordinates of masked pixels
    repair_rows, repair_cols = np.where(mask)
    repair_coords = np.column_stack((repair_rows, repair_cols))
    
    #interpolate values
    repaired_values = griddata(known_coords,
                               known_values,
                               repair_coords,
                               method='cubic',
                               fill_value=np.mean(known_values))
    # create output image
    result = image.copy()
    result[mask] = repaired_values
    return result

def telea_repair(image, mask, radius=3):
    return cv2.inpaint(image, mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)

def navier_strokes_repair(image, mask, radius=3):
    return cv2.inpaint(image, mask.astype(np.uint8), radius, cv2.INPAINT_NS)

def kriging_repair(image, mask, kernel_size=10):
    # Get coordinates of known and unknown pixels
    rows, cols = np.where(~mask)
    known_coords = np.column_stack((rows, cols))
    known_values = image[~mask]
    
    repair_rows, repair_cols = np.where(mask)
    repair_coords = np.column_stack((repair_rows, repair_cols))
    
    # Define and fit Gausian process
    kernel = RBF(length_scale=kernel_size)
    gp = GaussianProcessRegressor(kernel=kernel)
    
    # Fit subset if too many points
    if len(known_coords) > 1000:
        subset_indices = np.random.choice(len(known_coords), size=1000, replace=False)
        known_coords = known_coords[subset_indices]
        known_values = known_values[subset_indices]
    else:
        gp.fit(known_coords, known_values)

    # Predict values for masked regions
    repaired_values = gp.predict(repair_coords)
    
    # Create output image
    result = image.copy()
    result[mask] = repaired_values
    return result

def biharmonic_repair(image, mask, alpha=1e-6):
    height, width = image.shape[:2]
    n = height * width
    
    #Create_laplacian operator
    def create_laplacian_matrix(height, width):
        n = height * width
        diagonals = [
            -4 * np.ones(n),
            np.ones(n - 1),
            np.ones(n - width),
            np.ones(n - width - 1)
        ]

        offsets = [0, 1, -1, width, -width]
        return csr_matrix((diagonals, offsets), shape=(n, n))
    
    # Create and solve biharmonic system
    L = create_laplacian_matrix(height, width)
    L2 = L.dot(L)
    
    A = L2 + alpha * csr_matrix(
        (mask.flatten(), (range(n), range(n)))
    )
    b = alpha * image.flatten() * mask.flatten()
    
    x = linalg.solve(A, b)
    return x.reshape((height, width))
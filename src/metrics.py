import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def calculate_mse_roi(original, processed, mask=None):
    """Calculates MSE only within the masked ROI."""
    if mask is None:
        return np.mean((original - processed) ** 2)
    
    # Use the mask to select only tissue pixels
    error = (original[mask > 0] - processed[mask > 0]) ** 2
    return np.mean(error) if error.size > 0 else 0

def create_tissue_mask(img, threshold=0.05, exclude_corner=True):
    """
    Isolates biological tissue and optionally removes 
    the clinical UI box in the lower right corner.
    """
    mask = (img > threshold).astype(np.uint8)
    
    if exclude_corner:
        h, w = mask.shape
        # Define the box: Last 20% of height and Last 25% of width
        # You can adjust these percentages based on your specific images
        box_h = int(h * 0.8)
        box_w = int(w * 0.75)
        mask[box_h:, box_w:] = 0 
        
    # Fill small holes to keep the anatomy solid
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def calculate_psnr(original, processed):
    """Calculates Peak Signal-to-Noise Ratio."""
    mse = calculate_mse_roi(original, processed)
    if mse == 0: return 100
    # Since our data_prep.py standardizes to [0, 1]
    max_pixel = 1.0 
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def binarize_edge_map(edge_map, method='mean'):
    """
    Converts a gradient magnitude image (Sobel/Prewitt) into a binary edge map.
    Essential for Pratt's FOM calculation.
    """
    if method == 'mean':
        threshold = np.mean(edge_map) + np.std(edge_map)
    else:
        # Simple percentage of max intensity
        threshold = 0.3 * np.max(edge_map)
    
    return (edge_map > threshold).astype(np.uint8)

def pratt_fom(detected_edges, ground_truth, alpha=0.11):
    """
    Calculates Pratt's Figure of Merit.
    'detected_edges' and 'ground_truth' must be binary (0 or 1).
    """
    # Ensure inputs are binary
    detected_edges = (detected_edges > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)
    
    # Distance from every pixel to the nearest ground truth edge
    # We invert the ground truth because edt measures distance to non-zero pixels
    # If the pixel is already on an edge, distance is 0.
    dist_map = distance_transform_edt(ground_truth == 0)
    
    # Get total counts
    i_g = np.sum(ground_truth)
    i_d = np.sum(detected_edges)
    
    if i_g == 0 or i_d == 0: return 0
    
    # Calculate distances for each detected edge pixel
    d_i = dist_map[detected_edges > 0]
    
    # Pratt's Formula implementation
    fom = np.sum(1.0 / (1.0 + alpha * d_i**2))
    
    return fom / max(i_g, i_d)
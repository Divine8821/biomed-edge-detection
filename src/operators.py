import cv2
import numpy as np

def normalize_operator_output(img):
    """Utility to normalize operator results to [0, 1]."""
    if np.max(img) == np.min(img):
        return img
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def apply_sobel(img):
    """Standard Sobel operator (3x3)."""
    # cv2.Sobel includes built-in smoothing (weight of 2 in the center)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return normalize_operator_output(magnitude)

def apply_prewitt(img):
    """Prewitt operator (3x3)."""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    
    grad_x = cv2.filter2D(img, -1, kernel_x)
    grad_y = cv2.filter2D(img, -1, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return normalize_operator_output(magnitude)

def apply_canny(img, low_threshold=50, high_threshold=150):
    """
    Canny Edge Detector.
    Note: OpenCV's Canny requires 8-bit input. 
    We convert our [0,1] float to [0,255] uint8, process, and convert back.
    """
    # 1. Scale float [0,1] to uint8 [0,255]
    img_8bit = (img * 255).astype(np.uint8)
    
    # 2. Apply Canny 
    # Internally, this does: 1. Gaussian Blur, 2. Sobel Gradient, 
    # 3. Non-Maximum Suppression, 4. Hysteresis Thresholding
    edges = cv2.Canny(img_8bit, low_threshold, high_threshold)
    
    # 3. Normalize back to [0, 1]
    return normalize_operator_output(edges.astype(float))


def apply_roberts(img):
    """Roberts Cross operator (2x2)."""
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    
    grad_x = cv2.filter2D(img, -1, kernel_x)
    grad_y = cv2.filter2D(img, -1, kernel_y)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return normalize_operator_output(magnitude)

def apply_kirsch(img):
    """Kirsch Compass operator (detects maximum response in 8 directions)."""
    # 8-directional kernels
    kernels = [
        np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]]), # N
        np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]]), # NW
        np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]]), # W
        np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]]), # SW
        np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]]), # S
        np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]]), # SE
        np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]]), # E
        np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]])  # NE
    ]
    
    responses = [cv2.filter2D(img, cv2.CV_64F, k) for k in kernels]
    magnitude = np.max(responses, axis=0)
    return normalize_operator_output(magnitude)

def apply_frei_chen(img):
    """Isotropic Frei-Chen edge detection (using sqrt(2) weights)."""
    sqrt2 = np.sqrt(2)
    # The first two basis kernels for edge detection
    kernel_x = np.array([[-1, 0, 1], [-sqrt2, 0, sqrt2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -sqrt2, -1], [0, 0, 0], [1, sqrt2, 1]], dtype=np.float64)
    
    grad_x = cv2.filter2D(img, -1, kernel_x)
    grad_y = cv2.filter2D(img, -1, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return normalize_operator_output(magnitude)

def apply_log(img, sigma=1.5):
    """Laplacian of Gaussian (LoG) for noise-robust second-order detection."""
    # Step 1: Smooth with Gaussian
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    # Step 2: Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    # We return the absolute value to visualize the edges as positive spikes
    return normalize_operator_output(np.abs(laplacian))
# src/data_prep.py
import os
import cv2
import numpy as np

def normalize_image(img):
    """Min-Max normalization to [0, 1] range."""
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

def process_ultrasound(raw_dir, output_dir):
    """Converts 2D images (US/Xray) to normalized .npy slices."""
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(valid_exts)]
    for f in files:
        img = cv2.imread(os.path.join(raw_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            norm_img = normalize_image(img.astype(float))
            np.save(os.path.join(output_dir, f.split('.')[0] + ".npy"), norm_img)

if __name__ == "__main__":
    # Execution
    process_ultrasound('data/raw/Ultrasound/', 'data/processed/Ultrasound')
    
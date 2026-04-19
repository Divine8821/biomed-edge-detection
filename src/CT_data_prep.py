import os
import cv2
import numpy as np
import pydicom

def normalize_image(img):
    """Min-Max normalization to [0, 1] range."""
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

def process_ct_dicom(raw_dir, output_dir):
    """
    Processes CT DICOM files. 
    Handles Hounsfield Unit scaling if rescale slope/intercept are present.
    """
    os.makedirs(output_dir, exist_ok=True)
    dicom_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.dcm')]
    
    for f in dicom_files:
        ds = pydicom.dcmread(os.path.join(raw_dir, f))
        img = ds.pixel_array.astype(float)
        
        # CT specific: Rescale to Hounsfield Units (HU)
        # Most CT DICOMs have these attributes to convert raw sensor data to HU
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept
            
        # Standardize to [0, 1] for our operators
        norm_img = normalize_image(img)
        
        # Save as .npy
        np.save(os.path.join(output_dir, f.replace('.dcm', '.npy')), norm_img)

if __name__ == "__main__":
    # Execution
    process_ct_dicom('data/raw/CT/', 'data/processed/CT')
import pydicom
import cv2
import numpy as np
import os 

def load_processed_slice(file_path):
    """
    Loads a standardized .npy slice.
    Returns: numpy array (float64, normalized [0,1])
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No slice found at {file_path}")
    
    # .npy files preserve the exact float values from the NIfTI/DICOM source
    data = np.load(file_path)
    return data

def get_modality_files(processed_dir, modality):
    """
    Returns a list of all .npy files for a specific modality (mri, ct, ultrasound).
    """
    target_dir = os.path.join(processed_dir, modality)
    if not os.path.isdir(target_dir):
        return []
    
    return [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.npy')]
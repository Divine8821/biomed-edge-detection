import os
import nibabel as nib
import numpy as np
import random

def normalize_image(img):
    """Min-Max normalization to [0, 1] range."""
    denom = np.max(img) - np.min(img)
    if denom == 0: return img
    return (img - np.min(img)) / (denom + 1e-8)

def extract_mri_slices(raw_dir, output_dir, total_slices=50):
    os.makedirs(output_dir, exist_ok=True)
    
    # Robustly find files even with trailing dots
    nifti_files = [f for f in os.listdir(raw_dir) 
                   if f.rstrip('.').endswith('.nii') or f.rstrip('.').endswith('.nii.gz')]
    
    if not nifti_files:
        print(f"Error: No valid NIfTI files found in {raw_dir}")
        return

    slices_per_vol = total_slices // len(nifti_files)
    
    for vol_idx, filename in enumerate(nifti_files):
        path = os.path.join(raw_dir, filename)
        try:
            img = nib.load(path)
            data = img.get_fdata()
            
            # Select middle 60% to avoid empty space
            z_dim = data.shape[2]
            z_range = range(int(z_dim*0.2), int(z_dim*0.8))
            
            selected = random.sample(z_range, min(len(z_range), slices_per_vol))
            
            for idx in selected:
                slice_data = normalize_image(data[:, :, idx])
                save_name = f"mri_v{vol_idx}_s{idx}.npy"
                np.save(os.path.join(output_dir, save_name), slice_data)
                
            print(f"Successfully processed {filename}")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

if __name__ == "__main__":
    # Explicitly pass the paths here
    extract_mri_slices("data/raw/MRI", "data/processed/MRI", total_slices=50)
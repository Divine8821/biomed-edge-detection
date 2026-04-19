import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
from src.operators import (apply_sobel, apply_canny, apply_kirsch, 
                           apply_prewitt, apply_log, apply_roberts, apply_frei_chen)
from src.metrics import (calculate_psnr, pratt_fom, create_tissue_mask, 
                         binarize_edge_map, calculate_mse_roi)

def run_analysis():
    processed_base = 'data/processed/'
    modalities = ['mri', 'CT', 'Ultrasound']
    results_list = []

    for mode in modalities:
        mode_path = os.path.join(processed_base, mode)
        if not os.path.exists(mode_path):
            print(f"Skipping {mode}: path not found.")
            continue
        
        files = [f for f in os.listdir(mode_path) if f.endswith('.npy')]
        print(f"Processing {len(files)} slices for {mode}...")

        for f in files:
            img = np.load(os.path.join(mode_path, f))
            mask = create_tissue_mask(img)
            
            # 1. Ground Truth (Baseline)
            gt_edges = apply_canny(img)
            
            # 2. Test Operators
            raw_preds = {
                'Sobel': apply_sobel(img),
                'Prewitt': apply_prewitt(img),
                'Kirsch': apply_kirsch(img),
                'LoG': apply_log(img),
                'Roberts': apply_roberts(img),
                'Frei-Chen': apply_frei_chen(img)
            }
            
            # 3. Evaluation
            for name, pred_magnitude in raw_preds.items():
                # PSNR (calculated on grayscale gradient within ROI)
                psnr_val = calculate_psnr(img, pred_magnitude)
                
                # FOM (calculated on binary maps within ROI)
                pred_bin = binarize_edge_map(pred_magnitude) * mask
                gt_bin = gt_edges * mask
                fom_val = pratt_fom(pred_bin, gt_bin)
                
                results_list.append({
                    'Modality': mode,
                    'Operator': name,
                    'PSNR': psnr_val,
                    'FOM': fom_val
                })

    # Save and Print Results
    df = pd.DataFrame(results_list)
    df.to_csv('results/metrics_comparison.csv', index=False)
    summary = df.groupby(['Modality', 'Operator']).mean()
    print("\nFinal Results Summary:\n", summary)

if __name__ == "__main__":
    run_analysis()


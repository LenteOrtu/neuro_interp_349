#!/usr/bin/env python3
"""
Preprocess T1 ADNI data for MICCAI ViT-B MAE classification.

The T1 data is already:
  - MNI registered ("Spatially Normalized")
  - Skull-stripped ("Masked")
  - Bias-corrected ("N3 corrected")

This script applies:
  1. Reorient LAS → RAS (using nibabel affine)
  2. Z-score normalize over nonzero voxels
  3. Resize 110³ → 128³ (trilinear interpolation)
  4. Save as float32 .npy with channel dim (1, 128, 128, 128)

Usage:
    python preprocess_t1.py
    python preprocess_t1.py --split_path data/splits/t1_split.csv
"""

import argparse
import csv
import os
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def preprocess_volume(nifti_path, target_shape=(128, 128, 128)):
    """Load, reorient, normalize, resize a single T1 volume."""
    img = nib.load(nifti_path)

    # 1. Reorient to RAS
    ornt = nib.orientations.io_orientation(img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(ornt, ras_ornt)
    data = nib.orientations.apply_orientation(
        img.get_fdata(dtype=np.float32), transform
    )

    # 2. Z-score normalize over nonzero voxels
    nz = data > 0
    if nz.any():
        mu = data[nz].mean()
        sigma = data[nz].std()
        data = np.where(nz, (data - mu) / (sigma + 1e-8), 0)

    # 3. Resize to target shape
    current_shape = np.array(data.shape)
    target = np.array(target_shape)
    zoom_factors = target / current_shape
    data_resized = zoom(data, zoom_factors, order=1)  # trilinear

    # 4. Add channel dimension
    return data_resized[np.newaxis].astype(np.float32)  # (1, 128, 128, 128)


def main():
    parser = argparse.ArgumentParser(description="Preprocess T1 data for MICCAI ViT")
    parser.add_argument(
        "--split_path",
        default="data/splits/t1_split.csv",
        help="Path to split CSV with patient_id, split, label, nifti_path columns",
    )
    parser.add_argument(
        "--output_dir",
        default="data/T1_preprocessed_miccai",
        help="Output directory for preprocessed .npy files",
    )
    parser.add_argument("--force", action="store_true", help="Reprocess existing files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load split
    records = []
    with open(args.split_path) as f:
        for row in csv.DictReader(f):
            records.append(row)

    print(f"Total records: {len(records)}")

    start = time.time()
    succeeded = 0
    skipped = 0
    failed = 0

    for i, rec in enumerate(records):
        pid = rec["patient_id"]
        nifti_path = rec["nifti_path"]
        out_path = os.path.join(args.output_dir, f"{pid}.npy")

        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue

        try:
            volume = preprocess_volume(nifti_path)
            np.save(out_path, volume)
            succeeded += 1
        except Exception as e:
            print(f"  FAILED PID {pid}: {e}")
            failed += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  Progress: {i + 1}/{len(records)} ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s — succeeded: {succeeded}, skipped: {skipped}, failed: {failed}")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()

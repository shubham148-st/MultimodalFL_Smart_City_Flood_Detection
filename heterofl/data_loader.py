"""
Data loading and preprocessing module for the Multimodal Flood Detection system.

This module is responsible for loading heterogeneous data sources (satellite imagery patches,
rainfall time-series, and water-level sensor data), aligning them by timestamp, and 
structuring them into a unified PyTorch TensorDataset. It handles missing modalities
and generates a proper Train/Validation/Test split.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def get_dataset():
    """
    Load, preprocess, and align multimodal dataset.

    Returns:
        tuple: (train_ds, val_ds, test_ds)
            train_ds (torch.utils.data.Subset): Training dataset split (70%).
            val_ds (torch.utils.data.Subset): Validation dataset split (15%).
            test_ds (torch.utils.data.Subset): Test dataset split (15%).
            Returns None if required CSV files are not found.
    """
    print("dataset loading")

    # Rainfall CSV
    try:
        df_rain = pd.read_csv("dataset/tsmixer_input.csv")
        rain_cols = [f't{i}' for i in range(10)]
        rain_data = df_rain[rain_cols].values.astype(np.float32)
        X_rain_raw = torch.tensor(rain_data).unsqueeze(-1)
    except FileNotFoundError:
        print(" Error: tsmixer_input.csv not found.")
        return None

    # Flood / Water-Level CSV
    try:
        df_water = pd.read_csv("dataset/flood_dataset.csv")
        water_cols = [f'h{i}' for i in range(10)]
        water_data = df_water[water_cols].values.astype(np.float32)
        X_water_raw = torch.tensor(water_data).unsqueeze(-1)

        labels_raw = (df_water['water_level'].values > 5.0).astype(np.float32)
        Y_raw = torch.tensor(labels_raw).unsqueeze(-1)
    except FileNotFoundError:
        print(" Error: flood_dataset.csv not found.")
        return None

    # Satellite .npy patches
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npy_folder = os.path.join(base_dir, "dataset", "flood_dataset_npy")
    npy_path = os.path.join(npy_folder, "*.npy")

    npy_files = sorted(glob.glob(npy_path))
    print(f"\n[DEBUG] Found {len(npy_files)} total .npy files in folder.")

    # Only take exactly as many images as we have tabular rows
    target_samples = len(X_rain_raw)
    npy_files = npy_files[:target_samples]
    print(f"[DEBUG] Sliced dataset. Loading exactly {len(npy_files)} images to match CSV data...")

    patches = []
    PATCH_SIZE = 64

    for f in npy_files:
        try:
            data = np.load(f)
            if len(data.shape) == 3:
                if data.shape[2] == 5:
                    data = data[:, :, 0]
                elif data.shape[0] == 5:
                    data = data[0, :, :]

            if len(data.shape) == 2:
                h, w = data.shape
                cy, cx = h // 2, w // 2
                sy, sx = cy - PATCH_SIZE // 2, cx - PATCH_SIZE // 2
                if sy >= 0 and sx >= 0:
                    patch = data[sy:sy + PATCH_SIZE, sx:sx + PATCH_SIZE]
                    patches.append(patch)
        except Exception as e:
            print(f"[ERROR] Failed to load {os.path.basename(f)}: {e}")

    print(f"[DEBUG] Successfully extracted {len(patches)} valid 64x64 patches.\n")

    if len(patches) > 0:
        X_np = np.array(patches)
        if len(X_np.shape) == 3:
            X_np = X_np[:, np.newaxis, :, :]  # [N, 1, 64, 64]

        # Per-patch z-score normalization
        # Satellite patches have wildly different dynamic ranges.
        # Normalizing each patch independently prevents the CNN from
        # learning to respond to absolute brightness instead of structure.
        mean = X_np.mean(axis=(2, 3), keepdims=True)  # [N,1,1,1]
        std  = X_np.std(axis=(2, 3), keepdims=True).clip(min=1e-6)
        X_np = (X_np - mean) / std

        X_img_raw = torch.tensor(X_np, dtype=torch.float32)
    else:
        print("[WARNING] 0 patches found. Pipeline will use noise.")
        X_img_raw = torch.empty(0)

    # Align lengths
    len_rain  = len(X_rain_raw)
    len_water = len(X_water_raw)
    len_img   = len(X_img_raw) if len(X_img_raw) > 0 else 0
    max_len   = max(len_rain, len_water, len_img)

    # Print class balance
    flood_frac = Y_raw.mean().item() * 100
    print(f"[DEBUG] Class balance - Flood: {flood_frac:.1f}% | No-Flood: {100-flood_frac:.1f}%")

    def pad_tensor(t, target_len, dim_shape):
        """
        Pad or truncate a tensor to match a target temporal length.
        
        Args:
            t (torch.Tensor): Input tensor to pad or truncate.
            target_len (int): The desired length of the tensor.
            dim_shape (tuple): The shape of the individual elements (excluding batch/sequence dim).
            
        Returns:
            torch.Tensor: The length-aligned tensor.
        """
        current_len = len(t)
        if current_len >= target_len:
            return t[:target_len]
        needed = target_len - current_len
        padding = torch.zeros((needed, *dim_shape), dtype=torch.float32)
        if current_len == 0:
            return padding
        return torch.cat([t, padding], dim=0)

    X_rain  = pad_tensor(X_rain_raw,  max_len, (10, 1))
    X_water = pad_tensor(X_water_raw, max_len, (10, 1))
    Y       = pad_tensor(Y_raw,       max_len, (1,))

    if len_img > 0:
        X_img = pad_tensor(X_img_raw, max_len, (1, 64, 64))
    else:
        X_img = torch.randn(max_len, 1, 64, 64)

    dataset = TensorDataset(X_img, X_rain, X_water, Y)

    # 70/15/15 Train/Val/Test Split
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len   = int(0.15 * total_len)
    test_len  = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    print(f" Dataset Split - Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


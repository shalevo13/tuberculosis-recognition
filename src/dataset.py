"""
Dataset and DataLoader pipeline for Tuberculosis Chest X-Rays.
Implements stratified splitting and medical-safe data augmentations.
"""

import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_METADATA_CSV = os.path.join(PROJECT_ROOT, "data", "metadata.csv")
DEFAULT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images")

class ChestXRayDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Chest X-ray images and labels.
    """
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        
        # Load grayscale image
        image = Image.open(img_path).convert("L")
        label = torch.tensor(row["label"], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """
    Data augmentations tailored for Chest Radiographs.
    - Train: subtle rotation, scaling, brightness adjustments (no vertical flipping!)
    - Val/Test: deterministic normalization only.
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        # Normalize to [-1, 1] range: (x - 0.5) / 0.5
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, val_transform


def create_dataloaders(metadata_csv=None, images_dir=None, 
                       batch_size=16, val_size=0.15, test_size=0.15, seed=42):
    """
    Performs stratified train/val/test split and creates DataLoaders.
    """
    if metadata_csv is None:
        metadata_csv = DEFAULT_METADATA_CSV
    elif not os.path.isabs(metadata_csv) and not os.path.exists(metadata_csv):
        metadata_csv = os.path.join(PROJECT_ROOT, metadata_csv)

    if images_dir is None:
        images_dir = DEFAULT_IMAGES_DIR
    elif not os.path.isabs(images_dir) and not os.path.exists(images_dir):
        images_dir = os.path.join(PROJECT_ROOT, images_dir)

    df = pd.read_csv(metadata_csv)
    
    # 1. Stratified split: Separate Test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    
    # 2. Separate Validation set from remaining data
    val_rel_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_rel_size, random_state=seed, stratify=train_val_df["label"]
    )

    train_transform, val_transform = get_transforms()

    train_ds = ChestXRayDataset(train_df, images_dir, transform=train_transform)
    val_ds = ChestXRayDataset(val_df, images_dir, transform=val_transform)
    test_ds = ChestXRayDataset(test_df, images_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df
    }


if __name__ == "__main__":
    loaders = create_dataloaders()
    train_loader = loaders["train_loader"]
    print("DataLoader Split Summary:")
    print(f"  Train samples: {len(loaders['train_df'])} (TB: {loaders['train_df']['label'].sum()})")
    print(f"  Val samples:   {len(loaders['val_df'])} (TB: {loaders['val_df']['label'].sum()})")
    print(f"  Test samples:  {len(loaders['test_df'])} (TB: {loaders['test_df']['label'].sum()})")

    # Inspect a single batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch inspection:")
    print(f"  Images tensor shape: {images.shape} (Batch, Channel, Height, Width)")
    print(f"  Labels tensor shape: {labels.shape}")
    print(f"  Pixel values range:  [{images.min().item():.2f}, {images.max().item():.2f}]")

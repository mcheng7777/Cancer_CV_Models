"""Custom dataset for IDC cancer classification with subject-based splitting."""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class IDCDataset(Dataset):
    """Dataset for IDC cancer images with nested subject ID structure.
    
    Structure: {data_root}/{subject_id}/{0|1}/*.png
    """
    
    def __init__(
        self,
        data_root: str,
        subject_ids: List[str],
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize IDC Dataset.
        
        Args:
            data_root: Root directory containing subject ID folders
            subject_ids: List of subject IDs to include in this dataset
            transform: Optional transform to apply to images
        """
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids
        self.transform = transform
        
        # Collect all image paths and labels
        self.samples: List[Tuple[str, int]] = []
        
        for subject_id in subject_ids:
            subject_dir = self.data_root / subject_id
            
            # Load negative class (0)
            neg_dir = subject_dir / "0"
            if neg_dir.exists():
                for img_path in neg_dir.glob("*.png"):
                    self.samples.append((str(img_path), 0))
            
            # Load positive class (1)
            pos_dir = subject_dir / "1"
            if pos_dir.exists():
                for img_path in pos_dir.glob("*.png"):
                    self.samples.append((str(img_path), 1))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label by index.
        
        Returns:
            Tuple of (image tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(
    mode: str = "train",
    image_size: int = 224,
) -> transforms.Compose:
    """Get data transforms for training/validation/test.
    
    Args:
        mode: One of 'train', 'val', or 'test'
        image_size: Target image size (ResNet18 expects 224)
    
    Returns:
        Compose transform
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def split_subjects_by_id(
    data_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split subject IDs into train/val/test sets.
    
    This ensures no data leakage - all images from a subject stay together.
    
    Args:
        data_root: Root directory containing subject ID folders
        train_ratio: Proportion of subjects for training
        val_ratio: Proportion of subjects for validation
        test_ratio: Proportion of subjects for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_subject_ids, val_subject_ids, test_subject_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    data_root = Path(data_root)
    
    # Get all subject IDs
    subject_ids = [
        d.name for d in data_root.iterdir()
        if d.is_dir() and (d / "0").exists() or (d / "1").exists()
    ]
    subject_ids.sort()  # Sort for reproducibility
    
    # Shuffle with seed
    np.random.seed(random_seed)
    np.random.shuffle(subject_ids)
    
    # Calculate split indices
    n_total = len(subject_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train:n_train + n_val]
    test_ids = subject_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    image_size: int = 224,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with subject-based splitting.
    
    Args:
        data_root: Root directory containing subject ID folders
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        train_ratio: Proportion of subjects for training
        val_ratio: Proportion of subjects for validation
        test_ratio: Proportion of subjects for testing
        image_size: Target image size
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split subjects
    train_ids, val_ids, test_ids = split_subjects_by_id(
        data_root, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # Create datasets
    train_dataset = IDCDataset(
        data_root=data_root,
        subject_ids=train_ids,
        transform=get_transforms("train", image_size),
    )
    val_dataset = IDCDataset(
        data_root=data_root,
        subject_ids=val_ids,
        transform=get_transforms("val", image_size),
    )
    test_dataset = IDCDataset(
        data_root=data_root,
        subject_ids=test_ids,
        transform=get_transforms("test", image_size),
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


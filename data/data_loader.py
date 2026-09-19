"""
Data Loader for E-Commerce Recommendation System
=================================================

This module implements data loading and preprocessing for
e-commerce recommendation datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
import os


class EcommerceDataset(Dataset):
    """
    Dataset for e-commerce recommendation with multi-modal features.
    """

    def __init__(self, data: pd.DataFrame, config: Dict):
        self.data = data.reset_index(drop=True)
        self.config = config
        self.max_behavior_seq_len = config.get('max_behavior_seq_len', 50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        sample = {
            'user_id': row['user_id'],
            'item_id': row['item_id'],
            'label': torch.tensor(row['label'], dtype=torch.float32)
        }

        # User features
        if 'user_features' in row:
            sample['user_features'] = torch.tensor(
                row['user_features'], dtype=torch.float32
            )
        else:
            # Default user features
            sample['user_features'] = torch.randn(self.config.get('user_feature_dim', 64))

        # Item features
        if 'item_features' in row:
            sample['item_features'] = torch.tensor(
                row['item_features'], dtype=torch.float32
            )
        else:
            sample['item_features'] = torch.randn(self.config.get('item_feature_dim', 128))

        # Behavior sequence
        if 'behavior_seq' in row:
            behavior_seq = row['behavior_seq']
            if isinstance(behavior_seq, list):
                sample['behavior_seq'] = torch.tensor(behavior_seq, dtype=torch.long)
            else:
                sample['behavior_seq'] = torch.zeros(self.max_behavior_seq_len, dtype=torch.long)
        else:
            sample['behavior_seq'] = torch.zeros(self.max_behavior_seq_len, dtype=torch.long)

        # Behavior length
        if 'behavior_length' in row:
            sample['behavior_length'] = torch.tensor(
                row['behavior_length'], dtype=torch.long
            )
        else:
            sample['behavior_length'] = torch.tensor(0, dtype=torch.long)

        # Multi-modal features
        if 'item_image_features' in row and row['item_image_features'] is not None:
            sample['item_image_features'] = torch.tensor(
                row['item_image_features'], dtype=torch.float32
            )
        else:
            sample['item_image_features'] = torch.zeros(2048, dtype=torch.float32)

        if 'item_text_features' in row and row['item_text_features'] is not None:
            sample['item_text_features'] = torch.tensor(
                row['item_text_features'], dtype=torch.float32
            )
        else:
            sample['item_text_features'] = torch.zeros(768, dtype=torch.float32)

        # CTR/CVR labels for multi-task
        if 'ctr_label' in row:
            sample['ctr_label'] = torch.tensor(row['ctr_label'], dtype=torch.float32)
        if 'cvr_label' in row:
            sample['cvr_label'] = torch.tensor(row['cvr_label'], dtype=torch.float32)

        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.
    """
    collated = {}

    for key in batch[0].keys():
        if key in ['user_id', 'item_id']:
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated


def create_synthetic_data(
    num_users: int = 10000,
    num_items: int = 50000,
    num_interactions: int = 100000,
    max_behavior_seq_len: int = 50
) -> pd.DataFrame:
    """
    Create synthetic e-commerce interaction data.

    Args:
        num_users: Number of users
        num_items: Number of items
        num_interactions: Number of interactions to generate
        max_behavior_seq_len: Maximum behavior sequence length

    Returns:
        DataFrame with interaction data
    """
    print(f"Generating synthetic data: {num_users} users, {num_items} items, {num_interactions} interactions...")

    data = []

    for i in range(num_interactions):
        user_id = random.randint(0, num_users - 1)
        item_id = random.randint(0, num_items - 1)

        # Generate interaction features
        user_features = np.random.randn(64).astype(np.float32)
        item_features = np.random.randn(128).astype(np.float32)

        # Simulate behavior sequence
        behavior_len = min(random.randint(0, max_behavior_seq_len), max_behavior_seq_len)
        behavior_seq = [random.randint(0, num_items - 1) for _ in range(behavior_len)]
        # Pad sequence
        behavior_seq = behavior_seq + [0] * (max_behavior_seq_len - behavior_len)

        # Multi-modal features (random for synthetic data)
        item_image_features = np.random.randn(2048).astype(np.float32)
        item_text_features = np.random.randn(768).astype(np.float32)

        # Labels (with some realistic patterns)
        # Higher behavior length -> higher likelihood of click
        ctr_prob = 0.1 + 0.01 * min(behavior_len, 10)
        cvr_prob = 0.05 + 0.005 * min(behavior_len, 10)

        ctr_label = 1 if random.random() < ctr_prob else 0
        cvr_label = 1 if ctr_label == 1 and random.random() < cvr_prob else 0
        label = ctr_label  # For single-task models

        data.append({
            'user_id': user_id,
            'item_id': item_id,
            'user_features': user_features,
            'item_features': item_features,
            'behavior_seq': behavior_seq,
            'behavior_length': behavior_len,
            'item_image_features': item_image_features,
            'item_text_features': item_text_features,
            'ctr_label': ctr_label,
            'cvr_label': cvr_label,
            'label': label
        })

        if (i + 1) % 20000 == 0:
            print(f"  Generated {i + 1}/{num_interactions} interactions")

    df = pd.DataFrame(data)
    print("Data generation complete!")
    return df


def load_amazon_electronics_data(data_dir: str = 'data/amazon') -> pd.DataFrame:
    """
    Load Amazon Electronics dataset (simulated format).

    For real data, download from:
    https://jmcauley.ucsd.edu/data/amazon_v2/

    Returns:
        DataFrame with interaction data
    """
    # This is a placeholder - in practice, load real Amazon data
    print("Loading Amazon Electronics data (simulated)...")

    # For demonstration, create synthetic data that mimics Amazon format
    return create_synthetic_data(
        num_users=5000,
        num_items=10000,
        num_interactions=50000
    )


def load_yoochoose_data(data_dir: str = 'data/yoochoose') -> pd.DataFrame:
    """
    Load YooChoose session-based recommendation dataset.

    For real data, download from:
    https://www.recodatasets.com/recsys/

    Returns:
        DataFrame with interaction data
    """
    print("Loading YooChoose data (simulated)...")

    # For demonstration, create synthetic session data
    return create_synthetic_data(
        num_users=3000,
        num_items=8000,
        num_interactions=30000,
        max_behavior_seq_len=30
    )


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets.

    Args:
        df: Input DataFrame
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and evaluation.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = EcommerceDataset(train_df, config)
    val_dataset = EcommerceDataset(val_df, config)
    test_dataset = EcommerceDataset(test_df, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 256),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 256),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 256),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


class DataAugmenter:
    """
    Data augmentation for recommendation models.
    """

    @staticmethod
    def add_noise(features: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to features."""
        noise = np.random.randn(*features.shape) * noise_level
        return features + noise

    @staticmethod
    def mask_features(features: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:
        """Randomly mask features."""
        mask = np.random.random(features.shape) > mask_ratio
        return features * mask

    @staticmethod
    def shuffle_sequence(sequence: List, shuffle_ratio: float = 0.2) -> List:
        """Randomly shuffle a portion of sequence elements."""
        n = len(sequence)
        shuffle_count = int(n * shuffle_ratio)
        indices = random.sample(range(n), shuffle_count)

        shuffled = sequence.copy()
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            shuffled[idx] = sequence[indices[(i + 1) % len(indices)]]

        return shuffled


if __name__ == '__main__':
    print("Data Loader Test")
    print("=" * 50)

    # Create synthetic data
    df = create_synthetic_data(
        num_users=1000,
        num_items=5000,
        num_interactions=5000
    )

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Split data
    train_df, val_df, test_df = split_data(df)

    # Create dataloaders
    config = {
        'batch_size': 64,
        'max_behavior_seq_len': 50,
        'user_feature_dim': 64,
        'item_feature_dim': 128
    }

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"User features shape: {batch['user_features'].shape}")
    print(f"Item features shape: {batch['item_features'].shape}")
    print(f"Behavior seq shape: {batch['behavior_seq'].shape}")
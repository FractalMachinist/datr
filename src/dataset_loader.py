"""
OkCupid Dataset Loader

Creates (trait_vector, essay_text) pairs for training.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle

from trait_vectorizer import TraitVectorizer


class OkCupidDataset(Dataset):
    """Dataset of (trait_vector, essay_text) pairs."""
    
    ESSAY_PROMPTS = {
        'essay0': 'My self summary',
        'essay1': 'What I\'m doing with my life',
        'essay2': 'I\'m really good at',
        'essay3': 'The first thing people usually notice about me',
        'essay4': 'Favorite books, movies, show, music, and food',
        'essay5': 'The six things I could never do without',
        'essay6': 'I spend a lot of time thinking about',
        'essay7': 'On a typical Friday night I am',
        'essay8': 'The most private thing I am willing to admit',
        'essay9': 'You should message me if...',
    }
    
    def __init__(self, csv_path: str, vectorizer: TraitVectorizer, 
                 max_essays: int = None, seed: int = 42):
        """
        Load OkCupid dataset.
        
        Args:
            csv_path: Path to okcupid_profiles.csv
            vectorizer: TraitVectorizer instance
            max_essays: Max total essays to include (for debugging)
            seed: Random seed
        """
        np.random.seed(seed)
        self.vectorizer = vectorizer
        
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Build (traits_vector, essay_text) pairs
        self.data = []
        essay_cols = [col for col in df.columns if col.startswith('essay')]
        
        essay_count = 0
        for idx, row in df.iterrows():
            profile_dict = row.to_dict()
            traits_vector = vectorizer.vectorize(profile_dict)
            
            # Add one entry per essay this profile has
            for essay_col in essay_cols:
                essay_text = profile_dict.get(essay_col, '')
                
                # Skip empty essays
                if pd.isna(essay_text) or (isinstance(essay_text, str) and len(essay_text.strip()) == 0):
                    continue
                
                # Create the training text: "Essay prompt\n{essay text}"
                prompt = self.ESSAY_PROMPTS.get(essay_col, essay_col)
                text = f"{prompt}\n{essay_text}"
                
                self.data.append((traits_vector, text))
                essay_count += 1
                
                if max_essays and essay_count >= max_essays:
                    break
            
            if max_essays and essay_count >= max_essays:
                break
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1} profiles, {essay_count} essays so far...")
        
        print(f"\nDataset loaded: {len(self.data)} (trait_vector, essay_text) pairs")
        print(f"  From {(idx+1)} profiles")
        
        # Compute text statistics
        text_lengths = [len(text) for _, text in self.data]
        print(f"  Text lengths: min={min(text_lengths)}, max={max(text_lengths)}, "
              f"mean={np.mean(text_lengths):.0f}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """Return (traits_vector, essay_text)."""
        return self.data[idx]
    
    def save(self, path: str):
        """Save dataset to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Dataset saved to {path}")
    
    @staticmethod
    def load(path: str) -> List[Tuple[np.ndarray, str]]:
        """Load dataset from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} examples from {path}")
        return data


def create_dataloaders(dataset: OkCupidDataset, 
                       train_frac: float = 0.9,
                       batch_size: int = 32,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders.
    
    Note: We need custom collate_fn since we have variable-length text.
    """
    n = len(dataset)
    train_size = int(n * train_frac)
    
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    def collate_fn(batch):
        """Collate variable-length sequences."""
        traits_list = []
        texts_list = []
        for traits, text in batch:
            traits_list.append(torch.from_numpy(traits))
            texts_list.append(text)
        
        traits_batch = torch.stack(traits_list)  # (B, 86)
        return traits_batch, texts_list
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_dataset)} examples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} examples, {len(val_loader)} batches")
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset loading
    df = pd.read_csv('/home/zach/Documents/Proj/datr/downloads/okcupid_profiles.csv')
    vectorizer = TraitVectorizer(df)
    
    dataset = OkCupidDataset(
        '/home/zach/Documents/Proj/datr/downloads/okcupid_profiles.csv',
        vectorizer,
        max_essays=1000  # For testing
    )
    
    # Sample a few examples
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES")
    print("="*80)
    for i in range(3):
        traits, text = dataset[i]
        print(f"\nExample {i}:")
        print(f"  Traits vector: shape={traits.shape}, min={traits.min():.3f}, "
              f"max={traits.max():.3f}")
        print(f"  Text ({len(text)} chars): {text[:100]}...")
    
    # Test dataloaders
    print("\n" + "="*80)
    print("TEST DATALOADERS")
    print("="*80)
    train_loader, val_loader = create_dataloaders(dataset, batch_size=16)
    
    batch = next(iter(train_loader))
    traits_batch, text_batch = batch
    print(f"\nBatch shapes:")
    print(f"  Traits: {traits_batch.shape}")
    print(f"  Texts: {len(text_batch)} items")
    print(f"\nSample batch text[0]: {text_batch[0][:100]}...")

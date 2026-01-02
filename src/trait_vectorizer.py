"""
Trait Vector Encoding Scheme

Maps profile traits to vectors according to the architecture decisions in README:
- Continuous values (height, income) -> scalars
- Categorical values (education, ethnicity) -> one-hot vectors
- Ordered categorical (drugs, drinks) -> scalar 0-1 ranges
- Multi-categorical (pets, offspring, ethnicity) -> split to individual binary features

Missing values are handled by:
- Continuous: random draw from dataset distribution
- Categorical: zero-hot vector
- Ordered categorical: random draw from dataset distribution
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any


class TraitVectorizer:
    """Converts profile traits to fixed-size numeric vectors."""
    
    # Define traits and their types
    CONTINUOUS_TRAITS = ['age', 'height', 'income']
    
    # Ordered categorical: never (0) -> sometimes (0.5) -> often (1.0)
    ORDERED_TRAITS = {
        'drinks': ['not at all', 'rarely', 'socially', 'often', 'very often', 'desperately'],
        'drugs': ['never', 'sometimes', 'often'],
        'smokes': ['no', 'trying to quit', 'when drinking', 'sometimes', 'yes'],
    }
    
    # Regular categorical traits (one-hot encoded)
    CATEGORICAL_TRAITS = {
        'sex': ['m', 'f'],
        'orientation': ['straight', 'gay', 'bisexual'],
        'status': ['single', 'available', 'seeing someone', 'married', 'unknown'],
        'body_type': ['skinny', 'thin', 'average', 'fit', 'athletic', 'curvy', 'full figured', 
                      'overweight', 'jacked', 'a little extra', 'used up', 'rather not say'],
        'diet': ['strictly anything', 'mostly anything', 'anything', 'mostly other', 'strictly other',
                 'vegetarian', 'mostly vegetarian', 'vegan', 'mostly vegan', 'strictly vegetarian'],
        'education': ['high school', 'two-year college', 'college/university', 'masters program', 
                      'doctorate program', 'space camp', 'working on high school', 'working on two-year college',
                      'working on college/university', 'working on masters program', 'working on doctorate program',
                      'working on space camp'],
        'job': ['student', 'transportation', 'hospitality / travel', 'sales / marketing / biz dev',
                'banking / financial / real estate', 'entertainment / media', 'medicine / health',
                'artistic / musical / writer', 'computer / hardware / software', 'other'],
        'religion': ['atheism', 'agnosticism', 'christianity', 'catholicism', 'islam', 'judaism',
                     'buddhism', 'hinduism', 'other'],
        'sign': ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpio',
                 'sagittarius', 'capricorn', 'aquarius', 'pisces'],
    }
    
    # Multi-categorical traits: split into binary features per value
    MULTI_CATEGORICAL_TRAITS = {
        'offspring': {
            'has_kids': ["has kids", "has a kid"],
            'wants_kids': ["has kids", "might want", "wants kids"],
            'has_no_kids': ["doesn't have kids"],
        },
        'pets': {
            'likes_dogs': ['likes dogs', 'has dogs'],
            'likes_cats': ['likes cats', 'has cats'],
        },
        # Note: ethnicity will be handled specially as it contains multiple values per entry
    }
    
    # We'll skip location and speaks (too high cardinality)
    SKIP_TRAITS = ['location', 'speaks', 'last_online']
    
    def __init__(self, df: pd.DataFrame):
        """Initialize vectorizer by computing statistics from training data."""
        self.df = df
        self.stats = {}
        self.trait_names = []
        self.vector_size = 0
        
        # Compute stats for missing value imputation
        self._compute_stats()
        self._build_trait_layout()
    
    def _compute_stats(self):
        """Compute statistics for handling missing values."""
        # Continuous trait stats
        for trait in self.CONTINUOUS_TRAITS:
            valid_data = self.df[trait].dropna()
            self.stats[trait] = {
                'mean': valid_data.mean(),
                'std': valid_data.std(),
                'min': valid_data.min(),
                'max': valid_data.max(),
            }
        
        # Ordered trait value distributions
        for trait, values in self.ORDERED_TRAITS.items():
            counts = self.df[trait].value_counts()
            total = counts.sum()
            self.stats[trait] = {v: counts.get(v, 0) / total for v in values}
    
    def _build_trait_layout(self):
        """Build the mapping of trait names to vector indices."""
        idx = 0
        
        # Continuous traits (1 scalar each)
        for trait in self.CONTINUOUS_TRAITS:
            self.trait_names.append((trait, 'continuous', (idx, idx + 1)))
            idx += 1
        
        # Ordered categorical traits (1 scalar each, normalized 0-1)
        for trait in self.ORDERED_TRAITS.keys():
            self.trait_names.append((trait, 'ordered', (idx, idx + 1)))
            idx += 1
        
        # Categorical traits (one-hot)
        for trait, values in self.CATEGORICAL_TRAITS.items():
            n_values = len(values)
            self.trait_names.append((trait, 'categorical', (idx, idx + n_values)))
            idx += n_values
        
        # Multi-categorical traits (binary features)
        for trait, subtrait_dict in self.MULTI_CATEGORICAL_TRAITS.items():
            for subtrait_name in subtrait_dict.keys():
                full_name = f"{trait}_{subtrait_name}"
                self.trait_names.append((full_name, 'multi_categorical', (idx, idx + 1)))
                idx += 1
        
        self.vector_size = idx
    
    def _impute_continuous(self, value: Any, trait: str) -> float:
        """Impute missing continuous value."""
        if pd.notna(value):
            return float(value)
        # Draw from normal distribution based on dataset stats
        stats = self.stats[trait]
        return np.random.normal(stats['mean'], stats['std'])
    
    def _impute_ordered(self, value: Any, trait: str) -> float:
        """Impute missing ordered categorical value."""
        if pd.notna(value):
            values = self.ORDERED_TRAITS[trait]
            return values.index(value) / (len(values) - 1)
        # Random draw from distribution
        values = self.ORDERED_TRAITS[trait]
        probs = [self.stats[trait].get(v, 1e-6) for v in values]
        probs = np.array(probs) / sum(probs)
        idx = np.random.choice(len(values), p=probs)
        return idx / (len(values) - 1)
    
    def _encode_categorical(self, value: Any, trait: str) -> np.ndarray:
        """Encode categorical trait as one-hot (or zero-hot if missing)."""
        values = self.CATEGORICAL_TRAITS[trait]
        one_hot = np.zeros(len(values), dtype=np.float32)
        
        if pd.notna(value) and value in values:
            idx = values.index(value)
            one_hot[idx] = 1.0
        # If missing or not in values, keep zero-hot
        
        return one_hot
    
    def _encode_multi_categorical(self, row: pd.Series, trait: str, subtrait: str) -> float:
        """Encode multi-categorical trait as binary feature."""
        value = row.get(trait, np.nan) if isinstance(row, dict) else row.get(trait)
        if pd.isna(value):
            return 0.0
        
        value_str = str(value).lower()
        patterns = self.MULTI_CATEGORICAL_TRAITS[trait][subtrait]
        
        for pattern in patterns:
            if pattern.lower() in value_str:
                return 1.0
        return 0.0
    
    def vectorize(self, profile: Dict[str, Any]) -> np.ndarray:
        """
        Convert a profile dict to a trait vector.
        
        Args:
            profile: Dict with trait names as keys (can be partial)
            
        Returns:
            numpy array of shape (vector_size,)
        """
        vector = np.zeros(self.vector_size, dtype=np.float32)
        
        # Continuous traits
        for trait in self.CONTINUOUS_TRAITS:
            value = profile.get(trait, np.nan)
            normalized = self._impute_continuous(value, trait)
            # Normalize to roughly 0-1 range using dataset stats
            stats = self.stats[trait]
            normalized = (normalized - stats['min']) / (stats['max'] - stats['min'] + 1e-6)
            normalized = np.clip(normalized, 0, 1)
            
            idx_start, idx_end = self._get_trait_indices(trait)
            vector[idx_start:idx_end] = normalized
        
        # Ordered categorical
        for trait in self.ORDERED_TRAITS.keys():
            value = profile.get(trait, np.nan)
            normalized = self._impute_ordered(value, trait)
            idx_start, idx_end = self._get_trait_indices(trait)
            vector[idx_start:idx_end] = normalized
        
        # Categorical (one-hot)
        for trait in self.CATEGORICAL_TRAITS.keys():
            value = profile.get(trait, np.nan)
            one_hot = self._encode_categorical(value, trait)
            idx_start, idx_end = self._get_trait_indices(trait)
            vector[idx_start:idx_end] = one_hot
        
        # Multi-categorical
        for trait, subtrait_dict in self.MULTI_CATEGORICAL_TRAITS.items():
            for subtrait_name in subtrait_dict.keys():
                value = self._encode_multi_categorical(profile, trait, subtrait_name)
                full_name = f"{trait}_{subtrait_name}"
                idx_start, idx_end = self._get_trait_indices(full_name)
                vector[idx_start:idx_end] = value
        
        return vector
    
    def _get_trait_indices(self, trait_name: str) -> Tuple[int, int]:
        """Get the vector indices for a trait."""
        for name, typ, (start, end) in self.trait_names:
            if name == trait_name:
                return start, end
        raise ValueError(f"Unknown trait: {trait_name}")
    
    def get_vector_info(self) -> Dict[str, Any]:
        """Return info about the vector structure."""
        return {
            'vector_size': self.vector_size,
            'traits': [(name, typ) for name, typ, _ in self.trait_names],
        }


if __name__ == '__main__':
    # Test with actual data
    df = pd.read_csv('/home/zach/Documents/Proj/datr/downloads/okcupid_profiles.csv')
    
    vectorizer = TraitVectorizer(df)
    print(f"Vector size: {vectorizer.vector_size}")
    print(f"\nTrait layout ({len(vectorizer.trait_names)} traits):")
    for name, typ, (start, end) in vectorizer.trait_names:
        print(f"  {name:30s} ({typ:18s}) [{start:3d}:{end:3d}]")
    
    # Test vectorization with a real profile
    print("\n" + "="*80)
    print("TEST: Vectorize profile 0")
    print("="*80)
    profile_dict = df.iloc[0].to_dict()
    print("\nProfile traits:")
    for k, v in profile_dict.items():
        if not k.startswith('essay'):
            print(f"  {k}: {v}")
    
    vector = vectorizer.vectorize(profile_dict)
    print(f"\nVector shape: {vector.shape}")
    print(f"Vector values (first 20): {vector[:20]}")
    print(f"Vector stats: min={vector.min():.3f}, max={vector.max():.3f}, mean={vector.mean():.3f}")
    
    # Test with missing values
    print("\n" + "="*80)
    print("TEST: Vectorize incomplete profile")
    print("="*80)
    partial_profile = {
        'age': 28,
        'sex': 'f',
        'height': 66.0,
        'orientation': 'straight',
    }
    print(f"Partial profile: {partial_profile}")
    vector2 = vectorizer.vectorize(partial_profile)
    print(f"Vector shape: {vector2.shape}")
    print(f"Vector values (first 20): {vector2[:20]}")

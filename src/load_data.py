"""Data loading and stratified splitting for IMDB sentiment analysis."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(csv_path):
    """
    Load IMDB dataset and create stratified train/validation/test splits.
    
    Split ratio: 60% train, 20% validation, 20% test (stratified by sentiment)
    Stratification ensures balanced class distribution across all splits.
    
    Args:
        csv_path: Path to IMDB Dataset CSV file with 'review' and 'sentiment' columns
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        - X: List of review texts (strings)
        - y: List of labels (1 = positive, 0 = negative)
    
    Example:
        >>> X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('data/IMDB Dataset.csv')
        >>> print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    """
    # Load dataset from CSV
    df = pd.read_csv(csv_path)
    
    # Extract texts and convert sentiments to binary labels
    X = df["review"].tolist()
    y = [1 if s.lower() == "positive" else 0 for s in df["sentiment"]]
    
    # Display dataset statistics
    print(f"Total samples: {len(y)}")
    print(f"Positive samples: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"Negative samples: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    # First split: 60% train, 40% temp (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    # Second split: Split temp into 50/50 for validation and test (20% each of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Display split statistics
    print(f"Train samples: {len(y_train)} (60%)")
    print(f"Validation samples: {len(y_val)} (20%)")
    print(f"Test samples: {len(y_test)} (20%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

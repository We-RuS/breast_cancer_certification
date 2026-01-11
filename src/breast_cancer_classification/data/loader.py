"""Data loading module."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from typing import Tuple, Dict, Any

def load_raw_data() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load raw breast cancer dataset from sklearn.

    :return: tuple
        (x, y, metadata)
    """
    data  = load_breast_cancer()
    X = data.data
    y = data.target

    metadata = {
        'feature_names': data.feature_names,
        'target_names': data.target_names,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
    }
    return X, y, metadata

def get_class_distribution(y: np.ndarray) -> Dict[str, Any]:
    """
    Calculate class distribution.
    :param y: np.ndarray
        Target labels
        
    :return:dict
        Class distribution statistics
    """
    unique, counts = np.unique(y, return_counts=True)
    
    return {
        "classes": unique.tolist(),
        "counts": counts.tolist(),
        "proportions": (counts / len(y)).tolist(),
        "is_balanced": np.all(np.abs(counts / len(y) - 0.5) < 0.1)
    }
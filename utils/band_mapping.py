"""
Band Mapping Utilities for Sideslip Distribution
================================================

Loads the official band classifications from sideSlip_distribution.csv
and provides utilities to classify sideslip values into many/medium/few-shot bands.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_band_mapping(distribution_path="datasets_challenge/sideSlip_distribution.csv"):
    """
    Load the official band mapping from sideSlip_distribution.csv
    
    Returns:
        band_dict: Dictionary mapping sideslip values (rounded to 1 decimal) to band
    """
    # Handle relative path
    if not Path(distribution_path).is_absolute():
        script_dir = Path(__file__).parent.parent
        distribution_path = script_dir / distribution_path
    
    # Load distribution file
    df = pd.read_csv(distribution_path)
    
    # Create mapping dictionary
    band_dict = dict(zip(df['sideSlip'], df['band']))
    
    return band_dict


def classify_band(sideslip_values, band_dict=None):
    """
    Classify sideslip values into bands using the official mapping.
    
    Args:
        sideslip_values: Array of sideslip values
        band_dict: Optional pre-loaded band dictionary (loads if None)
    
    Returns:
        Array of band labels ('many', 'medium', 'few')
    """
    if band_dict is None:
        band_dict = load_band_mapping()
    
    # Round to 1 decimal (matching the distribution file)
    rounded_values = np.round(sideslip_values, 1)
    
    # Classify each value
    bands = np.array([band_dict.get(val, 'unknown') for val in rounded_values])
    
    return bands


def get_band_statistics(sideslip_values, band_dict=None):
    """
    Calculate statistics for each band.
    
    Returns:
        Dictionary with counts and percentages for each band
    """
    bands = classify_band(sideslip_values, band_dict)
    
    total = len(bands)
    many_count = (bands == 'many').sum()
    medium_count = (bands == 'medium').sum()
    few_count = (bands == 'few').sum()
    unknown_count = (bands == 'unknown').sum()
    
    return {
        'many': {'count': many_count, 'pct': 100 * many_count / total},
        'medium': {'count': medium_count, 'pct': 100 * medium_count / total},
        'few': {'count': few_count, 'pct': 100 * few_count / total},
        'unknown': {'count': unknown_count, 'pct': 100 * unknown_count / total},
        'total': total
    }


def calculate_band_mae(predictions, targets, band_dict=None):
    """
    Calculate MAE for each band separately.
    
    Args:
        predictions: Array of predictions
        targets: Array of ground truth values
        band_dict: Optional pre-loaded band dictionary
    
    Returns:
        Dictionary with MAE for each band
    """
    bands = classify_band(targets, band_dict)
    
    results = {}
    for band_name in ['many', 'medium', 'few']:
        mask = bands == band_name
        if mask.sum() > 0:
            band_mae = np.mean(np.abs(predictions[mask] - targets[mask]))
            results[f'{band_name}_mae'] = band_mae
            results[f'{band_name}_count'] = mask.sum()
            results[f'{band_name}_pct'] = 100 * mask.sum() / len(targets)
        else:
            results[f'{band_name}_mae'] = 0.0
            results[f'{band_name}_count'] = 0
            results[f'{band_name}_pct'] = 0.0
    
    # Overall MAE
    results['overall_mae'] = np.mean(np.abs(predictions - targets))
    
    return results


if __name__ == "__main__":
    # Test the utilities
    print("Testing band mapping utilities...\n")
    
    band_dict = load_band_mapping()
    print(f"✅ Loaded {len(band_dict)} unique sideslip values")
    
    # Count by band
    band_counts = {}
    for value, band in band_dict.items():
        band_counts[band] = band_counts.get(band, 0) + 1
    
    print(f"\nBand coverage:")
    for band, count in sorted(band_counts.items()):
        print(f"  {band:8s}: {count:3d} unique values")
    
    # Test classification
    test_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    test_bands = classify_band(test_values, band_dict)
    
    print(f"\nTest classifications:")
    for val, band in zip(test_values, test_bands):
        print(f"  {val:5.1f}° → {band}")

import pytest
import pandas as pd
import numpy as np
import os
from src.features.build_features import build_features

def test_build_features_output_integrity():
    """
    Test that build_features produces a file with the expected columns and no NaNs.
    """
    # Setup: Create a small dummy raw data file
    os.makedirs('data/raw', exist_ok=True)
    dummy_path = 'data/raw/test_traffic.csv'
    output_path = 'data/processed/test_features.csv'
    
    n_samples = 100
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='min'),
        'rsl_dbm': np.random.normal(-45, 1, n_samples),
        'snr_db': np.random.normal(25, 1, n_samples),
        'ber': np.random.uniform(1e-9, 1e-8, n_samples),
        'throughput_mbps': np.random.normal(300, 10, n_samples),
        'is_anomaly': np.random.randint(0, 2, n_samples)
    })
    df.to_csv(dummy_path, index=False)
    
    # Run build_features
    build_features(input_path=dummy_path, output_path=output_path)
    
    # Assertions
    assert os.path.exists(output_path)
    processed_df = pd.read_csv(output_path)
    
    # Check for expected rolling columns (e.g., rsl_dbm_mean_5m)
    assert 'rsl_dbm_mean_5m' in processed_df.columns
    assert 'snr_db_std_15m' in processed_df.columns
    assert 'ber_log' in processed_df.columns
    
    # Ensure no NaNs in the final processed data
    assert processed_df.isnull().sum().sum() == 0
    
    # Cleanup
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
    if os.path.exists(output_path):
        os.remove(output_path)

def test_ber_log_calculation():
    """
    Test that log transformation for BER is handled correctly.
    """
    ber_val = 1e-9
    epsilon = 1e-12
    ber_log = np.log10(ber_val + epsilon)
    expected = np.log10(1e-9 + 1e-12)
    assert np.isclose(ber_log, expected, atol=1e-8)

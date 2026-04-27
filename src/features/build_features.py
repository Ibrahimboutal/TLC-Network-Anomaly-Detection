import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def build_features(df=None, input_path=None, output_path=None, scaler=None):
    """
    Builds features from raw network traffic data.
    Can accept a DataFrame or a file path.
    If scaler is provided, uses it to transform. Otherwise fits a new one.
    """
    # Load data
    if df is None:
        if input_path is None:
             input_path = 'data/raw/network_traffic.csv'
        df = pd.read_csv(input_path)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # 1. Rolling Statistics (Capture Trends)
    windows = [5, 15, 60] # minutes
    features_to_roll = ['rsl_dbm', 'snr_db', 'throughput_mbps']
    
    for window in windows:
        for col in features_to_roll:
            if col in df.columns:
                # Rolling Mean
                df[f'{col}_mean_{window}m'] = df[col].rolling(window=window).mean()
                # Rolling Std (Volatility)
                df[f'{col}_std_{window}m'] = df[col].rolling(window=window).std()
                # Deviation from mean (Z-score like)
                df[f'{col}_dev_{window}m'] = df[col] - df[f'{col}_mean_{window}m']
            
    # 2. Lagged Features (Capture sudden changes)
    lags = [1, 5]
    for lag in lags:
        for col in features_to_roll:
            if col in df.columns:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                df[f'{col}_diff_{lag}'] = df[col] - df[f'{col}_lag_{lag}']
            
    # 3. Log transform BER (since it spans multiple orders of magnitude)
    if 'ber' in df.columns:
        df['ber_log'] = np.log10(df['ber'] + 1e-12)
    
    # Drop rows with NaNs from rolling/lagging
    df = df.dropna()
    
    # Define feature set
    # Exclude timestamp and target labels
    exclude = ['timestamp', 'is_anomaly', 'ber']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # 4. Scaling
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        # Align columns with scaler's expected feature names
        if hasattr(scaler, "feature_names_in_"):
            expected_features = scaler.feature_names_in_
            # Check for missing features and fill with 0
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0.0
            # Transform only the expected features in the correct order
            df[expected_features] = scaler.transform(df[expected_features])
        else:
            # Fallback if no feature names (shouldn't happen with DataFrames)
            df[feature_cols] = scaler.transform(df[feature_cols])
    
    # Save processed data if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    
    return df, scaler

if __name__ == "__main__":
    build_features(output_path='data/processed/features.csv')

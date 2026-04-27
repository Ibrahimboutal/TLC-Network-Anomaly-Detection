import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def build_features(input_path='data/raw/network_traffic.csv', output_path='data/processed/features.csv'):
    # Load data
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # 1. Rolling Statistics (Capture Trends)
    windows = [5, 15, 60] # minutes
    features_to_roll = ['rsl_dbm', 'snr_db', 'throughput_mbps']
    
    for window in windows:
        for col in features_to_roll:
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
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            df[f'{col}_diff_{lag}'] = df[col] - df[f'{col}_lag_{lag}']
            
    # 3. Log transform BER (since it spans multiple orders of magnitude)
    df['ber_log'] = np.log10(df['ber'] + 1e-12)
    
    # Drop rows with NaNs from rolling/lagging
    df = df.dropna()
    
    # Define feature set
    # Exclude timestamp and target labels
    exclude = ['timestamp', 'is_anomaly', 'ber']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # 4. Scaling
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete. Shape: {df_scaled.shape}")
    print(f"Total features: {len(feature_cols)}")
    return df_scaled

if __name__ == "__main__":
    build_features()

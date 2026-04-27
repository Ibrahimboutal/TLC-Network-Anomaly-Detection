import pandas as pd
import numpy as np
import os
from datetime import datetime

def ingest_4g_data(path):
    """
    Kaggle: Anomaly Detection in 4G Cellular Networks
    """
    try:
        df = pd.read_csv(path, sep=';')
        # Time is HH:MM. Add a dummy date to make it a proper timestamp
        df['timestamp'] = pd.to_datetime('2024-01-01 ' + df['Time'])
        
        # Mapping
        df = df.rename(columns={
            'meanThr_DL': 'throughput_mbps'
        })
        
        # Synthetic placeholders for missing features (for model compatibility)
        # In a real scenario, we'd find datasets with these or retrain.
        # Here we use 'normal' defaults.
        df['rsl_dbm'] = -45.0
        df['snr_db'] = 25.0
        df['ber'] = 1e-9
        
        # This dataset doesn't have labels in the test file, so we'll 
        # use it for unsupervised validation or assume it's all normal for now
        # unless we find labels.
        df['is_anomaly'] = 0 
        
        return df[['timestamp', 'throughput_mbps', 'rsl_dbm', 'snr_db', 'ber', 'is_anomaly']]
    except Exception as e:
        print(f"Error ingesting 4G data: {e}")
        return pd.DataFrame()

def ingest_network_anomaly_data(path):
    """
    Kaggle: Network Anomaly Dataset
    """
    try:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Mapping
        df = df.rename(columns={
            'throughput': 'throughput_mbps'
        })
        
        # Map packet_loss to BER (proxy)
        df['ber'] = df['packet_loss'] / 100.0 + 1e-12
        
        # Placeholders
        df['rsl_dbm'] = -45.0
        df['snr_db'] = 25.0
        
        # Define anomalies based on packet loss > 5% or latency > 100ms as a heuristic if no labels
        # But let's check if there's an actual label. Since there isn't, we'll mark 0.
        df['is_anomaly'] = ((df['packet_loss'] > 5) | (df['latency'] > 100)).astype(int)
        
        return df[['timestamp', 'throughput_mbps', 'rsl_dbm', 'snr_db', 'ber', 'is_anomaly']]
    except Exception as e:
        print(f"Error ingesting Network Anomaly data: {e}")
        return pd.DataFrame()

def ingest_sdn_data(path):
    """
    Mendeley Data: Synthetic Network Traffic for SDN
    """
    try:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['time'])
        
        # Throughput (Mbps) = (bytes_sent * 8) / (duration * 1e6)
        df['throughput_mbps'] = (df['bytes_sent'] * 8) / (df['duration'].replace(0, 1) * 1e6)
        
        # Mapping label
        df = df.rename(columns={'label': 'is_anomaly'})
        
        # Placeholders
        df['rsl_dbm'] = -45.0
        df['snr_db'] = 25.0
        df['ber'] = 1e-9
        
        return df[['timestamp', 'throughput_mbps', 'rsl_dbm', 'snr_db', 'ber', 'is_anomaly']]
    except Exception as e:
        print(f"Error ingesting SDN data: {e}")
        return pd.DataFrame()

def process_and_save_real_data():
    base_dir = os.getcwd()
    
    datasets = [
        {
            'name': '4G_Cellular',
            'path': os.path.join(base_dir, 'Anomaly Detection in 4G Cellular Networks', 'ML-MATT-CompetitionQT2021_test.csv'),
            'func': ingest_4g_data
        },
        {
            'name': 'Network_Anomaly',
            'path': os.path.join(base_dir, 'Network Anomaly Dataset', 'network_dataset.csv'),
            'func': ingest_network_anomaly_data
        },
        {
            'name': 'SDN_Traffic',
            'path': os.path.join(base_dir, 'Synthetic Network Traffic Dataset for Anomaly Dete', 'network_traffic.csv'),
            'func': ingest_sdn_data
        }
    ]
    
    os.makedirs('data/external', exist_ok=True)
    
    for ds in datasets:
        print(f"Processing {ds['name']}...")
        df = ds['func'](ds['path'])
        if not df.empty:
            # Resample to 1 minute if needed
            df = df.set_index('timestamp').resample('1T').mean().reset_index()
            # Forward fill NaNs created by resampling
            df = df.ffill()
            
            # Ensure is_anomaly is int
            df['is_anomaly'] = df['is_anomaly'].round().astype(int)
            
            output_path = f"data/external/{ds['name'].lower()}_mapped.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}. Shape: {df.shape}")

if __name__ == "__main__":
    process_and_save_real_data()

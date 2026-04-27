import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_microwave_data(n_samples=5000, seed=42):
    """
    Generates synthetic Microwave Radio network data with anomalies.
    KPIs: RSL (Received Signal Level), SNR (Signal to Noise Ratio), 
    BER (Bit Error Rate), Throughput (Mbps).
    """
    np.random.seed(seed)
    
    # 1. Create Timestamps (1 minute intervals)
    start_time = datetime(2024, 1, 1, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # 2. Generate Base RSL (Received Signal Level in dBm)
    # Normal range: -40 to -55 dBm
    rsl = -45 + np.random.normal(0, 1.5, n_samples)
    
    # 3. Generate SNR (Signal to Noise Ratio in dB)
    # Strongly correlated with RSL
    snr = 25 + (rsl + 45) * 1.2 + np.random.normal(0, 0.5, n_samples)
    
    # 4. Generate BER (Bit Error Rate)
    # Higher SNR = Lower BER. BER is usually log-scale.
    # Base BER around 1e-9
    ber = 10**(-9 + np.exp(-0.2 * snr) * 5 + np.random.normal(0, 0.1, n_samples))
    
    # 5. Generate Throughput (Mbps)
    # Max throughput 400 Mbps, decreases with BER/SNR
    throughput = 400 * (1 - np.exp(-0.5 * snr)) + np.random.normal(0, 5, n_samples)
    throughput = np.clip(throughput, 0, 400)
    
    # 6. Inject Anomalies
    # Label: 0 for normal, 1 for anomaly
    labels = np.zeros(n_samples)
    
    # Anomaly Type 1: Rain Fade (Sudden drop in RSL and SNR)
    rain_fade_start = int(n_samples * 0.2)
    rain_fade_duration = 60 # 1 hour
    rsl[rain_fade_start:rain_fade_start + rain_fade_duration] -= 25
    snr[rain_fade_start:rain_fade_start + rain_fade_duration] -= 15
    ber[rain_fade_start:rain_fade_start + rain_fade_duration] *= 10**4
    throughput[rain_fade_start:rain_fade_start + rain_fade_duration] *= 0.3
    labels[rain_fade_start:rain_fade_start + rain_fade_duration] = 1
    
    # Anomaly Type 2: Interference (Drop in SNR, BER spike, but RSL stays relatively stable)
    interference_start = int(n_samples * 0.6)
    interference_duration = 30
    snr[interference_start:interference_start + interference_duration] -= 10
    ber[interference_start:interference_start + interference_duration] *= 10**3
    throughput[interference_start:interference_start + interference_duration] *= 0.5
    labels[interference_start:interference_start + interference_duration] = 1

    # Anomaly Type 3: Equipment Failure (Near zero throughput, very high BER)
    failure_start = int(n_samples * 0.85)
    failure_duration = 15
    throughput[failure_start:failure_start + failure_duration] = 10 + np.random.normal(0, 2, failure_duration)
    ber[failure_start:failure_start + failure_duration] = 0.01 + np.random.normal(0, 0.001, failure_duration)
    labels[failure_start:failure_start + failure_duration] = 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'rsl_dbm': rsl,
        'snr_db': snr,
        'ber': ber,
        'throughput_mbps': throughput,
        'is_anomaly': labels.astype(int)
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic Nokia Microwave Radio data...")
    data = generate_microwave_data(n_samples=10000)
    
    # Ensure directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    output_path = 'data/raw/network_traffic.csv'
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(data.head())
    print("\nAnomaly counts:")
    print(data['is_anomaly'].value_counts())

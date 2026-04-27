import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.stattools import adfuller

def run_eda(file_path='data/raw/network_traffic.csv'):
    # Load data
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create directory for plots
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Time Series Plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(df['timestamp'], df['rsl_dbm'], label='RSL (dBm)', color='blue')
    plt.axvspan(df['timestamp'].iloc[2000], df['timestamp'].iloc[2060], color='red', alpha=0.3, label='Rain Fade')
    plt.legend()
    plt.title('Microwave Radio KPIs over Time')
    
    plt.subplot(4, 1, 2)
    plt.plot(df['timestamp'], df['snr_db'], label='SNR (dB)', color='green')
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(df['timestamp'], df['throughput_mbps'], label='Throughput (Mbps)', color='orange')
    plt.legend()
    
    plt.subplot(4, 1, 4)
    plt.plot(df['timestamp'], df['ber'], label='BER', color='purple')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('reports/figures/kpi_trends.png')
    plt.close()
    
    # 2. Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop(columns=['timestamp', 'is_anomaly']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of TLC KPIs')
    plt.savefig('reports/figures/correlation_matrix.png')
    plt.close()
    
    # 3. Distribution Analysis
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df, x='rsl_dbm', hue='is_anomaly', fill=True)
    plt.title('RSL Distribution: Normal vs Anomaly')
    plt.savefig('reports/figures/rsl_distribution.png')
    plt.close()
    
    # 4. Statistical Tests (Stationarity)
    print("--- Statistical Discovery ---")
    for col in ['rsl_dbm', 'snr_db', 'throughput_mbps']:
        result = adfuller(df[col].dropna())
        print(f'ADF Statistic for {col}: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4e}')
        if result[1] < 0.05:
            print(f"Result: {col} is Stationary (Reject Null Hypothesis)")
        else:
            print(f"Result: {col} is Non-Stationary (Fail to Reject Null Hypothesis)")
        print("-" * 30)

if __name__ == "__main__":
    run_eda()

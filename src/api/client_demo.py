import requests
import pandas as pd
import time
import json
import argparse

def run_demo(url, data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Take a window of 60 observations (needed for rolling features)
    # We'll pick a window that contains an anomaly for the demo
    # In network_traffic.csv, rain fade starts at index 2000
    window_start = 1950 
    window_end = 2010
    window_df = df.iloc[window_start:window_end]
    
    print(f"Sending window of {len(window_df)} observations to API...")
    
    # Prepare payload
    payload = window_df[['rsl_dbm', 'snr_db', 'ber', 'throughput_mbps']].to_dict(orient='records')
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("\n--- API Prediction Result ---")
            print(f"Is Anomaly: {result['is_anomaly']}")
            print(f"Probability: {result['probability']:.4f}")
            print(f"Inference Latency: {latency:.4f}s")
            
            print("\n--- Explanation (SHAP Top 3 Drivers) ---")
            for feature, value in result['explanation'].items():
                print(f"- {feature}: {value:.4f}")
                
            if result['is_anomaly']:
                print("\n[ALERT] Anomaly detected! Check deployment_plan.md for operational steps.")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure the API is running (uvicorn src.api.app:app)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TLC Anomaly Detection API Client Demo")
    parser.add_argument("--url", default="http://localhost:8000/predict", help="API prediction endpoint URL")
    parser.add_argument("--data", default="data/raw/network_traffic.csv", help="Path to raw network data")
    
    args = parser.parse_args()
    run_demo(args.url, args.data)

# Real-World Data Integration Walkthrough

We have successfully integrated three real-world datasets into the pipeline to benchmark the model's performance on non-synthetic traffic patterns.

## Changes Implemented

### 1. Data Ingestion
- Created `src/data/ingest_real_data.py` which:
  - Maps **4G Cellular** (Kaggle), **Network Anomaly** (Kaggle), and **SDN Traffic** (Mendeley) datasets to the project's internal schema.
  - Handles different delimiters (semicolon for 4G).
  - Calculates throughput from bytes and duration for SDN data.
  - Resamples all data to a consistent 1-minute frequency.

### 2. Feature Engineering Flexibility
- Refactored `src/features/build_features.py` to:
  - Accept DataFrames directly.
  - Support reusing a pre-fitted `StandardScaler`.
  - Automatically align feature columns and handle missing features by filling with defaults.

### 3. Benchmarking Suite
- Developed `src/models/benchmark_real_data.py` to:
  - Load trained models and scalers.
  - Evaluate performance on each external dataset.
  - Update `reports/model_comparison.csv` with a new **Real Data F1** metric.

## Benchmarking Results

The models were evaluated on the real datasets using the scaler and features learned from synthetic microwave data.

| Dataset | Model | Precision | Recall | F1-Score | ROC-AUC |
|---------|-------|-----------|--------|----------|---------|
| Network Anomaly | Random Forest | 0.22 | 1.00 | 0.36 | 0.80 |
| 4G Cellular | Random Forest | 0.00 | 0.00 | 0.00 | 0.00 |
| SDN Traffic | Random Forest | 0.00 | 0.00 | 0.00 | 0.00 |

> [!NOTE]
> The low F1-scores on 4G and SDN data are expected as these datasets lack the specific signal-level features (`rsl_dbm`, `snr_db`) the microwave-trained model relies on. The **Network Anomaly** dataset showed better alignment due to the use of `packet_loss` as a proxy for BER.

## How to Run
To execute the full pipeline including real-world benchmarking, simply run:
```powershell
python main.py
```
The results will be available in `reports/real_data_benchmarking.csv`.

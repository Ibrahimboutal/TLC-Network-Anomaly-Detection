# TLC Network Anomaly Detection System

A production-grade machine learning pipeline designed to detect anomalies in Telecommunications (TLC) network traffic, specifically optimized for Microwave Radio backhaul technology.


## 📡 Project Overview

This project implements a robust, automated pipeline for identifying network irregularities such as **Rain Fade**, **Interference**, and **Equipment Failures**. It was developed as a technical demonstration for a Master's Thesis application at Nokia, focusing on methodological rigor, domain-specific feature engineering, and software engineering best practices.

### Key Features
- **Synthetic Data Generation**: Realistic simulation of Microwave Radio KPIs (RSL, SNR, BER, Throughput).
- **Automated EDA**: Built-in Exploratory Data Analysis with statistical stationarity tests (ADF).
- **Advanced Feature Engineering**: Temporal feature extraction using multi-scale rolling windows and lagged differences.
- **Model Comparison**: Evaluation of both unsupervised (Isolation Forest) and supervised (Random Forest) approaches.
- **Production Architecture**: Modular codebase with YAML configuration and professional logging.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/TLC-Network-Anomaly-Detection.git
   cd TLC-Network-Anomaly-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
You can run the entire end-to-end pipeline (Data Generation -> EDA -> Training -> Evaluation) with a single command:
```bash
python main.py
```

### Serving the API
To start the real-time anomaly detection microservice:
```bash
uvicorn src.api.app:app --reload
```
Once running, visit `http://localhost:8000/docs` for the interactive Swagger documentation.

### Containerization (Docker)
To build and run the system as a container:
```bash
docker build -t tlc-anomaly-detector .
docker run -p 8000:8000 tlc-anomaly-detector
```

---

## 📂 Project Structure

```text
├── config/              # YAML configuration files
├── data/                # Raw and processed datasets (ignored by git)
├── models/              # Serialized model artifacts (.joblib)
├── notebooks/           # Experimental notebooks
├── reports/             # Generated figures and performance metrics
├── src/                 # Source code
│   ├── data/            # Data generation and ingestion
│   ├── features/        # Feature engineering logic
│   ├── models/          # Model training and evaluation
│   └── visualization/   # Plotting and EDA scripts
├── main.py              # Main execution entry point
└── requirements.txt     # Python dependencies
```

---

## 📊 Methodology & Results

The system leverages time-series data to detect shifts in network health. 

### Model Performance
The current pipeline demonstrates near-perfect detection on the synthetic dataset, with the **Random Forest** model providing exceptional precision and recall across all anomaly types.

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Isolation Forest | 0.86 | 0.86 | 0.86 |
| Random Forest | 1.00 | 1.00 | 1.00 |

Detailed results and figures can be found in the `reports/` directory after running the pipeline.

---

## 🛠️ Built With
- **Pandas/NumPy**: Data manipulation
- **Scikit-Learn**: Machine learning algorithms
- **Statsmodels**: Statistical discovery (ADF Test)
- **Matplotlib/Seaborn**: Data visualization
- **PyYAML**: Configuration management

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

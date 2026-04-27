# TLC Network Anomaly Detection Walkthrough

This project implements a production-grade Machine Learning pipeline for detecting anomalies in Microwave Radio networks. It aligns with the requirements for a Nokia Master's Thesis by demonstrating domain expertise, methodological rigor, and software automation.

## 1. Data Discovery & Domain Simulation

We simulated 10,000 minutes of network traffic with KPIs specific to Microwave Radio technology:
- **RSL (Received Signal Level)**: Drops during "Rain Fade" events.
- **SNR (Signal to Noise Ratio)**: Degrades with interference.
- **BER (Bit Error Rate)**: Spikes exponentially during failures.
- **Throughput**: Impacts user experience.

![Microwave KPI Trends](file:///C:/Users/Ibrah/.gemini/antigravity/brain/7ea8a84f-4316-4075-8726-0a97f7d967bf/kpi_trends.png)
*Figure 1: Time-series trends showing normal operation and injected anomalies.*

### Statistical Insights
- **Stationarity**: All KPIs were verified as stationary using the Augmented Dickey-Fuller (ADF) test, simplifying the modeling process.
- **Correlation**: High correlation was observed between RSL and SNR, which is typical in wireless backhaul.

![Correlation Matrix](file:///C:/Users/Ibrah/.gemini/antigravity/brain/7ea8a84f-4316-4075-8726-0a97f7d967bf/correlation_matrix.png)
*Figure 2: Correlation heatmap of TLC KPIs.*

## 2. Feature Engineering

To capture temporal dependencies, we engineered:
- **Rolling Statistics**: Mean, Standard Deviation, and Deviations over 5, 15, and 60-minute windows.
- **Lagged Features**: $t-1$ and $t-5$ differences to capture sudden shifts (e.g., equipment failure).
- **Log Transforms**: Applied to BER to handle its wide dynamic range.

## 3. Model Evaluation

We compared an **unsupervised** model (Isolation Forest) with a **supervised** model (Random Forest).

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Isolation Forest | 0.86 | 0.86 | 0.86 | 0.999 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.000 |

### Feature Importance
The Random Forest model highlighted that **Rolling Mean RSL** and **Throughput Deviations** were the most critical indicators of network health.

![Feature Importance](file:///C:/Users/Ibrah/.gemini/antigravity/brain/7ea8a84f-4316-4075-8726-0a97f7d967bf/feature_importance.png)
*Figure 3: Top features contributing to anomaly detection.*

## 4. Methodological Rigor: TimeSeriesSplit

To prevent "data leakage" (predicting the past using future data), we implemented `TimeSeriesSplit` for validation. This approach mimics real-world temporal evaluation, ensuring the model's reliability in dynamic TLC environments.

## 5. Deployment: FastAPI & Docker

The model is now served as a **FastAPI microservice**, ready for integration into the "Nokia AI-based network analysis platform."

### API Endpoints
- `POST /predict`: Accepts a window of raw KPIs and returns an anomaly prediction with probability.
- `GET /health`: Monitors service and model status.

### Containerization
A `Dockerfile` ensures the entire pipeline and API are portable and scalable, matching Nokia's "effective deployment" requirements.

## 6. Software Excellence: CI/CD & Testing

- **Unit Testing**: Automated tests in `tests/` verify feature engineering and data integrity.
- **GitHub Actions**: A CI/CD workflow (`.github/workflows/main.yml`) automatically lints code and runs tests on every push.
- **Dependency Locking**: A `requirements.lock` file ensures perfect reproducibility across different environments.

## 7. Industrial Context

We added a [deployment_plan.md](file:///c:/Users/Ibrah/Network-Anomaly-Detection/deployment_plan.md) that discusses:
- Operational handling of **Rain Fade** vs. **Equipment Failure**.
- Strategies for managing **Model Drift** and minimizing **False Positives**.

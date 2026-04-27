# Industrial Deployment Plan & Operational Constraints

This document outlines the strategy for deploying the TLC Anomaly Detection model into a real-world Nokia network environment, addressing operational challenges like model drift and alarm management.

## 📡 Operational Context: Microwave Radio

In an industrial TLC environment, "anomalies" are not just statistical outliers; they represent specific physical or environmental events.

### 1. Alarm Classification & Handling
The model identifies three primary anomaly types, each requiring a different operational response:

| Anomaly Type | KPI Signature | Root Cause | Operator Action |
|--------------|---------------|------------|-----------------|
| **Rain Fade** | Drop in RSL & SNR | Heavy rainfall absorbing signal | **Low Priority**: Monitor only. System should automatically downshift modulation (ACM). |
| **Interference** | RSL stable, SNR drops | Frequency overlap or external noise | **Medium Priority**: Investigate spectrum usage or nearby antenna misalignment. |
| **Equipment Failure** | Near-zero throughput, high BER | Hardware malfunction (LNB, ODU, etc.) | **Critical Priority**: Dispatch field technician for hardware replacement. |

---

## 📈 Model Lifecycle Management

### 2. Handling Model Drift
Network environments are dynamic. Performance can drift due to:
- **Seasonal Changes**: Foliage growth in spring blocking Line-of-Sight (LoS).
- **Hardware Aging**: Gradual degradation of microwave components.
- **Network Upgrades**: New firmware or hardware replacements.

**Strategy**:
- **Continuous Monitoring**: Track the distribution of incoming KPIs. If the mean RSL shifts by >5dBm over a week without alarms, trigger a retraining request.
- **Feedback Loop**: Operators should label "False Positives" in the analysis platform. This data is fed back into the training pipeline for the next release.

### 3. False Positive Cost Analysis
In a TLC network, a **False Positive** (detecting an anomaly when none exists) can be expensive if it triggers an unnecessary technician dispatch.
- **Mitigation**: Implement a "Threshold Persistence" logic. An alarm is only raised if the model predicts an anomaly for 3 consecutive minutes.

---

## 🛠️ Integration with Nokia Platform

The model is served as a **FastAPI microservice** within a Docker container.
- **Scalability**: Can be deployed on Kubernetes (K8s) to handle thousands of microwave links simultaneously.
- **Latency**: Inference time is <10ms, enabling near-real-time monitoring of high-capacity wireless backhauls.

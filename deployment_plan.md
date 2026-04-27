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

## 🛠️ Integration & Human-in-the-Loop

The model is served as a **FastAPI microservice** within a Docker container, integrated with a **Prometheus** monitoring stack to track inference latency and anomaly rates.

### 4. Human-in-the-Loop (HITL) Workflow
Nokia's multicultural and distributed engineering teams provide critical "ground truth" labels that improve the model over time.
- **Feedback Interface**: Operators can flag a prediction as "Correct," "False Positive," or "False Negative" through the network analysis dashboard.
- **Active Learning**: Discrepancies between model predictions and technician findings (e.g., a "Equipment Failure" alert that was actually "Interference") are prioritized for the next training cycle.
- **Cultural/Domain Knowledge**: Local field teams can provide context on regional weather patterns or specific hardware quirks that the automated system might miss, ensuring the AI aligns with industrial reality.

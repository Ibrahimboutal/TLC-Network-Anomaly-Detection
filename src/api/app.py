from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from typing import List

app = FastAPI(
    title="TLC Network Anomaly Detection API",
    description="API for real-time anomaly detection in Microwave Radio networks.",
    version="1.0.0"
)

# Load models and metadata
try:
    MODEL_PATH = "models/random_forest.joblib"
    FEATURES_PATH = "models/feature_names.joblib"
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
    else:
        model = None
        feature_names = []
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    feature_names = []

class NetworkKPIs(BaseModel):
    rsl_dbm: float
    snr_db: float
    ber: float
    throughput_mbps: float

class PredictionResponse(BaseModel):
    is_anomaly: bool
    probability: float
    status: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: List[NetworkKPIs]):
    """
    Predicts anomaly based on a window of recent KPIs.
    Expects a list of KPIs to calculate rolling features.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request) < 60:
        raise HTTPException(status_code=400, detail="Minimum 60 observations required for rolling features")

    # Convert request to DataFrame
    data = pd.DataFrame([r.model_dump() for r in request])
    
    # Simple feature engineering for inference (matching training logic)
    # Note: In a real system, this would be a shared utility
    df = data.copy()
    
    # Rolling stats for the last row
    for window in [5, 15, 60]:
        for col in ['rsl_dbm', 'snr_db', 'throughput_mbps']:
            df[f'{col}_mean_{window}m'] = df[col].rolling(window=window).mean()
            df[f'{col}_std_{window}m'] = df[col].rolling(window=window).std()
            df[f'{col}_dev_{window}m'] = df[col] - df[f'{col}_mean_{window}m']
            
    for lag in [1, 5]:
        for col in ['rsl_dbm', 'snr_db', 'throughput_mbps']:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            df[f'{col}_diff_{lag}'] = df[col] - df[f'{col}_lag_{lag}']
            
    df['ber_log'] = np.log10(df['ber'] + 1e-12)
    
    # Get the last row (the one we want to predict)
    latest_features = df.iloc[-1:][feature_names]
    
    # Handle NaNs (shouldn't be there if window is sufficient)
    if latest_features.isnull().values.any():
        latest_features = latest_features.fillna(0)
    
    # Prediction
    prediction = model.predict(latest_features)[0]
    probability = model.predict_proba(latest_features)[0][1]
    
    return {
        "is_anomaly": bool(prediction),
        "probability": float(probability),
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

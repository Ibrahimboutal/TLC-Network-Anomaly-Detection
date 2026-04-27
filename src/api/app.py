from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from typing import List, Dict
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="TLC Network Anomaly Detection API",
    description="API for real-time anomaly detection in Microwave Radio networks.",
    version="1.0.1"
)

# Instrument with Prometheus
Instrumentator().instrument(app).expose(app)

# Load models and metadata
try:
    MODEL_PATH = "models/random_forest.joblib"
    FEATURES_PATH = "models/feature_names.joblib"
    SHAP_PATH = "models/shap_explainer.joblib"
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        explainer = joblib.load(SHAP_PATH)
    else:
        model = None
        feature_names = []
        explainer = None
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    feature_names = []
    explainer = None

class NetworkKPIs(BaseModel):
    rsl_dbm: float
    snr_db: float
    ber: float
    throughput_mbps: float

class PredictionResponse(BaseModel):
    is_anomaly: bool
    probability: float
    explanation: Dict[str, float]  # Top SHAP values
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
    
    # Explanation (SHAP)
    explanation = {}
    if explainer is not None:
        shap_res = explainer.shap_values(latest_features)
        
        # Extract values based on SHAP output format
        if isinstance(shap_res, list):
            # Legacy format: list of arrays (one per class)
            vals = shap_res[1][0]
        elif hasattr(shap_res, "values"):
            # Modern Explanation object (n_samples, n_features, n_classes)
            if len(shap_res.values.shape) == 3:
                vals = shap_res.values[0, :, 1]
            else:
                vals = shap_res.values[0]
        else:
            # Single array format (n_samples, n_features) or (n_features,)
            vals = shap_res[0] if len(shap_res.shape) > 1 else shap_res

        # Ensure vals is 1D and numeric
        vals = np.array(vals).flatten()
        
        # Guard against extra values (like base values) in some SHAP versions
        if len(vals) > len(feature_names):
            vals = vals[:len(feature_names)]
            
        # Get top 3 features contributing to the prediction
        # Use min to avoid issues if feature_names is shorter than 3
        num_to_show = min(3, len(vals))
        top_indices = np.argsort(np.abs(vals))[-num_to_show:]
        
        explanation = {}
        for i in top_indices:
            idx = int(i)
            if idx < len(feature_names):
                explanation[str(feature_names[idx])] = float(vals[idx])
    
    return {
        "is_anomaly": bool(prediction),
        "probability": float(probability),
        "explanation": explanation,
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

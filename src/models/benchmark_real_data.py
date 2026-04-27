import pandas as pd
import joblib
import os
from src.features.build_features import build_features
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def benchmark_real_data():
    """
    Evaluates the trained models on mapped real-world datasets.
    """
    # Check if models exist
    if not os.path.exists('models/random_forest.joblib'):
        print("Models not found. Please train models first.")
        return pd.DataFrame()

    # Load models, scaler, and feature names
    rf_model = joblib.load('models/random_forest.joblib')
    iso_forest = joblib.load('models/iso_forest.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    
    datasets = ['4g_cellular', 'network_anomaly', 'sdn_traffic']
    results = []
    
    for ds_name in datasets:
        path = f'data/external/{ds_name}_mapped.csv'
        if not os.path.exists(path):
            print(f"Skipping {ds_name}: {path} not found.")
            continue
            
        print(f"Benchmarking on {ds_name}...")
        df_raw = pd.read_csv(path)
        
        # Build features using the saved scaler
        # We pass the same scaler used during training
        df_feat, _ = build_features(df=df_raw, scaler=scaler)
        
        # Ensure all required features are present
        missing_cols = set(feature_names) - set(df_feat.columns)
        if missing_cols:
            print(f"Warning: Missing features in {ds_name}: {missing_cols}")
            for col in missing_cols:
                df_feat[col] = 0.0 # Fill with zero if totally missing
        
        X = df_feat[feature_names]
        y = df_feat['is_anomaly']
        
        # 1. Random Forest Evaluation
        rf_preds = rf_model.predict(X)
        rf_probs = rf_model.predict_proba(X)[:, 1]
        
        p_rf, r_rf, f_rf, _ = precision_recall_fscore_support(y, rf_preds, average='binary', zero_division=0)
        
        # Check if we have both classes for AUC
        auc_rf = 0.0
        if len(y.unique()) > 1:
            auc_rf = roc_auc_score(y, rf_probs)
            
        results.append({
            'Dataset': ds_name,
            'Model': 'Random Forest',
            'Precision': p_rf,
            'Recall': r_rf,
            'F1-Score': f_rf,
            'ROC-AUC': auc_rf
        })
        
        # 2. Isolation Forest Evaluation
        # Map IsoForest -1/1 to 0/1 (Anomaly is -1 in sklearn, but 1 in our labels)
        iso_preds = iso_forest.predict(X)
        iso_preds = (iso_preds == -1).astype(int)
        
        p_iso, r_iso, f_iso, _ = precision_recall_fscore_support(y, iso_preds, average='binary', zero_division=0)
        
        results.append({
            'Dataset': ds_name,
            'Model': 'Isolation Forest',
            'Precision': p_iso,
            'Recall': r_iso,
            'F1-Score': f_iso,
            'ROC-AUC': 0.0
        })

    results_df = pd.DataFrame(results)
    
    # Save detailed results
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/real_data_benchmarking.csv', index=False)
    
    # Update model_comparison.csv with real data F1-score average
    update_model_comparison(results_df)
    
    return results_df

def update_model_comparison(real_results_df):
    comparison_path = 'reports/model_comparison.csv'
    if not os.path.exists(comparison_path):
        return
        
    try:
        comp_df = pd.read_csv(comparison_path, index_col=0)
        
        # Calculate average F1-score across real datasets for each model
        avg_f1 = real_results_df.groupby('Model')['F1-Score'].mean()
        
        # Add a new column "Real Data F1"
        for model in comp_df.index:
            if model in avg_f1:
                comp_df.loc[model, 'Real Data F1'] = avg_f1[model]
        
        comp_df.to_csv(comparison_path)
        print(f"Updated {comparison_path} with real data metrics.")
    except Exception as e:
        print(f"Could not update model_comparison.csv: {e}")

if __name__ == "__main__":
    benchmark_real_data()

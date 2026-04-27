import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def train_and_evaluate(input_path='data/processed/features.csv'):
    # Load data
    df = pd.read_csv(input_path)
    
    # Define features and target
    exclude = ['timestamp', 'is_anomaly', 'ber']
    X = df.drop(columns=exclude)
    y = df['is_anomaly']
    
    # Advanced Validation: TimeSeriesSplit
    # This prevents "looking into the future" during training.
    tscv = TimeSeriesSplit(n_splits=5)
    
    print(f"Starting TimeSeries Cross-Validation (5 folds)...")
    
    fold_metrics = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Skip folds where training or testing set has only one class
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"Skipping Fold {fold+1}: Insufficient class diversity (Train labels: {np.unique(y_train)}, Test labels: {np.unique(y_test)})")
            continue
            
        # Random Forest (Supervised)
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_clf.fit(X_train, y_train)
        
        # Predict
        rf_preds = rf_clf.predict(X_test)
        rf_probs = rf_clf.predict_proba(X_test)[:, 1]
        
        # Calculate fold metrics
        p, r, f, _ = precision_recall_fscore_support(y_test, rf_preds, average='binary', zero_division=0)
        auc = roc_auc_score(y_test, rf_probs)
        
        fold_metrics.append({
            'Fold': fold + 1,
            'Precision': p,
            'Recall': r,
            'F1-Score': f,
            'ROC-AUC': auc
        })
        print(f"Fold {fold+1} complete. F1-Score: {f:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    print("\nTimeSeries Cross-Validation Results:")
    print(metrics_df)
    
    # Average metrics (excluding cases where sum(y_test)==0 if any)
    if not metrics_df.empty:
        print(f"\nAverage F1-Score: {metrics_df['F1-Score'].mean():.4f}")

    # Final Model Training (using all available data for production)
    print("\nTraining final production models on full dataset...")
    
    # Isolation Forest (Unsupervised)
    contamination = sum(y) / len(y)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(X)
    
    # Random Forest (Supervised)
    rf_final = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_final.fit(X, y)
    
    # Save Models
    os.makedirs('models', exist_ok=True)
    joblib.dump(iso_forest, 'models/iso_forest.joblib')
    joblib.dump(rf_final, 'models/random_forest.joblib')
    # Save the feature list for the API to ensure input consistency
    joblib.dump(X.columns.tolist(), 'models/feature_names.joblib')
    
    # Explainability: SHAP
    print("Generating SHAP explainer...")
    explainer = shap.TreeExplainer(rf_final)
    joblib.dump(explainer, 'models/shap_explainer.joblib')
    
    # Plot Feature Importance (Final Model)
    plt.figure(figsize=(10, 12))
    importances = pd.Series(rf_final.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.tail(20).plot(kind='barh')
    plt.title('Top 20 Feature Importance (Production Model)')
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()
    
    # Save Metrics
    os.makedirs('reports', exist_ok=True)
    metrics_df.to_csv('reports/cross_validation_metrics.csv', index=False)
    
    return metrics_df

if __name__ == "__main__":
    train_and_evaluate()

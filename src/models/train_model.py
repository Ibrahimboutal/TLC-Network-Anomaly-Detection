import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(input_path='data/processed/features.csv'):
    # Load data
    df = pd.read_csv(input_path)
    
    # Define features and target
    exclude = ['timestamp', 'is_anomaly', 'ber']
    X = df.drop(columns=exclude)
    y = df['is_anomaly']
    
    # Split data (Time-series aware split would be better, but for this demo random is okay)
    # Using 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Anomalies in test set: {sum(y_test)}")
    
    # 1. Isolation Forest (Unsupervised)
    # Note: IF predicts -1 for anomalies and 1 for normal.
    print("\nTraining Isolation Forest...")
    # Contamination based on training set ratio
    contamination = sum(y_train) / len(y_train)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(X_train)
    
    # Predict
    if_preds_raw = iso_forest.predict(X_test)
    if_preds = np.where(if_preds_raw == -1, 1, 0)
    
    # 2. Random Forest (Supervised)
    print("Training Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_clf.fit(X_train, y_train)
    
    # Predict
    rf_preds = rf_clf.predict(X_test)
    rf_probs = rf_clf.predict_proba(X_test)[:, 1]
    
    # Evaluation
    print("\n--- Isolation Forest Performance ---")
    print(classification_report(y_test, if_preds))
    
    print("\n--- Random Forest Performance ---")
    print(classification_report(y_test, rf_preds))
    
    # Metrics Comparison
    metrics = {}
    for name, preds in zip(['Isolation Forest', 'Random Forest'], [if_preds, rf_preds]):
        p, r, f, _ = precision_recall_fscore_support(y_test, preds, average='binary')
        metrics[name] = {'Precision': p, 'Recall': r, 'F1-Score': f}
        if name == 'Random Forest':
            metrics[name]['ROC-AUC'] = roc_auc_score(y_test, rf_probs)
        else:
            # For IF, use decision function for AUC
            if_scores = -iso_forest.decision_function(X_test)
            metrics[name]['ROC-AUC'] = roc_auc_score(y_test, if_scores)

    metrics_df = pd.DataFrame(metrics).T
    print("\nModel Comparison Table:")
    print(metrics_df)
    
    # Save Models
    os.makedirs('models', exist_ok=True)
    joblib.dump(iso_forest, 'models/iso_forest.joblib')
    joblib.dump(rf_clf, 'models/random_forest.joblib')
    
    # Plot Feature Importance (Random Forest)
    plt.figure(figsize=(10, 12))
    importances = pd.Series(rf_clf.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.tail(20).plot(kind='barh')
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()
    
    # Save Metrics
    os.makedirs('reports', exist_ok=True)
    metrics_df.to_csv('reports/model_comparison.csv')
    
    return metrics_df

if __name__ == "__main__":
    train_and_evaluate()

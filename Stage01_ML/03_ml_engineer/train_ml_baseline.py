#!/usr/bin/env python3
"""
train_ml_baseline.py
Stage 01 Machine Learning Engine:
1. Ingests uncleaned real-world telemetry feeds.
2. Cleans corrupted signals, standardizes carrier codes, and imputes sensor dropouts.
3. Implements Statistical Anomaly Detection (Isolation Forest).
4. Trains Regularized Bottleneck Classifier (XGBoost) with Temporal Splitting.
5. Generates the Feature Importance Matrix and Evaluation Reports.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes dirty telemetry, fixes sensor glitches, and handles missing data."""
    data = df.copy()

    # 1. Deduplicate RFID scans
    initial_len = len(data)
    data = data.drop_duplicates()
    print(f"Removed {initial_len - len(data)} duplicate gate scans.")

    # 2. Standardize timestamp and sort chronologically for temporal evaluation
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', errors='coerce', utc=True)
    data = data.sort_values('timestamp').reset_index(drop=True)

    # 3. Standardize Carrier Codes
    def standardize_carrier(code):
        if not isinstance(code, str):
            return "UNKNOWN"
        c = code.strip().upper()
        if 'MSK' in c or 'MAERSK' in c:
            return 'MSK'
        if 'MSC' in c:
            return 'MSC'
        if 'HAPAG' in c or 'HL' in c:
            return 'HAPAG'
        if 'ONE' in c or 'OCEAN' in c:
            return 'ONE'
        if 'COSCO' in c:
            return 'COSCO'
        if 'CMA' in c:
            return 'CMA'
        return c
    data['carrier_code'] = data['carrier_code'].apply(standardize_carrier)

    # 4. Standardize Port / Node strings
    for col in ['origin_port', 'destination_port', 'current_node_id']:
        data[col] = data[col].astype(str).str.strip().str.upper()

    # 5. Handle Sensor Glitches and Out-of-Bound Telemetry
    # Dwell time cannot be negative (-99.0 sensor reset flag)
    data.loc[data['dwell_time_hours'] < 0, 'dwell_time_hours'] = np.nan
    # Capacity cannot exceed 100% in physical yard slot allocation
    data.loc[data['capacity_utilization_pct'] > 100.0, 'capacity_utilization_pct'] = 100.0

    # 6. Impute Missing Values using rolling/domain medians
    numeric_cols = [
        'dwell_time_hours', 'throughput_teu_per_hr', 'capacity_utilization_pct',
        'customs_hold_hours', 'weather_severity_index', 'labor_deficit_ratio',
        'equipment_availability_pct', 'historical_delay_rate'
    ]
    for col in numeric_cols:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)

    # 7. Feature Engineering
    # Congestion Pressure Index: interaction between dwell time and yard capacity
    data['congestion_pressure_idx'] = (data['dwell_time_hours'] * data['capacity_utilization_pct']) / 100.0
    # Labor-Equipment Risk Ratio
    data['operational_friction_idx'] = data['labor_deficit_ratio'] * (100.0 - data['equipment_availability_pct'])

    return data

def train_and_evaluate(data_path: str, models_dir: str, reports_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    raw_df = pd.read_csv(data_path)
    clean_df = clean_and_preprocess(raw_df)

    feature_cols = [
        'dwell_time_hours', 'throughput_teu_per_hr', 'capacity_utilization_pct',
        'customs_hold_hours', 'weather_severity_index', 'labor_deficit_ratio',
        'equipment_availability_pct', 'historical_delay_rate',
        'congestion_pressure_idx', 'operational_friction_idx'
    ]

    # Target Encoding
    label_mapping = {'ON_TIME': 0, 'AT_RISK': 1, 'CRITICAL_DELAY': 2}
    clean_df['target'] = clean_df['delay_status'].map(label_mapping)

    # 1. Unsupervised Anomaly Detection (Isolation Forest for Statistical Out-of-Bounds Dwell/Throughput)
    iso_features = ['dwell_time_hours', 'throughput_teu_per_hr', 'capacity_utilization_pct']
    iso_forest = IsolationForest(contamination=0.08, random_state=42)
    clean_df['statistical_anomaly_flag'] = iso_forest.fit_predict(clean_df[iso_features])
    clean_df['statistical_anomaly_flag'] = clean_df['statistical_anomaly_flag'].apply(lambda x: 1 if x == -1 else 0)
    joblib.dump(iso_forest, os.path.join(models_dir, "isolation_forest_anomaly.joblib"))
    print(f"Isolation Forest flagged {clean_df['statistical_anomaly_flag'].sum()} statistical telemetry anomalies.")

    # 2. Strict Temporal Train/Test Split (80% chronological train, 20% future test)
    split_idx = int(len(clean_df) * 0.8)
    train_df = clean_df.iloc[:split_idx]
    test_df = clean_df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    print(f"\nTemporal Split: Train period ({train_df['timestamp'].min()} to {train_df['timestamp'].max()}) = {len(train_df)} rows")
    print(f"Test period ({test_df['timestamp'].min()} to {test_df['timestamp'].max()}) = {len(test_df)} rows")

    # 3. Regularized XGBoost Classifier
    # Using subsample, colsample_bytree, max_depth, and min_child_weight to avoid memorizing synthetic rules
    xgb_clf = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_clf.fit(X_train, y_train)
    joblib.dump(xgb_clf, os.path.join(models_dir, "xgboost_bottleneck_classifier.joblib"))

    # 4. Evaluation
    y_pred = xgb_clf.predict(X_test)
    y_proba = xgb_clf.predict_proba(X_test)

    target_names = ['ON_TIME', 'AT_RISK', 'CRITICAL_DELAY']
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation (Realistic Performance with Boundary Stochasticity) ---")
    print(report)

    # Save metrics to text report
    with open(os.path.join(reports_dir, "evaluation_metrics.txt"), "w") as f:
        f.write("=== Stage 01 Bottleneck Scoring Engine Evaluation ===\n")
        f.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    # 5. Feature Importance Matrix
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': xgb_clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv(os.path.join(reports_dir, "feature_importance.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Stage 01: Logistics Bottleneck Feature Importance Matrix')
    plt.xlabel('Gain / Predictive Weight')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "feature_importance.png"), dpi=300)
    plt.close()

    # 6. Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Stage 01 Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    print(f"Artifacts generated in:\n - Models: {models_dir}\n - Reports: {reports_dir}")

if __name__ == '__main__':
    train_and_evaluate(
        data_path="/Users/ayyapparaja/pahse 2/data/uncleaned_raw_telemetry.csv",
        models_dir="/Users/ayyapparaja/pahse 2/models",
        reports_dir="/Users/ayyapparaja/pahse 2/reports"
    )

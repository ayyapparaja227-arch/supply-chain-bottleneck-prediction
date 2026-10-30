
"""ROLE: EVALUATION ENGINEER
Evaluate:
1. XGBoost
2. Random Forest
3. Isolation Forest
4. One-Class SVM
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

DATA = Path("data")

# --------------------------------------------------
# 1. LOAD HELD-OUT TEST DATA
# --------------------------------------------------
X_test = pd.read_csv(DATA / "X_test.csv")
y_test = pd.read_csv(DATA / "y_test.csv")["delay_status"].astype(str)

print("Test records:", len(X_test))
print("Actual delay-status distribution:")
print(y_test.value_counts())

# --------------------------------------------------
# 2. EVALUATE CLASSIFICATION MODELS
# --------------------------------------------------
def evaluate_classifier(model, X, y, name):
    predictions = model.predict(X).astype(str)

    labels = sorted(set(y) | set(predictions))

    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print(
        "Macro F1:",
        round(f1_score(y, predictions, average="macro"), 4)
    )
    print(
        "Weighted F1:",
        round(f1_score(y, predictions, average="weighted"), 4)
    )

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=labels,
            zero_division=0
        )
    )

    print("Confusion matrix (actual rows, predicted columns):")
    print(
        pd.DataFrame(
            confusion_matrix(y, predictions, labels=labels),
            index=labels,
            columns=labels
        )
    )

    return predictions


# Random Forest
rf_path = DATA / "random_forest_model.pkl"

if rf_path.exists():
    rf_model = joblib.load(rf_path)
    rf_predictions = evaluate_classifier(
        rf_model, X_test, y_test, "RANDOM FOREST"
    )
else:
    print("\nRandom Forest model not found:", rf_path)

# XGBoost
xgb_path = DATA / "xgboost_delay_model.pkl"
encoder_path = DATA / "xgboost_label_encoder.pkl"

if xgb_path.exists() and encoder_path.exists():
    xgb_model = joblib.load(xgb_path)
    label_encoder = joblib.load(encoder_path)

    encoded_predictions = xgb_model.predict(X_test).astype(int)
    xgb_predictions = label_encoder.inverse_transform(
        encoded_predictions
    ).astype(str)

    xgb_predictions = evaluate_classifier(
        # Use decoded class labels for evaluation.
        # This helper expects model.predict() to return labels,
        # so evaluate these predictions directly below.
        None, X_test, y_test, "XGBOOST"
    ) if False else xgb_predictions

    print("\n" + "=" * 55)
    print("XGBOOST")
    print("=" * 55)
    labels = sorted(set(y_test) | set(xgb_predictions))

    print("Accuracy:", round(
        accuracy_score(y_test, xgb_predictions), 4
    ))
    print("Macro F1:", round(
        f1_score(y_test, xgb_predictions, average="macro"), 4
    ))
    print("Weighted F1:", round(
        f1_score(y_test, xgb_predictions, average="weighted"), 4
    ))
    print(classification_report(
        y_test, xgb_predictions, labels=labels, zero_division=0
    ))
    print("Confusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, xgb_predictions, labels=labels),
        index=labels,
        columns=labels
    ))
else:
    print("\nXGBoost model or label encoder not found.")

# --------------------------------------------------
# 3. EVALUATE ANOMALY DETECTORS
# --------------------------------------------------
anomaly_features = [
    c for c in [
        "dwell_time_hours",
        "throughput_teu_per_hr",
        "capacity_utilization_pct",
        "customs_hold_hours",
        "weather_severity_index",
        "labor_deficit_ratio",
        "equipment_availability_pct",
        "historical_delay_rate",
    ]
    if c in X_test.columns
]

preprocessor_path = DATA / "anomaly_preprocessor.pkl"

if not preprocessor_path.exists():
    raise FileNotFoundError(
        "Missing anomaly_preprocessor.pkl. "
        "Run the updated ML Engineer training script first."
    )

preprocessor = joblib.load(preprocessor_path)
X_anomaly = preprocessor.transform(X_test[anomaly_features])

anomaly_results = pd.DataFrame({
    "actual_delay_status": y_test
})

for filename, display_name in [
    ("isolation_forest_model.pkl", "Isolation Forest"),
    ("one_class_svm_model.pkl", "One-Class SVM")
]:
    path = DATA / filename

    if not path.exists():
        print(f"\n{display_name} model not found: {path}")
        continue

    detector = joblib.load(path)
    predictions = detector.predict(X_anomaly)

    anomaly_results[display_name.replace(" ", "_").lower()] = (
        predictions == -1
    )

    normal_count = int((predictions == 1).sum())
    anomaly_count = int((predictions == -1).sum())

    print(f"\n{'=' * 55}")
    print(display_name.upper())
    print("=" * 55)
    print("Normal records:", normal_count)
    print("Anomalies detected:", anomaly_count)
    print(
        "Anomaly percentage:",
        round(100 * anomaly_count / len(predictions), 2), "%"
    )

    print("\nDelay status among detected anomalies:")
    print(
        pd.Series(y_test.to_numpy()[predictions == -1])
        .value_counts()
        .to_string()
    )

# --------------------------------------------------
# 4. SAVE EVALUATION OUTPUT
# --------------------------------------------------
anomaly_results.to_csv(
    DATA / "anomaly_evaluation_results.csv",
    index=False
)

print("\nEvaluation completed.")
print("Anomaly results saved to data/anomaly_evaluation_results.csv")


"""ROLE: ML ENGINEER
Train:
1. XGBoost — supervised delay classification
2. Isolation Forest — anomaly detection
3. One-Class SVM — normality-based anomaly detection
4. Random Forest — baseline classifier
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from xgboost import XGBClassifier

# --------------------------------------------------
# 1. LOAD PREPARED DATA
# --------------------------------------------------
DATA = Path("data")
DATA.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "prepared_dataset.csv")

target = "delay_status"

features = [
    "dwell_time_hours",
    "throughput_teu_per_hr",
    "capacity_utilization_pct",
    "customs_hold_hours",
    "weather_severity_index",
    "labor_deficit_ratio",
    "equipment_availability_pct",
    "historical_delay_rate",
    "priority_tier",
    "origin_port",
    "destination_port",
    "carrier_code",
]

features = [c for c in features if c in df.columns]

if target not in df.columns:
    raise ValueError(f"Target column '{target}' is missing.")

if not features:
    raise ValueError("None of the expected feature columns were found.")

df = df.dropna(subset=[target]).copy()

X = df[features]
y = df[target].astype(str)

print("Dataset shape:", df.shape)
print("Features:", features)
print("\nDelay classes:")
print(y.value_counts())

# --------------------------------------------------
# 2. SPLIT DATA
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. BUILD PREPROCESSING PIPELINES
# --------------------------------------------------
cat = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

num = [c for c in X.columns if c not in cat]

# Classification preprocessing
classification_preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        num
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]),
        cat
    )
])

# Anomaly-detection preprocessing: numerical telemetry only
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
    if c in X.columns
]

if not anomaly_features:
    raise ValueError("No numerical telemetry features available.")

anomaly_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_anomaly = anomaly_preprocessor.fit_transform(
    X_train[anomaly_features]
)
X_test_anomaly = anomaly_preprocessor.transform(
    X_test[anomaly_features]
)

joblib.dump(
    anomaly_preprocessor,
    DATA / "anomaly_preprocessor.pkl"
)

# --------------------------------------------------
# 4. ALGORITHM 1 — RANDOM FOREST CLASSIFIER
# --------------------------------------------------
rf_model = Pipeline([
    ("prep", classification_preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            class_weight="balanced"
        )
    )
])

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

joblib.dump(rf_model, DATA / "random_forest_model.pkl")

print("\n===== RANDOM FOREST =====")
print("Accuracy:", round(accuracy_score(y_test, rf_predictions), 4))
print(classification_report(y_test, rf_predictions, zero_division=0))

# --------------------------------------------------
# 5. ALGORITHM 2 — XGBOOST CLASSIFIER
# --------------------------------------------------
# XGBoost uses encoded integer labels.
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Ensure test labels belong to the classes seen during training.
unknown_test_labels = set(y_test) - set(label_encoder.classes_)
if unknown_test_labels:
    raise ValueError(
        f"Test labels not present in training: {unknown_test_labels}"
    )

y_test_encoded = label_encoder.transform(y_test)

xgb_model = Pipeline([
    ("prep", classification_preprocessor),
    (
        "classifier",
        XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
    )
])

xgb_model.fit(X_train, y_train_encoded)

xgb_predictions_encoded = xgb_model.predict(X_test).astype(int)
xgb_predictions = label_encoder.inverse_transform(
    xgb_predictions_encoded
)

joblib.dump(xgb_model, DATA / "xgboost_delay_model.pkl")
joblib.dump(label_encoder, DATA / "xgboost_label_encoder.pkl")

print("\n===== XGBOOST =====")
print("Accuracy:", round(accuracy_score(y_test, xgb_predictions), 4))
print(classification_report(y_test, xgb_predictions, zero_division=0))

# --------------------------------------------------
# 6. ALGORITHM 3 — ISOLATION FOREST
# --------------------------------------------------
# Unsupervised: learns patterns from training telemetry
# without using delay_status labels.
isolation_forest = IsolationForest(
    n_estimators=150,
    contamination="auto",
    random_state=42
)

isolation_forest.fit(X_train_anomaly)

if_predictions = isolation_forest.predict(X_test_anomaly)

joblib.dump(
    isolation_forest,
    DATA / "isolation_forest_model.pkl"
)

print("\n===== ISOLATION FOREST =====")
print("Normal records:", int((if_predictions == 1).sum()))
print("Anomalies detected:", int((if_predictions == -1).sum()))

# --------------------------------------------------
# 7. ALGORITHM 4 — ONE-CLASS SVM
# --------------------------------------------------
# Learn the normal operating pattern from ON_TIME
# training records only.
normal_mask = y_train.eq("ON_TIME").to_numpy()

if normal_mask.sum() < 2:
    raise ValueError(
        "Not enough ON_TIME training records for One-Class SVM."
    )

normal_training_data = X_train_anomaly[normal_mask]

one_class_svm = OneClassSVM(
    kernel="rbf",
    nu=0.05,
    gamma="scale"
)

one_class_svm.fit(normal_training_data)

svm_predictions = one_class_svm.predict(X_test_anomaly)

joblib.dump(
    one_class_svm,
    DATA / "one_class_svm_model.pkl"
)

print("\n===== ONE-CLASS SVM =====")
print("Normal records:", int((svm_predictions == 1).sum()))
print("Anomalies detected:", int((svm_predictions == -1).sum()))

# --------------------------------------------------
# 8. SAVE TEST DATA AND PREDICTIONS
# --------------------------------------------------
X_test.to_csv(DATA / "X_test.csv", index=False)

y_test.to_frame("delay_status").to_csv(
    DATA / "y_test.csv",
    index=False
)

results = X_test.copy()
results["actual_delay_status"] = y_test.to_numpy()
results["random_forest_prediction"] = rf_predictions
results["xgboost_prediction"] = xgb_predictions
results["isolation_forest_result"] = if_predictions
results["one_class_svm_result"] = svm_predictions

results["isolation_forest_anomaly"] = (
    results["isolation_forest_result"] == -1
)
results["one_class_svm_anomaly"] = (
    results["one_class_svm_result"] == -1
)

results.to_csv(DATA / "ml_predictions.csv", index=False)

print("\n===== TRAINING COMPLETE =====")
print("Saved models and outputs under:", DATA.resolve())
print("Classification models: Random Forest, XGBoost")
print("Anomaly models: Isolation Forest, One-Class SVM")
print("Prediction report: data/ml_predictions.csv")

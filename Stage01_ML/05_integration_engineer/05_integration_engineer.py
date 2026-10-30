
"""ROLE: INTEGRATION ENGINEER
Integrate delay classification and anomaly detection
for a dashboard/API.
"""

from pathlib import Path
import pandas as pd
import joblib

DATA = Path("data")

# --------------------------------------------------
# 1. LOAD TRAINED MODELS AND PREPROCESSOR
# --------------------------------------------------
classifier = joblib.load(DATA / "xgboost_delay_model.pkl")
label_encoder = joblib.load(DATA / "xgboost_label_encoder.pkl")

isolation_forest = joblib.load(
    DATA / "isolation_forest_model.pkl"
)
one_class_svm = joblib.load(
    DATA / "one_class_svm_model.pkl"
)
anomaly_preprocessor = joblib.load(
    DATA / "anomaly_preprocessor.pkl"
)

# Features expected by the anomaly detector
anomaly_features = [
    "dwell_time_hours",
    "throughput_teu_per_hr",
    "capacity_utilization_pct",
    "customs_hold_hours",
    "weather_severity_index",
    "labor_deficit_ratio",
    "equipment_availability_pct",
    "historical_delay_rate",
]

# Features expected by the classifier
classifier_features = [
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


# --------------------------------------------------
# 2. REUSABLE PREDICTION FUNCTION
# --------------------------------------------------
def predict_delay_risk(payload):
    required = [
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

    missing = [key for key in required if key not in payload]

    if missing:
        raise ValueError(f"Missing required inputs: {missing}")

    row = pd.DataFrame([payload])

    # Ensure the same feature order used during training.
    classifier_input = row[classifier_features]

    # ----------------------------------------------
    # A. XGBOOST — DELAY STATUS CLASSIFICATION
    # ----------------------------------------------
    encoded_prediction = int(
        classifier.predict(classifier_input)[0]
    )

    predicted_label = str(
        label_encoder.inverse_transform(
            [encoded_prediction]
        )[0]
    )

    probabilities = classifier.predict_proba(
        classifier_input
    )[0]

    class_labels = label_encoder.inverse_transform(
        classifier.named_steps["classifier"].classes_.astype(int)
    )

    confidence = float(max(probabilities))

    class_probabilities = {
        str(label): round(float(probability), 3)
        for label, probability in zip(
            class_labels, probabilities
        )
    }

    # ----------------------------------------------
    # B. ANOMALY DETECTION
    # ----------------------------------------------
    anomaly_input = row[anomaly_features]

    transformed = anomaly_preprocessor.transform(
        anomaly_input
    )

    if_result = int(
        isolation_forest.predict(transformed)[0]
    )
    svm_result = int(
        one_class_svm.predict(transformed)[0]
    )

    isolation_anomaly = if_result == -1
    svm_anomaly = svm_result == -1

    # ----------------------------------------------
    # C. OPERATOR REVIEW
    # ----------------------------------------------
    # Review is always required in this demonstration.
    # Anomaly flags are alerts, not proof of a delay.
    return {
        "shipment_id": payload.get(
            "shipment_id", "unknown"
        ),
        "delay_status_prediction": predicted_label,
        "confidence": round(confidence, 3),
        "class_probabilities": class_probabilities,
        "anomaly_detection": {
            "isolation_forest_anomaly": isolation_anomaly,
            "one_class_svm_anomaly": svm_anomaly,
            "any_anomaly_detected": (
                isolation_anomaly or svm_anomaly
            ),
        },
        "requires_operator_review": True,
        "note": (
            "Model predictions support operational review; "
            "they do not automatically authorize rerouting."
        ),
    }


# --------------------------------------------------
# 3. TEST WITH DEMONSTRATION DATA
# --------------------------------------------------
example = {
    "shipment_id": "SHP-DEMO-100",
    "dwell_time_hours": 28,
    "throughput_teu_per_hr": 12,
    "capacity_utilization_pct": 94,
    "customs_hold_hours": 8,
    "weather_severity_index": 7,
    "labor_deficit_ratio": 0.3,
    "equipment_availability_pct": 58,
    "historical_delay_rate": 0.7,
    "priority_tier": "HIGH",
    "origin_port": "PORT_A",
    "destination_port": "PORT_B",
    "carrier_code": "C1",
}

result = predict_delay_risk(example)
print(result)

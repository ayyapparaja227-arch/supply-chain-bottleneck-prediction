#!/usr/bin/env python3
"""
serve_api.py
Low-Latency Bottleneck Scoring API for Port Control Tower Operations.
Exposes real-time endpoint for container and shipment telemetry triage.
"""

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Global Logistics Bottleneck Scoring Engine API",
    description="Low-latency real-time triage for port-to-warehouse logistics telemetry",
    version="1.0.0"
)

# Load trained models
MODEL_PATH = "/Users/ayyapparaja/pahse 2/models/xgboost_bottleneck_classifier.joblib"
ISO_PATH = "/Users/ayyapparaja/pahse 2/models/isolation_forest_anomaly.joblib"

try:
    classifier = joblib.load(MODEL_PATH)
    anomaly_detector = joblib.load(ISO_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Warning: Models could not be loaded immediately: {e}")
    classifier = None
    anomaly_detector = None

STATUS_MAPPING = {0: "ON_TIME", 1: "AT_RISK", 2: "CRITICAL_DELAY"}

class ShipmentTelemetryRequest(BaseModel):
    shipment_id: str = Field(..., example="SHP-20260042")
    container_id: str = Field(..., example="MSKU9841203")
    carrier_code: str = Field(..., example="MSK")
    current_node_id: str = Field(..., example="PORT_US_LAX")
    dwell_time_hours: float = Field(..., example=28.5)
    throughput_teu_per_hr: float = Field(..., example=195.0)
    capacity_utilization_pct: float = Field(..., example=82.4)
    customs_hold_hours: float = Field(..., example=4.5)
    weather_severity_index: float = Field(default=1.0, example=1.2)
    labor_deficit_ratio: float = Field(default=0.02, example=0.035)
    equipment_availability_pct: float = Field(default=92.0, example=88.0)
    historical_delay_rate: float = Field(default=0.15, example=0.18)

class BottleneckPredictionResponse(BaseModel):
    shipment_id: str
    predicted_delay_status: str
    confidence_score: float
    is_statistical_anomaly: bool
    risk_probabilities: dict
    recommended_tactical_action: str

@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "engine": "Stage-01 Bottleneck Scorer", "models_loaded": classifier is not None}

@app.post("/predict_bottleneck", response_model=BottleneckPredictionResponse)
def score_shipment(payload: ShipmentTelemetryRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Sanitize and clamp inputs
    dwell = max(0.0, payload.dwell_time_hours)
    capacity = min(100.0, max(0.0, payload.capacity_utilization_pct))
    
    # Feature engineering
    congestion_pressure_idx = (dwell * capacity) / 100.0
    operational_friction_idx = payload.labor_deficit_ratio * (100.0 - payload.equipment_availability_pct)

    feature_cols = [
        'dwell_time_hours', 'throughput_teu_per_hr', 'capacity_utilization_pct',
        'customs_hold_hours', 'weather_severity_index', 'labor_deficit_ratio',
        'equipment_availability_pct', 'historical_delay_rate',
        'congestion_pressure_idx', 'operational_friction_idx'
    ]
    feature_df = pd.DataFrame([[
        dwell,
        payload.throughput_teu_per_hr,
        capacity,
        payload.customs_hold_hours,
        payload.weather_severity_index,
        payload.labor_deficit_ratio,
        payload.equipment_availability_pct,
        payload.historical_delay_rate,
        congestion_pressure_idx,
        operational_friction_idx
    ]], columns=feature_cols)

    # 1. Anomaly check
    iso_cols = ['dwell_time_hours', 'throughput_teu_per_hr', 'capacity_utilization_pct']
    iso_df = pd.DataFrame([[dwell, payload.throughput_teu_per_hr, capacity]], columns=iso_cols)
    anomaly_pred = anomaly_detector.predict(iso_df)[0]
    is_anomaly = bool(anomaly_pred == -1)

    # 2. Classifier scoring
    probs = classifier.predict_proba(feature_df)[0]
    pred_class_idx = int(np.argmax(probs))
    pred_status = STATUS_MAPPING[pred_class_idx]
    confidence = float(probs[pred_class_idx])

    # Tactical recommendations
    if pred_status == "CRITICAL_DELAY":
        action = "TRIGGER_DIVERSION: Reroute to secondary inland rail ramp or reserve priority yard berth."
    elif pred_status == "AT_RISK":
        action = "DISPATCH_WARNING: Buffer yard pickup window and alert drayage carrier."
    else:
        action = "STANDARD_DISPATCH: Maintain baseline schedule."

    return BottleneckPredictionResponse(
        shipment_id=payload.shipment_id,
        predicted_delay_status=pred_status,
        confidence_score=round(confidence, 4),
        is_statistical_anomaly=is_anomaly,
        risk_probabilities={
            "ON_TIME": round(float(probs[0]), 4),
            "AT_RISK": round(float(probs[1]), 4),
            "CRITICAL_DELAY": round(float(probs[2]), 4)
        },
        recommended_tactical_action=action
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

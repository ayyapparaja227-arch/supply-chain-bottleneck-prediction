#!/usr/bin/env python3
"""
Global Supply Chain Logistics Bottleneck Predictor
Claymorphism Design System Edition
Streamlit Operations Console (Stage 01)
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CLAYMORPHISM DESIGN SYSTEM
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Bottleneck Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Claymorphism Authoritative Tokens & CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --dp-color-ground: #f1f2fa;
        --dp-color-ink: #383139;
        --dp-color-ink-secondary: #48404b;
        --dp-color-accent: #f6a3ca;
        --dp-color-surface-support: #c0efdc;
        --dp-color-surface-compare: #ded0f9;
        --dp-color-surface-warm: #f9e7a8;
        --dp-color-surface-cool: #c5e5fb;
        --dp-color-surface: #fffdf7;
        --dp-typography-face: 'Instrument Sans', sans-serif;
    }

    /* Page Background */
    .stApp {
        background-color: var(--dp-color-ground) !important;
        color: var(--dp-color-ink) !important;
        font-family: var(--dp-typography-face) !important;
    }

    /* Global Typography Resets */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        font-family: var(--dp-typography-face) !important;
        color: var(--dp-color-ink);
    }

    /* Claymorphism Hero Pink Cushion */
    .clay-hero {
        background-color: #f6a3ca;
        border-radius: 38px;
        padding: 30px 36px;
        color: #383139;
        margin-bottom: 24px;
        box-shadow: 10px 10px 24px rgba(188, 155, 175, 0.45),
                    -10px -10px 24px rgba(255, 255, 255, 0.95),
                    inset 4px 4px 8px rgba(255, 255, 255, 0.7),
                    inset -4px -4px 8px rgba(190, 110, 150, 0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .clay-hero h1 {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        color: #383139 !important;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .clay-hero p {
        font-size: 14.5px;
        color: #48404b !important;
        margin: 8px 0 0 0;
        line-height: 1.45;
        max-width: 650px;
    }
    .clay-tag {
        display: inline-block;
        background-color: #fffdf7;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #383139;
        box-shadow: 4px 4px 10px rgba(188, 155, 175, 0.35),
                    -4px -4px 10px rgba(255, 255, 255, 0.8),
                    inset 2px 2px 4px rgba(255, 255, 255, 0.8),
                    inset -2px -2px 4px rgba(188, 155, 175, 0.2);
        margin-bottom: 8px;
    }

    /* Clay Pastel Fact Cushions */
    .clay-cushion {
        border-radius: 28px;
        padding: 18px 22px;
        box-shadow: 8px 8px 18px rgba(175, 180, 200, 0.4),
                    -8px -8px 18px rgba(255, 255, 255, 0.95),
                    inset 3px 3px 6px rgba(255, 255, 255, 0.75),
                    inset -3px -3px 6px rgba(170, 175, 195, 0.25);
        margin-bottom: 18px;
        transition: transform 0.18s ease;
    }
    .clay-cushion:hover {
        transform: translateY(-2px);
    }
    .clay-cushion.mint { background-color: #c0efdc; }
    .clay-cushion.cool { background-color: #c5e5fb; }
    .clay-cushion.lavender { background-color: #ded0f9; }
    .clay-cushion.warm { background-color: #f9e7a8; }
    .clay-cushion.cream { background-color: #fffdf7; }
    .clay-cushion.pink { background-color: #f6a3ca; }

    .cushion-label {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #48404b;
    }
    .cushion-val {
        font-size: 21px;
        font-weight: 800;
        color: #383139;
        margin-top: 4px;
        letter-spacing: -0.01em;
    }

    /* Shared Cream Tray Container */
    .clay-tray {
        background-color: #fffdf7;
        border-radius: 38px;
        padding: 28px 32px;
        box-shadow: 12px 12px 28px rgba(180, 185, 205, 0.35),
                    -12px -12px 28px rgba(255, 255, 255, 0.95),
                    inset 3px 3px 6px rgba(255, 255, 255, 0.9),
                    inset -3px -3px 6px rgba(180, 185, 205, 0.2);
        margin-bottom: 24px;
    }

    /* Prediction Result Pill Banners */
    .prediction-clay-ontime {
        background-color: #c0efdc;
        border-radius: 30px;
        padding: 22px 26px;
        color: #244635;
        box-shadow: 8px 8px 18px rgba(145, 195, 175, 0.45),
                    -8px -8px 18px rgba(255, 255, 255, 0.95),
                    inset 3px 3px 6px rgba(255, 255, 255, 0.8),
                    inset -3px -3px 6px rgba(120, 175, 150, 0.3);
        margin-bottom: 18px;
    }
    .prediction-clay-atrisk {
        background-color: #f9e7a8;
        border-radius: 30px;
        padding: 22px 26px;
        color: #4a3c10;
        box-shadow: 8px 8px 18px rgba(210, 195, 135, 0.45),
                    -8px -8px 18px rgba(255, 255, 255, 0.95),
                    inset 3px 3px 6px rgba(255, 255, 255, 0.8),
                    inset -3px -3px 6px rgba(195, 170, 100, 0.3);
        margin-bottom: 18px;
    }
    .prediction-clay-critical {
        background-color: #f6a3ca;
        border-radius: 30px;
        padding: 22px 26px;
        color: #451c2d;
        box-shadow: 8px 8px 18px rgba(205, 140, 170, 0.45),
                    -8px -8px 18px rgba(255, 255, 255, 0.95),
                    inset 3px 3px 6px rgba(255, 255, 255, 0.8),
                    inset -3px -3px 6px rgba(190, 110, 140, 0.3);
        margin-bottom: 18px;
    }

    .pred-badge-status {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .pred-badge-meta {
        font-size: 13.5px;
        font-weight: 600;
        margin-top: 4px;
        opacity: 0.95;
    }

    /* Tactical Action Pillow */
    .tactical-pillow {
        border-radius: 26px;
        padding: 18px 22px;
        color: #383139;
        box-shadow: 6px 6px 14px rgba(185, 170, 215, 0.35),
                    -6px -6px 14px rgba(255, 255, 255, 0.9),
                    inset 3px 3px 6px rgba(255, 255, 255, 0.75),
                    inset -3px -3px 6px rgba(175, 160, 205, 0.25);
        margin-bottom: 18px;
    }
    .tactical-title {
        font-size: 11.5px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .tactical-body {
        font-size: 13.5px;
        line-height: 1.5;
        font-weight: 500;
    }

    /* Clay Corridor Stepper */
    .clay-stepper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #fffdf7;
        border-radius: 34px;
        padding: 16px 24px;
        margin-bottom: 22px;
        box-shadow: 8px 8px 20px rgba(180, 185, 205, 0.3),
                    -8px -8px 20px rgba(255, 255, 255, 0.9),
                    inset 2px 2px 5px rgba(255, 255, 255, 0.8),
                    inset -2px -2px 5px rgba(180, 185, 205, 0.2);
    }
    .stepper-node {
        text-align: center;
        flex: 1;
    }
    .stepper-pill {
        width: 34px;
        height: 34px;
        border-radius: 999px;
        background-color: #c5e5fb;
        color: #204b6b;
        font-weight: 800;
        font-size: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 6px auto;
        box-shadow: 3px 3px 8px rgba(150, 185, 215, 0.45),
                    -3px -3px 8px rgba(255, 255, 255, 0.9),
                    inset 2px 2px 3px rgba(255, 255, 255, 0.8),
                    inset -2px -2px 3px rgba(150, 185, 215, 0.3);
    }
    .stepper-pill.alert {
        background-color: #f6a3ca;
        color: #451c2d;
    }
    .stepper-pill.active {
        background-color: #c0efdc;
        color: #1b4231;
    }
    .stepper-label {
        font-size: 11.5px;
        font-weight: 700;
        color: #48404b;
    }
    .stepper-bar {
        flex: 1;
        height: 5px;
        background-color: #ded0f9;
        border-radius: 999px;
        margin: 0 10px 16px 10px;
    }

    /* Clay Table Styling for Drivers */
    .table-tray {
        background-color: #fffdf7;
        border-radius: 28px;
        padding: 16px 20px;
        box-shadow: 6px 6px 16px rgba(180, 185, 205, 0.3),
                    -6px -6px 16px rgba(255, 255, 255, 0.9),
                    inset 2px 2px 4px rgba(255, 255, 255, 0.8),
                    inset -2px -2px 4px rgba(180, 185, 205, 0.15);
        margin-top: 10px;
    }
    .clay-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
    }
    .clay-table th {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #48404b;
        padding: 6px 12px;
        text-align: left;
    }
    .clay-table tr td {
        background-color: #f1f2fa;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 600;
        color: #383139;
        box-shadow: 3px 3px 8px rgba(185, 190, 210, 0.25),
                    -3px -3px 8px rgba(255, 255, 255, 0.8),
                    inset 1px 1px 3px rgba(255, 255, 255, 0.8),
                    inset -1px -1px 3px rgba(185, 190, 210, 0.15);
    }
    .clay-table tr td:first-child {
        border-radius: 14px 0 0 14px;
    }
    .clay-table tr td:last-child {
        border-radius: 0 14px 14px 0;
    }

    /* Fix Dropdown Popovers (Never Black) */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[data-baseweb="popover"] > div {
        background-color: #fffdf7 !important;
        border-radius: 20px !important;
        box-shadow: 8px 8px 24px rgba(175, 180, 205, 0.45),
                    -8px -8px 24px rgba(255, 255, 255, 0.95),
                    inset 2px 2px 4px rgba(255, 255, 255, 0.8) !important;
        border: 1px solid #ded0f9 !important;
        padding: 6px !important;
    }
    li[role="option"] {
        background-color: #fffdf7 !important;
        color: #383139 !important;
        border-radius: 12px !important;
        font-family: 'Instrument Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 10px 14px !important;
        transition: all 0.15s ease !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #ded0f9 !important;
        color: #383139 !important;
    }

    /* Sidebar Overrides for Absolute Clarity & Legibility */
    section[data-testid="stSidebar"] {
        background-color: #e9ebf6 !important;
        border-right: none !important;
        box-shadow: 6px 0 20px rgba(165, 172, 195, 0.25) !important;
    }
    section[data-testid="stSidebar"] label {
        color: #383139 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: 0.02em !important;
    }
    
    /* Selectboxes and Inputs */
    div[data-baseweb="select"] > div {
        background-color: #fffdf7 !important;
        border-radius: 999px !important;
        border: none !important;
        box-shadow: 4px 4px 10px rgba(175, 180, 200, 0.35),
                    -4px -4px 10px rgba(255, 255, 255, 0.85),
                    inset 2px 2px 3px rgba(255, 255, 255, 0.8),
                    inset -2px -2px 3px rgba(175, 180, 200, 0.2) !important;
        color: #383139 !important;
        font-weight: 600 !important;
    }
    
    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #fffdf7;
        border-radius: 999px;
        border: none;
        padding: 8px 24px;
        font-size: 13.5px;
        font-weight: 700;
        color: #48404b;
        box-shadow: 5px 5px 12px rgba(175, 180, 205, 0.35),
                    -5px -5px 12px rgba(255, 255, 255, 0.9),
                    inset 2px 2px 3px rgba(255, 255, 255, 0.8),
                    inset -2px -2px 3px rgba(175, 180, 205, 0.2);
        transition: all 0.18s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f6a3ca !important;
        color: #383139 !important;
        box-shadow: 5px 5px 14px rgba(205, 140, 170, 0.5),
                    -5px -5px 14px rgba(255, 255, 255, 0.95),
                    inset 2px 2px 4px rgba(255, 255, 255, 0.85),
                    inset -2px -2px 4px rgba(190, 110, 140, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. MODEL LOADING (CLEAN & DIRECT)
# ---------------------------------------------------------------------
@st.cache_resource
def load_logistics_models():
    search_paths = [
        Path("data"),
        Path("Stage01_ML/data"),
        Path("../data"),
        Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage01_ML/data"),
        Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/data"),
        Path("/Users/ayyapparaja/pahse 2/data")
    ]
    
    classifier, label_encoder, isolation_forest, anomaly_preprocessor = None, None, None, None
    for p in search_paths:
        if (p / "xgboost_delay_model.pkl").exists():
            try:
                classifier = joblib.load(p / "xgboost_delay_model.pkl")
                label_encoder = joblib.load(p / "xgboost_label_encoder.pkl")
                isolation_forest = joblib.load(p / "isolation_forest_model.pkl")
                anomaly_preprocessor = joblib.load(p / "anomaly_preprocessor.pkl")
                return classifier, label_encoder, isolation_forest, anomaly_preprocessor, p
            except Exception:
                continue
    return None, None, None, None, None

classifier, label_encoder, isolation_forest, anomaly_preprocessor, loaded_path = load_logistics_models()

# ---------------------------------------------------------------------
# 3. CLAYMORPHISM OPENING HERO (PINK CUSHION)
# ---------------------------------------------------------------------
st.markdown("""
<div class="clay-hero">
    <div>
        <div class="clay-tag">🚢 Port-to-Warehouse Congestion Triage</div>
        <h1>Global Supply Chain Bottleneck Predictor</h1>
        <p>Tactile intelligence dispatch engine analyzing dwell waveforms, yard saturation, and customs turbulence to neutralize logistics bottlenecks before regional supply chains collapse.</p>
    </div>
    <div style="background-color: #fffdf7; border-radius: 999px; padding: 12px 24px; font-weight: 800; font-size: 13px; color: #383139; box-shadow: 4px 4px 10px rgba(190,140,165,0.4), -4px -4px 10px rgba(255,255,255,0.8), inset 2px 2px 4px rgba(255,255,255,0.8), inset -2px -2px 4px rgba(190,140,165,0.2);">
        ● Telemetry Online
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 4. SIDEBAR (CLAY PILLED INPUTS)
# ---------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Operational Scenarios")

scenarios = {
    "⚡ Select Disruption Scenario...": None,
    "🟢 Normal Flow (Shanghai ➔ LA)": {
        "dwell": 18.0, "throughput": 245.0, "capacity": 65.0, "customs": 1.5,
        "weather": 0.8, "labor": 0.015, "carrier": "MSK", "priority": "STANDARD",
        "origin": "PORT_CN_SHA", "dest": "PORT_US_LAX"
    },
    "🟡 Customs Backlog Shock (+14h Hold)": {
        "dwell": 33.5, "throughput": 185.0, "capacity": 82.5, "customs": 14.0,
        "weather": 1.4, "labor": 0.035, "carrier": "HAPAG", "priority": "EXPEDITED",
        "origin": "PORT_CN_SHA", "dest": "PORT_US_OAK"
    },
    "🔴 Severe Pacific Typhoon (Gale Lockout)": {
        "dwell": 38.0, "throughput": 110.0, "capacity": 89.0, "customs": 5.0,
        "weather": 5.2, "labor": 0.08, "carrier": "ONE", "priority": "CRITICAL_MEDICAL",
        "origin": "PORT_SGP_SIN", "dest": "PORT_US_LAX"
    },
    "⚠️ Yard Saturation Crisis (>100% Overflow)": {
        "dwell": 45.0, "throughput": 90.0, "capacity": 108.0, "customs": 8.5,
        "weather": 2.5, "labor": 0.14, "carrier": "COSCO", "priority": "STANDARD",
        "origin": "PORT_CN_SHA", "dest": "PORT_US_LGB"
    }
}

selected_preset_name = st.sidebar.selectbox("Load Disruption Preset:", list(scenarios.keys()))
preset = scenarios[selected_preset_name] if selected_preset_name != "⚡ Select Disruption Scenario..." else None

# Pre-populated values
val_dwell = preset["dwell"] if preset else 22.0
val_tp = preset["throughput"] if preset else 230.0
val_cap = preset["capacity"] if preset else 72.0
val_cust = preset["customs"] if preset else 2.5
val_weath = preset["weather"] if preset else 1.2
val_labor = preset["labor"] if preset else 0.02
val_carrier = preset["carrier"] if preset else "MSK"
val_priority = preset["priority"] if preset else "STANDARD"
val_origin = preset["origin"] if preset else "PORT_CN_SHA"
val_dest = preset["dest"] if preset else "PORT_US_LAX"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Route & Carrier Parameters")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    origin_list = ["PORT_CN_SHA", "PORT_SGP_SIN"]
    origin = st.selectbox("Origin Port", origin_list, index=origin_list.index(val_origin) if val_origin in origin_list else 0)
    
    carrier_list = ["MSK", "MSC", "HAPAG", "ONE", "COSCO", "CMA"]
    carrier = st.selectbox("Carrier", carrier_list, index=carrier_list.index(val_carrier) if val_carrier in carrier_list else 0)

with col_sb2:
    dest_list = ["PORT_US_LAX", "PORT_US_LGB", "PORT_US_OAK"]
    destination = st.selectbox("Destination Port", dest_list, index=dest_list.index(val_dest) if val_dest in dest_list else 0)
    
    priority_list = ["STANDARD", "EXPEDITED", "CRITICAL_MEDICAL"]
    priority = st.selectbox("Priority Tier", priority_list, index=priority_list.index(val_priority) if val_priority in priority_list else 0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Port & Yard Telemetry Sliders")

dwell = st.sidebar.slider("Dwell Time (Hours)", 4.0, 95.0, float(val_dwell), 0.5)
capacity = st.sidebar.slider("Yard Capacity (%)", 30.0, 120.0, float(val_cap), 1.0)
throughput = st.sidebar.slider("Berth Throughput (TEU/hr)", 60.0, 430.0, float(val_tp), 5.0)
customs = st.sidebar.slider("Customs Hold (Hours)", 0.0, 48.0, float(val_cust), 0.5)
weather = st.sidebar.slider("Weather Severity (0-10)", 0.0, 10.0, float(val_weath), 0.1)
labor = st.sidebar.slider("Labor Deficit Ratio", 0.0, 0.5, float(val_labor), 0.005)

# ---------------------------------------------------------------------
# 5. PREDICTION COMPUTATION
# ---------------------------------------------------------------------
if classifier is None:
    st.error("⚠️ Model artifacts not found. Please verify xgboost_delay_model.pkl exists.")
    st.stop()

# Prepare payload DataFrame
classifier_features = [
    "dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct",
    "customs_hold_hours", "weather_severity_index", "labor_deficit_ratio",
    "equipment_availability_pct", "historical_delay_rate",
    "priority_tier", "origin_port", "destination_port", "carrier_code"
]

row_dict = {
    "dwell_time_hours": dwell,
    "throughput_teu_per_hr": throughput,
    "capacity_utilization_pct": capacity,
    "customs_hold_hours": customs,
    "weather_severity_index": weather,
    "labor_deficit_ratio": labor,
    "equipment_availability_pct": 92.0,
    "historical_delay_rate": 0.15,
    "priority_tier": priority,
    "origin_port": origin,
    "destination_port": destination,
    "carrier_code": carrier
}
input_df = pd.DataFrame([row_dict])[classifier_features]

# Execute delay prediction
pred_code = int(classifier.predict(input_df)[0])
probabilities = classifier.predict_proba(input_df)[0]
class_labels = list(label_encoder.classes_)
pred_label = str(label_encoder.inverse_transform([pred_code])[0])

prob_dict = {str(lbl): float(pr) for lbl, pr in zip(class_labels, probabilities)}
p_on = prob_dict.get("ON_TIME", 0.0)
p_risk = prob_dict.get("AT_RISK", 0.0)
p_crit = prob_dict.get("CRITICAL_DELAY", 0.0)

confidence_pct = round(float(max(probabilities)) * 100, 1)

# Sensor Anomaly check
anomaly_features = [
    "dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct",
    "customs_hold_hours", "weather_severity_index", "labor_deficit_ratio",
    "equipment_availability_pct", "historical_delay_rate"
]
anomaly_input = pd.DataFrame([row_dict])[anomaly_features]
scaled_anomaly = anomaly_preprocessor.transform(anomaly_input)
is_anomaly = bool(isolation_forest.predict(scaled_anomaly)[0] == -1)

# Composite Delay Risk Score (0 - 100%)
risk_score = round(p_risk * 50.0 + p_crit * 100.0, 1)

# ---------------------------------------------------------------------
# 6. PASTEL FACT CUSHIONS ROW
# ---------------------------------------------------------------------
origin_short = origin.replace("PORT_", "")
dest_short = destination.replace("PORT_", "")

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.markdown(f'''
    <div class="clay-cushion cool">
        <div class="cushion-label">Active Corridor</div>
        <div class="cushion-val" style="font-size:18px;">{origin_short} ➔ {dest_short}</div>
    </div>
    ''', unsafe_allow_html=True)

with f2:
    st.markdown(f'''
    <div class="clay-cushion lavender">
        <div class="cushion-label">Carrier & Priority</div>
        <div class="cushion-val" style="font-size:18px;">{carrier} • {priority[:4]}</div>
    </div>
    ''', unsafe_allow_html=True)

with f3:
    status_cushion = "mint" if not is_anomaly else "pink"
    status_text = "Nominal Flow" if not is_anomaly else "Irregular Sensor"
    st.markdown(f'''
    <div class="clay-cushion {status_cushion}">
        <div class="cushion-label">Sensor Telemetry</div>
        <div class="cushion-val" style="font-size:18px;">{status_text}</div>
    </div>
    ''', unsafe_allow_html=True)

with f4:
    r_cushion = "mint" if risk_score < 35 else ("warm" if risk_score < 70 else "pink")
    st.markdown(f'''
    <div class="clay-cushion {r_cushion}">
        <div class="cushion-label">Bottleneck Risk Index</div>
        <div class="cushion-val">{risk_score}%</div>
    </div>
    ''', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 7. CLAY ROUTE STEPPER (INFLATED TRANSIT PIPELINE)
# ---------------------------------------------------------------------
c_dwell_alert = "alert" if dwell > 32 else ("active" if dwell > 20 else "")
c_yard_alert = "alert" if capacity > 90 else ("active" if capacity > 75 else "")
c_gate_alert = "alert" if pred_label == "CRITICAL_DELAY" else ("active" if pred_label == "AT_RISK" else "")

st.markdown(f"""
<div class="clay-stepper">
    <div class="stepper-node">
        <div class="stepper-pill active">1</div>
        <div class="stepper-label">{origin_short}<br><span style="font-size:11px; opacity:0.8;">Origin Berth</span></div>
    </div>
    <div class="stepper-bar"></div>
    <div class="stepper-node">
        <div class="stepper-pill active">2</div>
        <div class="stepper-label">Ocean<br><span style="font-size:11px; opacity:0.8;">Transpacific</span></div>
    </div>
    <div class="stepper-bar"></div>
    <div class="stepper-node">
        <div class="stepper-pill {c_dwell_alert if c_dwell_alert else 'active'}">3</div>
        <div class="stepper-label">Customs<br><span style="font-size:11px; opacity:0.8;">{customs:.1f}h hold</span></div>
    </div>
    <div class="stepper-bar"></div>
    <div class="stepper-node">
        <div class="stepper-pill {c_yard_alert if c_yard_alert else 'active'}">4</div>
        <div class="stepper-label">Yard Staging<br><span style="font-size:11px; opacity:0.8;">{capacity:.0f}% capacity</span></div>
    </div>
    <div class="stepper-bar"></div>
    <div class="stepper-node">
        <div class="stepper-pill {c_gate_alert if c_gate_alert else 'active'}">5</div>
        <div class="stepper-label">{dest_short}<br><span style="font-size:11px; opacity:0.8;">Gate Clearance</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 8. SHARED CREAM TRAY & TABS
# ---------------------------------------------------------------------
tab_triage, tab_visuals = st.tabs(["🎯 Live Bottleneck Prediction", "📊 Visual Telemetry & What-If Simulation"])

with tab_triage:
    st.markdown('<div class="clay-tray">', unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns([1.15, 1.0], gap="large")
    
    with col_t1:
        # Prediction Pill Banner
        if pred_label == "ON_TIME":
            st.markdown(f"""
            <div class="prediction-clay-ontime">
                <div class="cushion-label" style="color:#204b36;">Shipment Velocity Forecast</div>
                <div class="pred-badge-status">🟢 ON TIME</div>
                <div class="pred-badge-meta">Confidence: <strong>{confidence_pct}%</strong> • Predicts standard multimodal gate clearance without bottleneck friction.</div>
            </div>
            """, unsafe_allow_html=True)
        elif pred_label == "AT_RISK":
            st.markdown(f"""
            <div class="prediction-clay-atrisk">
                <div class="cushion-label" style="color:#524213;">Shipment Velocity Forecast</div>
                <div class="pred-badge-status">🟡 AT RISK (Moderate Delay)</div>
                <div class="pred-badge-meta">Confidence: <strong>{confidence_pct}%</strong> • Moderate corridor dwell friction (+12h to +24h schedule slippage).</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-clay-critical">
                <div class="cushion-label" style="color:#4d1b31;">Shipment Velocity Forecast</div>
                <div class="pred-badge-status">🔴 CRITICAL DELAY</div>
                <div class="pred-badge-meta">Confidence: <strong>{confidence_pct}%</strong> • Severe terminal bottleneck detected (&gt;48h compounding delay).</div>
            </div>
            """, unsafe_allow_html=True)

        # Tactical Action Directive
        if pred_label == "CRITICAL_DELAY":
            st.markdown("""
            <div class="tactical-pillow" style="background-color: #f6a3ca;">
                <div class="tactical-title" style="color:#451c2d;">⚡ Tactical Action Directive // Priority Rail Diversion</div>
                <div class="tactical-body">
                    <strong>DISPATCH OFF-DOCK SHUNT:</strong> Immediate emergency diversion of container to Union Pacific/BNSF inland intermodal rail ramp. Avoid congested local highway gate to prevent cascading plant shutdown.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif pred_label == "AT_RISK":
            st.markdown("""
            <div class="tactical-pillow" style="background-color: #f9e7a8;">
                <div class="tactical-title" style="color:#4a3a0e;">⚠️ Tactical Action Directive // Buffer Extension</div>
                <div class="tactical-body">
                    <strong>GRANT APPOINTMENT EXTENSION:</strong> Extend distribution warehouse intake SLA by +18 hours. Direct drayage fleet to night-gate operations while berth throughput recovers.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tactical-pillow" style="background-color: #c0efdc;">
                <div class="tactical-title" style="color:#1b4231;">✅ Tactical Action Directive // Standard Transit Flow</div>
                <div class="tactical-body">
                    <strong>MAINTAIN SCHEDULED PASS:</strong> Gate turn times and dwell waves fall well within service standards. Continue scheduled inland trucking transfers.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Telemetry Risk Drivers Table (Built cleanly without indented multiline whitespace)
        st.markdown("<h4 style='font-size:16px; font-weight:800; color:#383139; margin-top:14px; margin-bottom:4px;'>🔍 Key Telemetry Drivers</h4>", unsafe_allow_html=True)
        drivers = [
            ("Container Dwell Time", f"{dwell:.1f} hrs", "< 24 hrs", "🔴 Critical" if dwell > 35 else ("🟡 Elevated" if dwell > 24 else "🟢 Normal")),
            ("Yard Saturation", f"{capacity:.1f}%", "< 80%", "🔴 Overflow" if capacity > 100 else ("🟡 High" if capacity > 80 else "🟢 Optimal")),
            ("Customs Hold", f"{customs:.1f} hrs", "< 4 hrs", "🔴 Backlogged" if customs > 8 else ("🟡 Queued" if customs > 4 else "🟢 Cleared")),
            ("Berth Crane Velocity", f"{throughput:.0f} TEU/h", "> 180 TEU/h", "🔴 Choked" if throughput < 120 else ("🟡 Moderate" if throughput < 180 else "🟢 Swift")),
            ("Weather Severity", f"{weather:.1f} / 10", "< 3.0", "🔴 Severe Gale" if weather > 4.5 else ("🟡 Adverse" if weather > 2.5 else "🟢 Calm"))
        ]
        
        table_rows = []
        for factor, obs, bench, status in drivers:
            bench_clean = bench.replace("<", "&lt;").replace(">", "&gt;")
            table_rows.append(f'<tr><td><strong>{factor}</strong></td><td>{obs}</td><td style="color:#48404b;">{bench_clean}</td><td>{status}</td></tr>')
        
        table_html = f'<div class="table-tray"><table class="clay-table"><thead><tr><th>Telemetry Channel</th><th>Observed</th><th>Safe Range</th><th>Status</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>'
        st.markdown(table_html, unsafe_allow_html=True)

    with col_t2:
        # VISUALIZATION 1: Claymorphic Bottleneck Risk Gauge
        st.markdown("<h4 style='font-size:16px; font-weight:800; color:#383139; margin-bottom:2px;'>⏱️ Bottleneck Risk Gauge</h4>", unsafe_allow_html=True)
        
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(risk_score),
            number={'suffix': "%", 'font': {'size': 38, 'family': 'Instrument Sans', 'color': '#383139'}},
            title={'text': "Composite Delay Risk", 'font': {'size': 13, 'color': '#48404b', 'family': 'Instrument Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': "#48404b", 'tickfont': {'family': 'Instrument Sans', 'size': 11, 'color': '#48404b'}},
                'bar': {'color': "#383139", 'thickness': 0.28},
                'bgcolor': "#fffdf7",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': '#c0efdc'},   # Mint
                    {'range': [35, 70], 'color': '#f9e7a8'},  # Butter Yellow
                    {'range': [70, 100], 'color': '#f6a3ca'}  # Pastel Pink
                ],
                'threshold': {
                    'line': {'color': "#383139", 'width': 3.5},
                    'thickness': 0.85,
                    'value': float(risk_score)
                }
            }
        ))
        gauge_fig.update_layout(
            height=215,
            margin=dict(l=25, r=25, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Instrument Sans", color="#383139")
        )
        st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})

        # VISUALIZATION 2: Claymorphic Probability Donut Chart
        st.markdown("<h4 style='font-size:16px; font-weight:800; color:#383139; margin-top:8px; margin-bottom:2px;'>📊 Calibrated Probability Share</h4>", unsafe_allow_html=True)
        
        donut_labels = ["On Time", "At Risk", "Critical Delay"]
        donut_values = [p_on * 100, p_risk * 100, p_crit * 100]
        donut_colors = ["#c0efdc", "#f9e7a8", "#f6a3ca"]

        donut_fig = go.Figure(data=[go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=0.62,
            marker=dict(colors=donut_colors, line=dict(color="#fffdf7", width=3)),
            textinfo='percent',
            hoverinfo='label+percent',
            textfont=dict(size=13, family="Instrument Sans", color="#383139")
        )])
        donut_fig.update_layout(
            height=215,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=11, color="#383139", family="Instrument Sans")),
            annotations=[dict(text=f"{pred_label.replace('_', ' ')}<br><span style='font-size:20px; font-weight:800; color:#383139;'>{confidence_pct}%</span>", x=0.5, y=0.5, font_size=12, font_family="Instrument Sans", showarrow=False)]
        )
        st.plotly_chart(donut_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)

with tab_visuals:
    st.markdown('<div class="clay-tray">', unsafe_allow_html=True)
    
    col_v1, col_v2 = st.columns(2, gap="large")
    
    with col_v1:
        # VISUALIZATION 3: 6-Axis Telemetry Health Radar (Clean Top Legend & No Modebar)
        st.markdown("<h4 style='font-size:16px; font-weight:800; color:#383139;'>🕸️ 6-Axis Telemetry Health Footprint</h4>", unsafe_allow_html=True)
        st.caption("Molded polygon comparison showing current shipment metrics against optimal baseline envelopes.")
        
        radar_categories = [
            'Dwell Time', 'Yard Saturation', 'Customs Duration',
            'Berth Throughput', 'Weather Severity', 'Labor Deficit'
        ]
        
        norm_dwell = min(100.0, (dwell / 48.0) * 100.0)
        norm_cap = min(100.0, (capacity / 110.0) * 100.0)
        norm_cust = min(100.0, (customs / 24.0) * 100.0)
        norm_tp_def = min(100.0, max(0.0, (1.0 - throughput / 350.0) * 100.0))
        norm_weath = min(100.0, (weather / 6.0) * 100.0)
        norm_labor = min(100.0, (labor / 0.20) * 100.0)
        
        current_footprint = [norm_dwell, norm_cap, norm_cust, norm_tp_def, norm_weath, norm_labor]
        baseline_envelope = [40, 60, 20, 30, 20, 15]

        radar_fig = go.Figure()
        
        # Baseline Envelope (Lavender Soft Area)
        radar_fig.add_trace(go.Scatterpolar(
            r=baseline_envelope + [baseline_envelope[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor='rgba(222, 208, 249, 0.45)',
            line=dict(color='#b59fe0', dash='dash', width=2),
            name='Optimal Baseline'
        ))
        
        # Current Observation (Pink Molded Area)
        radar_fig.add_trace(go.Scatterpolar(
            r=current_footprint + [current_footprint[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor='rgba(246, 163, 202, 0.45)',
            line=dict(color='#d4659a', width=3),
            name='Current Shipment'
        ))

        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#48404b", gridcolor="#e2e8f0"),
                angularaxis=dict(color="#383139", gridcolor="#e2e8f0", tickfont=dict(family="Instrument Sans", size=11, color="#383139"))
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(family="Instrument Sans", size=11, color="#383139")
            ),
            height=340,
            margin=dict(l=35, r=35, t=35, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(radar_fig, use_container_width=True, config={'displayModeBar': False})

    with col_v2:
        # VISUALIZATION 4: What-If Sensitivity Simulator (Top Legend & Ample Bottom Axis Padding)
        st.markdown("<h4 style='font-size:16px; font-weight:800; color:#383139;'>📈 What-If Sensitivity Simulator</h4>", unsafe_allow_html=True)
        st.caption("Simulates delay probability shift as Container Dwell Time climbs from 5h to 65h.")
        
        dwell_range = np.linspace(5.0, 65.0, 25)
        sim_data = []
        
        for d_sim in dwell_range:
            sim_row = row_dict.copy()
            sim_row["dwell_time_hours"] = d_sim
            sim_df = pd.DataFrame([sim_row])[classifier_features]
            sim_probs = classifier.predict_proba(sim_df)[0]
            sim_prob_dict = {str(lbl): float(pr) for lbl, pr in zip(class_labels, sim_probs)}
            sim_data.append({
                "Dwell_Hours": d_sim,
                "p_ontime": sim_prob_dict.get("ON_TIME", 0.0) * 100.0,
                "p_atrisk": sim_prob_dict.get("AT_RISK", 0.0) * 100.0,
                "p_critical": sim_prob_dict.get("CRITICAL_DELAY", 0.0) * 100.0
            })
            
        curve_df = pd.DataFrame(sim_data)
        
        curve_fig = go.Figure()
        curve_fig.add_trace(go.Scatter(
            x=curve_df["Dwell_Hours"], y=curve_df["p_ontime"],
            mode='lines', name='On Time',
            line=dict(color='#38b284', width=3.5)
        ))
        curve_fig.add_trace(go.Scatter(
            x=curve_df["Dwell_Hours"], y=curve_df["p_atrisk"],
            mode='lines', name='At Risk',
            line=dict(color='#e5a823', width=3.5)
        ))
        curve_fig.add_trace(go.Scatter(
            x=curve_df["Dwell_Hours"], y=curve_df["p_critical"],
            mode='lines', name='Critical Delay',
            line=dict(color='#e2558f', width=3.5)
        ))
        
        # Vertical marker for current dwell time
        curve_fig.add_vline(
            x=dwell, line_width=2.5, line_dash="dash", line_color="#383139",
            annotation_text=f"Current: {dwell:.1f}h", annotation_position="top left",
            annotation_font=dict(color="#383139", size=11, family="Instrument Sans")
        )

        curve_fig.update_layout(
            xaxis_title="Dwell Time (Hours)",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 105], gridcolor="#f1f2fa", color="#48404b", title_font=dict(family="Instrument Sans", size=12)),
            xaxis=dict(gridcolor="#f1f2fa", color="#48404b", title_font=dict(family="Instrument Sans", size=12)),
            height=340,
            margin=dict(l=35, r=20, t=35, b=45),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Instrument Sans", color="#383139"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(family="Instrument Sans", size=11, color="#383139")
            )
        )
        st.plotly_chart(curve_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 9. CLOSING & FOOTER
# ---------------------------------------------------------------------
st.markdown("""
<div style="background-color: #fffdf7; border-radius: 30px; padding: 18px 26px; display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: 600; color: #48404b; box-shadow: 6px 6px 16px rgba(185, 190, 210, 0.3), -6px -6px 16px rgba(255, 255, 255, 0.9);">
    <div>Global Supply Chain Logistics Intelligence Console • Production Edition</div>
    <div>Live Corridor Telemetry Active</div>
</div>
""", unsafe_allow_html=True)

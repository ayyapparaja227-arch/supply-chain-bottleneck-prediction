#!/usr/bin/env python3
"""
Global Supply Chain Logistics Bottleneck Predictor
Modern White & Royal Blue UI/UX with Rich Plotly Visualizations
Streamlit Operations Console (Stage 01)
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING (COOL WHITE WITH BLUE)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Bottleneck Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Cool White & Blue UI/UX
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Crisp Light Background */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        box-shadow: 2px 0 10px rgba(15, 23, 42, 0.02);
    }
    
    /* Header Banner with Blue Gradient */
    .header-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #0284C7 100%);
        padding: 22px 28px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
    }
    .header-banner h1 {
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        color: #FFFFFF !important;
    }
    .header-banner p {
        font-size: 14px;
        color: #E0F2FE;
        margin: 5px 0 0 0;
        font-weight: 400;
    }
    .header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(37, 99, 235, 0.08);
        border-color: #BFDBFE;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-val {
        font-size: 22px;
        font-weight: 800;
        color: #1E3A8A;
        margin-top: 4px;
    }
    .kpi-val.green { color: #059669; }
    .kpi-val.amber { color: #D97706; }
    .kpi-val.red   { color: #DC2626; }
    
    /* Dynamic Prediction Status Banners */
    .prediction-card-ontime {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 20px 24px;
        color: #065F46;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    .prediction-card-atrisk {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 2px solid #F59E0B;
        border-radius: 14px;
        padding: 20px 24px;
        color: #92400E;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);
    }
    .prediction-card-critical {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 20px 24px;
        color: #991B1B;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
    }
    .prediction-title {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .prediction-main-status {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .prediction-confidence {
        font-size: 14px;
        font-weight: 600;
        opacity: 0.9;
        margin-top: 4px;
    }

    /* Tactical Action Directive */
    .directive-card {
        background: #FFFFFF;
        border-left: 5px solid #2563EB;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .directive-card.critical {
        border-left-color: #EF4444;
        background: #FFFBFB;
    }
    .directive-card.warning {
        border-left-color: #F59E0B;
        background: #FFFDF9;
    }
    .directive-card.ontime {
        border-left-color: #10B981;
        background: #F9FDFB;
    }
    .directive-badge {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #2563EB;
        margin-bottom: 4px;
    }
    .directive-card.critical .directive-badge { color: #DC2626; }
    .directive-card.warning .directive-badge { color: #D97706; }
    .directive-card.ontime .directive-badge { color: #059669; }
    .directive-text {
        font-size: 13.5px;
        font-weight: 600;
        color: #1E293B;
        line-height: 1.5;
    }

    /* Corridor Progress Stepper */
    .corridor-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 22px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
    }
    .corridor-step {
        text-align: center;
        flex: 1;
    }
    .corridor-dot {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #EFF6FF;
        border: 2px solid #2563EB;
        color: #2563EB;
        font-size: 12px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 6px auto;
    }
    .corridor-dot.active {
        background: #2563EB;
        color: #FFFFFF;
    }
    .corridor-dot.alert {
        background: #EF4444;
        border-color: #DC2626;
        color: #FFFFFF;
    }
    .corridor-name {
        font-size: 11px;
        font-weight: 600;
        color: #475569;
    }
    .corridor-line {
        flex: 1;
        height: 3px;
        background: #E2E8F0;
        margin: 0 10px 20px 10px;
    }
    .corridor-line.active {
        background: #93C5FD;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0 20px;
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2563EB;
        border-color: #BFDBFE;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. MODEL LOADING HELPER
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
# 3. HEADER BANNER
# ---------------------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-badge">🚢 AI-Powered Logistics Intelligence</div>
    <h1>Global Supply Chain Bottleneck Predictor</h1>
    <p>Real-time Port-to-Warehouse Congestion Triage & Proactive Delay Mitigation</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 4. SIDEBAR: OPERATIONAL CONTROLS & PRESET DISRUPTIONS
# ---------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Operational Presets")

scenarios = {
    "⚡ Choose a Quick Preset...": None,
    "🟢 Normal Corridor Flow (Shanghai ➔ LA)": {
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

selected_preset_name = st.sidebar.selectbox("Quick-Load Disruption Scenario:", list(scenarios.keys()))
preset = scenarios[selected_preset_name] if selected_preset_name != "⚡ Choose a Quick Preset..." else None

# Pre-populated defaults
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
st.sidebar.markdown("### 📍 Shipment Route & Carrier")

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
st.sidebar.markdown("### 📡 Port & Yard Telemetry Inputs")

dwell = st.sidebar.slider("Container Dwell Time (Hours)", 4.0, 95.0, float(val_dwell), 0.5)
capacity = st.sidebar.slider("Yard Capacity Utilization (%)", 30.0, 120.0, float(val_cap), 1.0)
throughput = st.sidebar.slider("Berth Throughput (TEU/hr)", 60.0, 430.0, float(val_tp), 5.0)
customs = st.sidebar.slider("Customs Hold Duration (Hours)", 0.0, 48.0, float(val_cust), 0.5)
weather = st.sidebar.slider("Weather Severity Index (0-10)", 0.0, 10.0, float(val_weath), 0.1)
labor = st.sidebar.slider("Labor Deficit Ratio (0.0 - 0.5)", 0.0, 0.5, float(val_labor), 0.005)

# ---------------------------------------------------------------------
# 5. INFERENCE EXECUTION
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

# Anomaly check
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
# 6. TOP KPI SUMMARY ROW
# ---------------------------------------------------------------------
origin_short = origin.replace("PORT_", "")
dest_short = destination.replace("PORT_", "")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Active Corridor</div>
        <div class="kpi-val" style="font-size:20px;">{origin_short} ➔ {dest_short}</div>
    </div>
    ''', unsafe_allow_html=True)

with kpi2:
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Carrier & Priority</div>
        <div class="kpi-val" style="font-size:20px;">{carrier} • <span style="font-size:15px; color:#2563EB;">{priority}</span></div>
    </div>
    ''', unsafe_allow_html=True)

with kpi3:
    status_color = "green" if not is_anomaly else "amber"
    status_text = "Nominal" if not is_anomaly else "Irregular"
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Sensor Telemetry</div>
        <div class="kpi-val {status_color}" style="font-size:20px;">{"● " + status_text}</div>
    </div>
    ''', unsafe_allow_html=True)

with kpi4:
    risk_color = "green" if risk_score < 35 else ("amber" if risk_score < 70 else "red")
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Bottleneck Risk Index</div>
        <div class="kpi-val {risk_color}">{risk_score}%</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 7. ROUTE CORRIDOR FLOW STEPPER
# ---------------------------------------------------------------------
c_dwell_alert = "alert" if dwell > 30 else ("active" if dwell > 20 else "")
c_yard_alert = "alert" if capacity > 90 else ("active" if capacity > 75 else "")
c_gate_alert = "alert" if pred_label == "CRITICAL_DELAY" else ("active" if pred_label == "AT_RISK" else "")

st.markdown(f"""
<div class="corridor-container">
    <div class="corridor-step">
        <div class="corridor-dot active">1</div>
        <div class="corridor-name">Origin Berth<br><span style="color:#2563EB; font-weight:700;">{origin_short}</span></div>
    </div>
    <div class="corridor-line active"></div>
    <div class="corridor-step">
        <div class="corridor-dot active">2</div>
        <div class="corridor-name">Transpacific<br><span style="color:#64748B;">Transit</span></div>
    </div>
    <div class="corridor-line {'active' if dwell > 20 else ''}"></div>
    <div class="corridor-step">
        <div class="corridor-dot {c_dwell_alert if c_dwell_alert else 'active'}">3</div>
        <div class="corridor-name">Port Customs<br><span style="color:#64748B;">{customs:.1f}h hold</span></div>
    </div>
    <div class="corridor-line {'active' if capacity > 75 else ''}"></div>
    <div class="corridor-step">
        <div class="corridor-dot {c_yard_alert if c_yard_alert else 'active'}">4</div>
        <div class="corridor-name">Terminal Yard<br><span style="color:#64748B;">{capacity:.0f}% full</span></div>
    </div>
    <div class="corridor-line {'active' if pred_label != 'ON_TIME' else ''}"></div>
    <div class="corridor-step">
        <div class="corridor-dot {c_gate_alert if c_gate_alert else 'active'}">5</div>
        <div class="corridor-name">Destination Gate<br><span style="color:#2563EB; font-weight:700;">{dest_short}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 8. MAIN INTERACTIVE TABS
# ---------------------------------------------------------------------
tab_triage, tab_visuals = st.tabs(["🎯 Live Bottleneck Prediction", "📊 Telemetry Analytics & Sensitivity"])

# ---------------------------------------------------------------------
# TAB 1: PREDICTION & ACTION DIRECTIVE
# ---------------------------------------------------------------------
with tab_triage:
    col_pred_left, col_pred_right = st.columns([1.1, 1.0], gap="large")
    
    with col_pred_left:
        # Prediction Status Banner
        if pred_label == "ON_TIME":
            st.markdown(f"""
            <div class="prediction-card-ontime">
                <div class="prediction-title">Shipment Velocity Forecast</div>
                <div class="prediction-main-status">🟢 ON TIME</div>
                <div class="prediction-confidence">Confidence: <strong>{confidence_pct}%</strong> • Standard clearance window maintained</div>
            </div>
            """, unsafe_allow_html=True)
        elif pred_label == "AT_RISK":
            st.markdown(f"""
            <div class="prediction-card-atrisk">
                <div class="prediction-title">Shipment Velocity Forecast</div>
                <div class="prediction-main-status">🟡 AT RISK (Moderate Delay)</div>
                <div class="prediction-confidence">Confidence: <strong>{confidence_pct}%</strong> • Potential +12h to +24h schedule slippage</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card-critical">
                <div class="prediction-title">Shipment Velocity Forecast</div>
                <div class="prediction-main-status">🔴 CRITICAL DELAY</div>
                <div class="prediction-confidence">Confidence: <strong>{confidence_pct}%</strong> • Severe corridor gridlock detected (&gt;48h delay)</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Tactical Operational Directive
        if pred_label == "CRITICAL_DELAY":
            st.markdown("""
            <div class="directive-card critical">
                <div class="directive-badge">⚡ Tactical Action Directive // Emergency Escalation</div>
                <div class="directive-text">
                    <strong>REROUTE TO SECONDARY INLAND RAIL:</strong> Immediately trigger off-dock container drayage. Bypass congested terminal gates and rebook on Union Pacific/BNSF priority rail corridors to prevent compounding factory halts.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif pred_label == "AT_RISK":
            st.markdown("""
            <div class="directive-card warning">
                <div class="directive-badge">⚠️ Tactical Action Directive // Proactive Buffer Extension</div>
                <div class="directive-text">
                    <strong>EXTEND APPOINTMENT WINDOW:</strong> Grant +18 hour extension to distribution centers. Alert drayage dispatch to prioritize night-gate turns while monitoring terminal dwell closely.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="directive-card ontime">
                <div class="directive-badge">✅ Tactical Action Directive // Standard Transit Flow</div>
                <div class="directive-text">
                    <strong>MAINTAIN PLANNED DISPATCH:</strong> Container is clearing berth cycles within contractual SLA limits. Proceed with standard multimodal drayage schedule.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Sensor Anomaly Banner
        if is_anomaly:
            st.warning("⚠️ **Sensor Telemetry Irregularity**: Physical sensor readings deviate from historical operational distributions. Cross-reference berth gate counters.")
            
        # Leading Telemetry Drivers Table
        st.markdown("<h4 style='font-size:15px; font-weight:700; color:#1E3A8A; margin-top:14px; margin-bottom:8px;'>🔍 Key Telemetry Risk Drivers</h4>", unsafe_allow_html=True)
        drivers = [
            {"Factor": "Container Dwell Time", "Observed": f"{dwell:.1f} hrs", "Safe Range": "< 24 hrs", "Status": "🔴 Critical" if dwell > 35 else ("🟡 Elevated" if dwell > 24 else "🟢 Normal")},
            {"Factor": "Yard Saturation", "Observed": f"{capacity:.1f}%", "Safe Range": "< 80%", "Status": "🔴 Overflow" if capacity > 100 else ("🟡 High" if capacity > 80 else "🟢 Optimal")},
            {"Factor": "Customs Hold", "Observed": f"{customs:.1f} hrs", "Safe Range": "< 4 hrs", "Status": "🔴 Backlogged" if customs > 8 else ("🟡 Queued" if customs > 4 else "🟢 Cleared")},
            {"Factor": "Berth Crane Velocity", "Observed": f"{throughput:.0f} TEU/h", "Safe Range": "> 180 TEU/h", "Status": "🔴 Choked" if throughput < 120 else ("🟡 Moderate" if throughput < 180 else "🟢 Swift")},
            {"Factor": "Weather Severity", "Observed": f"{weather:.1f} / 10", "Safe Range": "< 3.0", "Status": "🔴 Severe Gale" if weather > 4.5 else ("🟡 Adverse" if weather > 2.5 else "🟢 Calm")}
        ]
        st.dataframe(pd.DataFrame(drivers), hide_index=True, use_container_width=True)

    with col_pred_right:
        # VISUALIZATION 1: Bottleneck Risk Speedometer Gauge (Plotly)
        st.markdown("<h4 style='font-size:15px; font-weight:700; color:#1E3A8A; margin-bottom:2px;'>⏱️ Bottleneck Risk Gauge</h4>", unsafe_allow_html=True)
        
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={'suffix': "%", 'font': {'size': 36, 'family': 'Plus Jakarta Sans', 'color': '#1E3A8A'}},
            title={'text': "Composite Delay Risk Index", 'font': {'size': 12, 'color': '#64748B'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': "#2563EB", 'thickness': 0.28},
                'bgcolor': "#F1F5F9",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': '#D1FAE5'},   # Soft Emerald
                    {'range': [35, 70], 'color': '#FEF3C7'},  # Soft Amber
                    {'range': [70, 100], 'color': '#FEE2E2'}  # Soft Crimson
                ],
                'threshold': {
                    'line': {'color': "#DC2626", 'width': 3},
                    'thickness': 0.8,
                    'value': risk_score
                }
            }
        ))
        gauge_fig.update_layout(
            height=210,
            margin=dict(l=20, r=20, t=25, b=10),
            paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans")
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        # VISUALIZATION 2: Calibrated Probability Donut Chart (Plotly)
        st.markdown("<h4 style='font-size:15px; font-weight:700; color:#1E3A8A; margin-top:8px; margin-bottom:2px;'>📊 Probability Share</h4>", unsafe_allow_html=True)
        
        donut_labels = ["On Time", "At Risk", "Critical Delay"]
        donut_values = [p_on * 100, p_risk * 100, p_crit * 100]
        donut_colors = ["#10B981", "#F59E0B", "#EF4444"]

        donut_fig = go.Figure(data=[go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=0.62,
            marker=dict(colors=donut_colors, line=dict(color="#FFFFFF", width=3)),
            textinfo='percent',
            hoverinfo='label+percent',
            textfont=dict(size=13, family="Plus Jakarta Sans", color="#FFFFFF")
        )])
        donut_fig.update_layout(
            height=210,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=12, color="#334155")),
            annotations=[dict(text=f"{pred_label.replace('_', ' ')}<br><span style='font-size:18px; font-weight:800; color:#1E3A8A;'>{confidence_pct}%</span>", x=0.5, y=0.5, font_size=11, font_family="Plus Jakarta Sans", showarrow=False)]
        )
        st.plotly_chart(donut_fig, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: ADVANCED TELEMETRY VISUALIZATIONS & SENSITIVITY
# ---------------------------------------------------------------------
with tab_visuals:
    col_v1, col_v2 = st.columns(2, gap="large")
    
    with col_v1:
        # VISUALIZATION 3: Telemetry Radar / Spider Chart (Plotly)
        st.markdown("<h4 style='font-size:16px; font-weight:700; color:#1E3A8A;'>🕸️ 6-Axis Telemetry Health Radar</h4>", unsafe_allow_html=True)
        st.caption("Compares current operational conditions against the recommended baseline operating envelope.")
        
        radar_categories = [
            'Dwell Time', 'Yard Saturation', 'Customs Duration',
            'Berth Slowdown', 'Weather Stress', 'Labor Deficit'
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
        
        # Baseline Area
        radar_fig.add_trace(go.Scatterpolar(
            r=baseline_envelope + [baseline_envelope[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor='rgba(148, 163, 184, 0.2)',
            line=dict(color='#94A3B8', dash='dash'),
            name='Optimal Baseline'
        ))
        
        # Current Observation
        radar_fig.add_trace(go.Scatterpolar(
            r=current_footprint + [current_footprint[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor='rgba(37, 99, 235, 0.28)',
            line=dict(color='#2563EB', width=2.5),
            name='Current Shipment'
        ))

        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#94A3B8", gridcolor="#E2E8F0"),
                angularaxis=dict(color="#1E293B", gridcolor="#E2E8F0")
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            height=340,
            margin=dict(l=35, r=35, t=20, b=20),
            paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with col_v2:
        # VISUALIZATION 4: What-If Sensitivity Simulator (Plotly)
        st.markdown("<h4 style='font-size:16px; font-weight:700; color:#1E3A8A;'>📈 What-If Sensitivity Curve</h4>", unsafe_allow_html=True)
        st.caption("Simulates delay probability shift as Container Dwell Time increases under present conditions.")
        
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
            line=dict(color='#10B981', width=3)
        ))
        curve_fig.add_trace(go.Scatter(
            x=curve_df["Dwell_Hours"], y=curve_df["p_atrisk"],
            mode='lines', name='At Risk',
            line=dict(color='#F59E0B', width=3)
        ))
        curve_fig.add_trace(go.Scatter(
            x=curve_df["Dwell_Hours"], y=curve_df["p_critical"],
            mode='lines', name='Critical Delay',
            line=dict(color='#EF4444', width=3)
        ))
        
        # Add vertical marker for current dwell time
        curve_fig.add_vline(
            x=dwell, line_width=2, line_dash="dash", line_color="#1E3A8A",
            annotation_text=f"Current: {dwell:.1f}h", annotation_position="top left",
            annotation_font=dict(color="#1E3A8A", size=11, family="Plus Jakarta Sans")
        )

        curve_fig.update_layout(
            xaxis_title="Dwell Time (Hours)",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 105], gridcolor="#F1F5F9"),
            xaxis=dict(gridcolor="#F1F5F9"),
            height=340,
            margin=dict(l=30, r=20, t=20, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(curve_fig, use_container_width=True)

# ---------------------------------------------------------------------
# 9. FOOTER
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #64748B; padding: 6px 0;">
    <div><strong>Global Supply Chain Logistics Intelligence Platform</strong> • Production Operations Engine</div>
    <div>Live Telemetry Active</div>
</div>
""", unsafe_allow_html=True)

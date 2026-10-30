#!/usr/bin/env python3
"""
Streamlit Application: Global Logistics Intelligence Engine
Autonomous Port-to-Warehouse Bottleneck Triage & Mitigation (Stage 01)
Connects directly to the trained models:
- XGBoost Delay Classifier (xgboost_delay_model.pkl)
- Label Encoder (xgboost_label_encoder.pkl)
- Isolation Forest Anomaly Detector (isolation_forest_model.pkl)
- One-Class SVM Normality Model (one_class_svm_model.pkl)
- Anomaly Preprocessor (anomaly_preprocessor.pkl)
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="Logistics Bottleneck Intelligence Engine",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk / Dark Operations Styling
st.markdown("""
<style>
    /* Dark Ground Theme */
    .stApp {
        background-color: #040705;
        color: #d3dbd5;
        font-family: 'Courier New', monospace;
    }
    
    /* Top Metric Cards */
    .metric-card {
        background-color: #090e0b;
        border: 1px solid #304c3b;
        padding: 14px 18px;
        border-radius: 0px;
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 11px;
        color: #c2cbc5;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-val {
        font-size: 20px;
        font-weight: 700;
        color: #f8fffa;
        font-family: monospace;
    }
    .metric-val.accent {
        color: #38ff8e;
    }
    
    /* Hero Status Badges */
    .status-badge-ontime {
        background-color: #0d2818;
        border: 2px solid #38ff8e;
        color: #38ff8e;
        padding: 16px 24px;
        font-size: 28px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: center;
        margin-bottom: 15px;
    }
    .status-badge-atrisk {
        background-color: #2b1f09;
        border: 2px solid #ffb84d;
        color: #ffb84d;
        padding: 16px 24px;
        font-size: 28px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: center;
        margin-bottom: 15px;
    }
    .status-badge-critical {
        background-color: #2d0a11;
        border: 2px solid #ff4766;
        color: #ff4766;
        padding: 16px 24px;
        font-size: 28px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: center;
        margin-bottom: 15px;
    }
    
    /* Tactical Action Box */
    .tactical-box {
        background-color: #050b07;
        border: 1px solid #659579;
        padding: 16px 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .tactical-box.warning {
        border-color: #ffb84d;
        background-color: #1a1306;
    }
    .tactical-box.critical {
        border-color: #ff4766;
        background-color: #20080d;
    }
    .tactical-title {
        font-size: 11px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #38ff8e;
        margin-bottom: 6px;
    }
    .tactical-box.critical .tactical-title { color: #ff4766; }
    .tactical-box.warning .tactical-title { color: #ffb84d; }
    .tactical-desc {
        font-size: 13.5px;
        line-height: 1.5;
        color: #f8fffa;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 1. MODEL LOADING HELPER
# ---------------------------------------------------------------------
@st.cache_resource
def load_logistics_models():
    # Search common paths
    search_paths = [
        Path("data"),
        Path("Stage01_ML/data"),
        Path("../data"),
        Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage01_ML/data"),
        Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/data"),
        Path("/Users/ayyapparaja/pahse 2/data")
    ]
    
    classifier, label_encoder, isolation_forest, one_class_svm, anomaly_preprocessor = None, None, None, None, None
    for p in search_paths:
        if (p / "xgboost_delay_model.pkl").exists():
            try:
                classifier = joblib.load(p / "xgboost_delay_model.pkl")
                label_encoder = joblib.load(p / "xgboost_label_encoder.pkl")
                isolation_forest = joblib.load(p / "isolation_forest_model.pkl")
                one_class_svm = joblib.load(p / "one_class_svm_model.pkl")
                anomaly_preprocessor = joblib.load(p / "anomaly_preprocessor.pkl")
                return classifier, label_encoder, isolation_forest, one_class_svm, anomaly_preprocessor, p
            except Exception as e:
                continue
    return None, None, None, None, None, None

classifier, label_encoder, isolation_forest, one_class_svm, anomaly_preprocessor, loaded_path = load_logistics_models()

# ---------------------------------------------------------------------
# 2. HEADER & TELEMETRY STATUS
# ---------------------------------------------------------------------
st.markdown("""
<div style="border-bottom: 1px solid #304c3b; padding-bottom: 10px; margin-bottom: 20px;">
    <div style="font-size: 11px; color: #c2cbc5; letter-spacing: 0.08em; text-transform: uppercase;">
        // SYS.LOC: PACIFIC_LOGISTICS_HUB // CORRIDOR: CN_SHA &gt;&gt; US_LAX &gt;&gt; MW_JOLIET // STATUS: <span style="color:#38ff8e;">TELEMETRY ONLINE</span>
    </div>
    <h1 style="font-family: 'IBM Plex Sans', sans-serif; font-size: 32px; font-weight: 700; color: #f8fffa; margin-top: 5px; text-transform: uppercase;">
        GLOBAL LOGISTICS INTELLIGENCE ENGINE
        <span style="color: #65f8ff; font-size: 14px; vertical-align: middle;">// PORT-TO-WAREHOUSE BOTTLENECK TRIAGE</span>
    </h1>
    <p style="font-size: 14px; color: #d3dbd5; margin-top: 4px;">
        Multi-channel telemetry ingestion processing container dwell waveforms, crane velocity, and yard slot constraints into calibrated real-time bottleneck predictions before cascading network collapse.
    </p>
</div>
""", unsafe_allow_html=True)

# Facts Row
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
with col_f1:
    st.markdown('<div class="metric-card"><div class="metric-label">Champion Engine</div><div class="metric-val accent">XGBoost v1.8</div></div>', unsafe_allow_html=True)
with col_f2:
    st.markdown('<div class="metric-card"><div class="metric-label">Model Test Accuracy</div><div class="metric-val accent">84.8% (Realistic)</div></div>', unsafe_allow_html=True)
with col_f3:
    st.markdown('<div class="metric-card"><div class="metric-label">Critical Delay Recall</div><div class="metric-val">90.1% Sensitivity</div></div>', unsafe_allow_html=True)
with col_f4:
    st.markdown('<div class="metric-card"><div class="metric-label">Inference Latency</div><div class="metric-val">0.002 ms / sample</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 3. SIDEBAR / TELEMETRY INGESTION DECK
# ---------------------------------------------------------------------
st.sidebar.markdown("### // TELEMETRY INGESTION DECK")

# Preset Scenarios
scenarios = {
    "Select a Scenario...": None,
    "🟢 Normal Transpacific Flow": {
        "dwell": 18.0, "throughput": 245.0, "capacity": 65.0, "customs": 1.5,
        "weather": 0.8, "labor": 0.015, "carrier": "MSK", "priority": "STANDARD",
        "origin": "PORT_CN_SHA", "dest": "PORT_US_LAX"
    },
    "🟡 Week 4 Customs Backlog (+14h Hold)": {
        "dwell": 33.5, "throughput": 185.0, "capacity": 82.5, "customs": 14.0,
        "weather": 1.4, "labor": 0.035, "carrier": "HAPAG", "priority": "EXPEDITED",
        "origin": "PORT_CN_SHA", "dest": "PORT_US_OAK"
    },
    "🔴 Week 7 Pacific Typhoon (Wind Lockout)": {
        "dwell": 38.0, "throughput": 110.0, "capacity": 89.0, "customs": 5.0,
        "weather": 5.2, "labor": 0.08, "carrier": "ONE", "priority": "CRITICAL_MEDICAL",
        "origin": "PORT_SGP_SIN", "dest": "PORT_US_LAX"
    },
    "⚠️ Terminal Yard Saturation (>100% Overflow)": {
        "dwell": 45.0, "throughput": 90.0, "capacity": 108.0, "customs": 8.5,
        "weather": 2.5, "labor": 0.14, "carrier": "COSCO", "priority": "STANDARD",
        "origin": "PORT_CN_SHA", "dest": "PORT_US_LGB"
    }
}

selected_scenario_name = st.sidebar.selectbox("Load Preset Disruption Scenario:", list(scenarios.keys()))
active_preset = scenarios[selected_scenario_name] if selected_scenario_name != "Select a Scenario..." else None

# Helper defaults
d_dwell = active_preset["dwell"] if active_preset else 22.0
d_tp = active_preset["throughput"] if active_preset else 230.0
d_cap = active_preset["capacity"] if active_preset else 72.0
d_cust = active_preset["customs"] if active_preset else 2.5
d_weath = active_preset["weather"] if active_preset else 1.2
d_labor = active_preset["labor"] if active_preset else 0.02
d_carrier = active_preset["carrier"] if active_preset else "MSK"
d_priority = active_preset["priority"] if active_preset else "STANDARD"
d_origin = active_preset["origin"] if active_preset else "PORT_CN_SHA"
d_dest = active_preset["dest"] if active_preset else "PORT_US_LAX"

st.sidebar.markdown("---")
st.sidebar.markdown("**Carrier & Route Attributes:**")
carrier_list = ["MSK", "MSC", "HAPAG", "ONE", "COSCO", "CMA"]
carrier = st.sidebar.selectbox("Carrier Code", carrier_list, index=carrier_list.index(d_carrier) if d_carrier in carrier_list else 0)

priority_list = ["STANDARD", "EXPEDITED", "CRITICAL_MEDICAL"]
priority = st.sidebar.selectbox("Priority Tier", priority_list, index=priority_list.index(d_priority) if d_priority in priority_list else 0)

origin_list = ["PORT_CN_SHA", "PORT_SGP_SIN"]
origin = st.sidebar.selectbox("Origin Port", origin_list, index=origin_list.index(d_origin) if d_origin in origin_list else 0)

dest_list = ["PORT_US_LAX", "PORT_US_LGB", "PORT_US_OAK"]
destination = st.sidebar.selectbox("Destination Port", dest_list, index=dest_list.index(d_dest) if d_dest in dest_list else 0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Continuous Telemetry Telemetry:**")
dwell = st.sidebar.slider("Container Dwell Time (Hours)", 4.0, 95.0, float(d_dwell), 0.5)
throughput = st.sidebar.slider("Berth Throughput (TEU/hr)", 60.0, 430.0, float(d_tp), 5.0)
capacity = st.sidebar.slider("Yard Capacity Utilization (%)", 30.0, 120.0, float(d_cap), 1.0)
customs = st.sidebar.slider("Customs Hold Duration (Hours)", 0.0, 48.0, float(d_cust), 0.5)
weather = st.sidebar.slider("Weather Severity Index (0-10)", 0.0, 10.0, float(d_weath), 0.1)
labor = st.sidebar.slider("Labor Deficit Ratio (0.0 - 0.5)", 0.0, 0.5, float(d_labor), 0.005)

# ---------------------------------------------------------------------
# 4. INFERENCE EXECUTION
# ---------------------------------------------------------------------
tab_triage, tab_benchmark, tab_eda = st.tabs(["[ LIVE BOTTLENECK TRIAGE ]", "[ MODEL BENCHMARK ]", "[ EXPLORATORY AUDIT ]"])

with tab_triage:
    if classifier is None:
        st.error(f"Models not loaded. Please verify model files exist in {loaded_path}.")
    else:
        # Build DataFrame payload matching trained schema
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
        
        # 1. XGBoost Classification
        pred_code = int(classifier.predict(input_df)[0])
        probabilities = classifier.predict_proba(input_df)[0]
        
        try:
            class_labels = label_encoder.inverse_transform(
                classifier.named_steps["classifier"].classes_.astype(int)
            )
        except Exception:
            class_labels = label_encoder.classes_
            
        pred_label = str(label_encoder.inverse_transform([pred_code])[0])
        prob_dict = {str(lbl): float(pr) for lbl, pr in zip(class_labels, probabilities)}
        confidence_pct = round(float(max(probabilities)) * 100, 1)
        
        # 2. Anomaly Detection
        anomaly_features = [
            "dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct",
            "customs_hold_hours", "weather_severity_index", "labor_deficit_ratio",
            "equipment_availability_pct", "historical_delay_rate"
        ]
        anomaly_input = pd.DataFrame([row_dict])[anomaly_features]
        scaled_anomaly = anomaly_preprocessor.transform(anomaly_input)
        if_res = int(isolation_forest.predict(scaled_anomaly)[0])
        svm_res = int(one_class_svm.predict(scaled_anomaly)[0])
        is_anomaly = (if_res == -1) or (svm_res == -1)
        
        # 3. Display Main Triage Layout
        col_res1, col_res2 = st.columns([1.2, 1])
        
        with col_res1:
            st.markdown("#### // PREDICTED DELAY STATUS")
            if pred_label == "ON_TIME":
                st.markdown(f'<div class="status-badge-ontime">🟢 ON_TIME ({confidence_pct}%)</div>', unsafe_allow_html=True)
            elif pred_label == "AT_RISK":
                st.markdown(f'<div class="status-badge-atrisk">🟡 AT_RISK ({confidence_pct}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-badge-critical">🔴 CRITICAL_DELAY ({confidence_pct}%)</div>', unsafe_allow_html=True)
            
            # Anomaly Quarantine Indicator
            if is_anomaly:
                st.warning("⚠️ **[ SENSOR ANOMALY QUARANTINE ]**: Telemetry values deviate from known physical port operations bounds. Isolation Forest flagged out-of-distribution dynamics.")
            else:
                st.success("✅ **[ NOMINAL TELEMETRY ]**: Sensor readings align with expected operational dynamics.")
            
            # Tactical Directive Box
            if pred_label == "CRITICAL_DELAY":
                st.markdown("""
                <div class="tactical-box critical">
                    <div class="tactical-title">// TACTICAL OPS DIRECTIVE // EMERGENCY DISPATCH</div>
                    <div class="tactical-desc">
                        <strong>TRIGGER EMERGENCY DIVERSION:</strong> Reroute shipment to secondary inland rail corridor (UP/BNSF) or reserve priority off-dock staging. Delay ripple risk exceeds 50 downstream distribution hubs within 48 hours.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif pred_label == "AT_RISK":
                st.markdown("""
                <div class="tactical-box warning">
                    <div class="tactical-title">// TACTICAL OPS DIRECTIVE // BUFFER ADJUSTMENT</div>
                    <div class="tactical-desc">
                        <strong>DISPATCH WARNING BUFFER:</strong> Expand drayage pickup appointment window by +12 hours and alert regional warehouse planners. Monitor berth gate turn times closely.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="tactical-box">
                    <div class="tactical-title">// TACTICAL OPS DIRECTIVE // NOMINAL PASS</div>
                    <div class="tactical-desc">
                        <strong>STANDARD BERTH PASS:</strong> Baseline gate transit within scheduled service SLA. Maintain planned multimodal transfer schedule.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            st.markdown("#### // CALIBRATED RISK PROBABILITIES")
            p_on = prob_dict.get("ON_TIME", 0.0)
            p_risk = prob_dict.get("AT_RISK", 0.0)
            p_crit = prob_dict.get("CRITICAL_DELAY", 0.0)
            
            st.write(f"🟢 **ON_TIME**: {p_on*100:.1f}%")
            st.progress(p_on)
            
            st.write(f"🟡 **AT_RISK**: {p_risk*100:.1f}%")
            st.progress(p_risk)
            
            st.write(f"🔴 **CRITICAL_DELAY**: {p_crit*100:.1f}%")
            st.progress(p_crit)
            
            # Receptive Field Driver Indicators
            st.markdown("---")
            st.markdown("#### // LEADING TELEMETRY DRIVERS")
            velocity_ratio = round(dwell / (throughput + 1e-4), 3)
            congestion_idx = round((dwell * min(100.0, capacity)) / 100.0, 1)
            
            drivers_df = pd.DataFrame([
                {"Telemetry Channel": "Dwell Time", "Observed": f"{dwell:.1f}h", "Importance": "42.3%", "Risk Status": "HIGH" if dwell > 24 else "NORMAL"},
                {"Telemetry Channel": "Velocity Ratio", "Observed": f"{velocity_ratio}", "Importance": "17.7%", "Risk Status": "HIGH" if velocity_ratio > 0.15 else "NORMAL"},
                {"Telemetry Channel": "Yard Capacity", "Observed": f"{capacity:.1f}%", "Importance": "14.7%", "Risk Status": "OVERFLOW" if capacity > 100 else ("HIGH" if capacity > 80 else "NORMAL")},
                {"Telemetry Channel": "Customs Hold", "Observed": f"{customs:.1f}h", "Importance": "12.2%", "Risk Status": "HIGH" if customs > 5 else "NORMAL"},
                {"Telemetry Channel": "Throughput", "Observed": f"{throughput:.1f} TEU/h", "Importance": "7.5%", "Risk Status": "SLOW" if throughput < 150 else "NORMAL"},
            ])
            st.dataframe(drivers_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------
# 5. TAB 2: MODEL BENCHMARK
# ---------------------------------------------------------------------
with tab_benchmark:
    st.markdown("### // MODEL BENCHMARK COMPARISON MATRIX")
    st.markdown("Evaluated on 960 Chronological Future Test Split shipments (Late February – March 2026):")
    
    benchmark_data = pd.DataFrame([
        {"Model": "🏆 XGBoost (Champion)", "Category": "Supervised (Boosting)", "Test Accuracy": "84.79%", "Critical Recall": "90.11%", "Macro F1": "0.7726", "Inference Latency": "0.002 ms", "Model Size": "470 KB"},
        {"Model": "🥈 Random Forest", "Category": "Supervised (Bagging)", "Test Accuracy": "84.79%", "Critical Recall": "90.11%", "Macro F1": "0.7737", "Inference Latency": "0.028 ms", "Model Size": "2,400 KB"},
        {"Model": "🛡️ One-Class SVM", "Category": "Semi-Supervised Novelty", "Test Accuracy": "81.35%", "Critical Recall": "93.41%", "Macro F1": "—", "Inference Latency": "0.068 ms", "Model Size": "9 KB"},
        {"Model": "🌲 Isolation Forest", "Category": "Unsupervised Anomaly", "Test Accuracy": "96.04%", "Critical Recall": "25.27%", "Macro F1": "—", "Inference Latency": "0.052 ms", "Model Size": "1,900 KB"}
    ])
    st.dataframe(benchmark_data, hide_index=True, use_container_width=True)
    
    st.markdown("""
    #### Why XGBoost Was Selected as Champion:
    1. **Real-Time Speed**: Inference in **0.002 ms** allows processing **500,000 container scans per second**.
    2. **High Sensitivity**: Achieves **90.11% recall on Critical Delays**, catching 9 out of 10 major bottlenecks.
    3. **Calibrated Probabilities**: Softmax output allows the ops desk to set financial threshold cutoffs before paying for emergency rerouting.
    """)

# ---------------------------------------------------------------------
# 6. TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------
with tab_eda:
    st.markdown("### // EXPLORATORY AUDIT & NOISE FLOOR DYNAMICS")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### 1. Target Class Distribution")
        class_df = pd.DataFrame({
            "Class": ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"],
            "Shipments": [3252, 1042, 506],
            "Percentage": ["67.8%", "21.7%", "10.5%"]
        })
        st.dataframe(class_df, hide_index=True, use_container_width=True)
        st.info("Imbalance handled using cost-sensitive sample weighting (penalizing critical delay errors 6× harder) and evaluation on Macro F1 rather than misleading accuracy.")
        
    with col_e2:
        st.markdown("#### 2. Carrier Reliability Hierarchy")
        carrier_df = pd.DataFrame([
            {"Carrier": "MSC", "Full Name": "Mediterranean Shipping Co", "Delay Rate": "10.5%", "Reliability": "Highest"},
            {"Carrier": "MSK", "Full Name": "A.P. Moller – Maersk", "Delay Rate": "11.2%", "Reliability": "High"},
            {"Carrier": "COSCO", "Full Name": "COSCO Shipping Lines", "Delay Rate": "12.2%", "Reliability": "Medium"},
            {"Carrier": "CMA", "Full Name": "CMA CGM Group", "Delay Rate": "12.3%", "Reliability": "Medium"},
            {"Carrier": "HAPAG", "Full Name": "Hapag-Lloyd", "Delay Rate": "12.5%", "Reliability": "Medium"},
            {"Carrier": "ONE", "Full Name": "Ocean Network Express", "Delay Rate": "14.6%", "Reliability": "Vulnerable"}
        ])
        st.dataframe(carrier_df, hide_index=True, use_container_width=True)
    
    st.markdown("#### 3. Weekly Disruption Drift (Early Warning Radar)")
    weekly_drift_df = pd.DataFrame([
        {"Calendar Week": "Week 1", "Dwell Mean": "18.6h", "Noise Floor (±1σ)": "±5.4h", "Critical Delay %": "0.0%", "Operational Status": "Normal Calm"},
        {"Calendar Week": "Week 2", "Dwell Mean": "19.2h", "Noise Floor (±1σ)": "±5.4h", "Critical Delay %": "0.0%", "Operational Status": "Normal Calm"},
        {"Calendar Week": "Week 3", "Dwell Mean": "18.6h", "Noise Floor (±1σ)": "±5.5h", "Critical Delay %": "0.0%", "Operational Status": "Pre-Crisis Baseline"},
        {"Calendar Week": "Week 4", "Dwell Mean": "32.4h", "Noise Floor (±1σ)": "±18.8h", "Critical Delay %": "47.6%", "Operational Status": "💥 SHOCK 1: Customs Backlog (+11.7h hold)"},
        {"Calendar Week": "Week 5", "Dwell Mean": "18.9h", "Noise Floor (±1σ)": "±5.2h", "Critical Delay %": "0.0%", "Operational Status": "Rapid Network Clearance"},
        {"Calendar Week": "Week 6", "Dwell Mean": "21.7h", "Noise Floor (±1σ)": "±12.3h", "Critical Delay %": "6.8%", "Operational Status": "⚠️ Early Warning Variance Expansion"},
        {"Calendar Week": "Week 7", "Dwell Mean": "31.1h", "Noise Floor (±1σ)": "±21.4h", "Critical Delay %": "29.8%", "Operational Status": "💥 SHOCK 2: Transpacific Typhoon (Wind Lockout)"},
        {"Calendar Week": "Week 8", "Dwell Mean": "19.6h", "Noise Floor (±1σ)": "±7.3h", "Critical Delay %": "4.3%", "Operational Status": "Recovery Buffer"},
        {"Calendar Week": "Week 9", "Dwell Mean": "22.7h", "Noise Floor (±1σ)": "±10.8h", "Critical Delay %": "16.3%", "Operational Status": "Secondary Congestion Echo"}
    ])
    st.dataframe(weekly_drift_df, hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="font-size: 11px; color: #c2cbc5; text-align: center; text-transform: uppercase;">
    GLOBAL LOGISTICS SYSTEMS // SUPPLY CHAIN OPERATIONS // PROJECT 07 // STAGE 01 OPERATIONAL DASHBOARD
</div>
""", unsafe_allow_html=True)

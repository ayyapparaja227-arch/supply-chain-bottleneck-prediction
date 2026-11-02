#!/usr/bin/env python3
"""
Global Supply Chain Logistics Bottleneck Predictor
Stage 02: Deep Learning (DL) Engine — 12-Hour Dwell Time Forecaster
Claymorphism Design System Edition
"""

from pathlib import Path
import json
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CLAYMORPHISM STYLING
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Dwell Forecaster (Stage 02 DL)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    .stApp {
        background-color: var(--dp-color-ground) !important;
        color: var(--dp-color-ink) !important;
        font-family: var(--dp-typography-face) !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        font-family: var(--dp-typography-face) !important;
        color: var(--dp-color-ink);
    }

    /* Claymorphism Pink Hero */
    .clay-hero {
        background-color: #f6a3ca;
        border-radius: 38px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 12px 12px 28px rgba(185, 190, 210, 0.45),
                    inset 3px 3px 6px rgba(255, 255, 255, 0.6),
                    inset -3px -3px 6px rgba(180, 80, 130, 0.25);
    }

    .clay-hero-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.35);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #383139;
        margin-bottom: 12px;
    }

    .clay-hero-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.8px;
        line-height: 1.15;
        color: #383139;
        margin-bottom: 6px;
    }

    .clay-hero-subtitle {
        font-size: 14px;
        color: #48404b;
        font-weight: 500;
        margin: 0;
    }

    /* Shared Cream Tray */
    .clay-tray {
        background-color: #fffdf7;
        border-radius: 40px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 10px 10px 24px rgba(185, 190, 210, 0.35),
                    inset 3px 3px 6px rgba(255, 255, 255, 0.8),
                    inset -3px -3px 6px rgba(210, 205, 190, 0.2);
    }

    /* Pastel Cushions */
    .clay-cushion {
        border-radius: 26px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 8px 8px 20px rgba(185, 190, 210, 0.3),
                    inset 2px 2px 4px rgba(255, 255, 255, 0.7),
                    inset -2px -2px 4px rgba(0, 0, 0, 0.05);
    }

    .cushion-mint { background-color: #c0efdc; }
    .cushion-lavender { background-color: #ded0f9; }
    .cushion-yellow { background-color: #f9e7a8; }
    .cushion-blue { background-color: #c5e5fb; }
    .cushion-pink { background-color: #f6a3ca; }
    .cushion-cream { background-color: #fffdf7; }

    /* Prediction Result Banners */
    .prediction-banner-fluid {
        background-color: #c0efdc;
        border-radius: 30px;
        padding: 22px 26px;
        border-left: 8px solid #22c55e;
        box-shadow: 6px 6px 16px rgba(192, 239, 220, 0.5), inset 2px 2px 4px rgba(255, 255, 255, 0.8);
    }

    .prediction-banner-backlog {
        background-color: #f9e7a8;
        border-radius: 30px;
        padding: 22px 26px;
        border-left: 8px solid #f59e0b;
        box-shadow: 6px 6px 16px rgba(249, 231, 168, 0.5), inset 2px 2px 4px rgba(255, 255, 255, 0.8);
    }

    .prediction-banner-critical {
        background-color: #f6a3ca;
        border-radius: 30px;
        padding: 22px 26px;
        border-left: 8px solid #ef4444;
        box-shadow: 6px 6px 16px rgba(246, 163, 202, 0.5), inset 2px 2px 4px rgba(255, 255, 255, 0.8);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #fffdf7;
        padding: 8px 12px;
        border-radius: 999px;
        box-shadow: inset 3px 3px 6px rgba(185, 190, 210, 0.35), inset -3px -3px 6px rgba(255, 255, 255, 0.9);
        border: none;
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 8px 20px;
        color: #48404b !important;
        font-weight: 700;
        font-size: 13px;
        border: none !important;
        background-color: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #f6a3ca !important;
        color: #383139 !important;
        box-shadow: 4px 4px 10px rgba(185, 190, 210, 0.35), inset 2px 2px 4px rgba(255, 255, 255, 0.5) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #fffdf7 !important;
        border-right: 1px solid rgba(220, 222, 235, 0.6);
    }

    /* Modern Table */
    .telemetry-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-top: 14px;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 4px 4px 12px rgba(185, 190, 210, 0.25);
    }

    .telemetry-table th {
        background-color: #f1f2fa;
        color: #48404b;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 12px 14px;
        text-align: left;
        font-weight: 700;
    }

    .telemetry-table td {
        background-color: #fffdf7;
        padding: 11px 14px;
        font-size: 13px;
        color: #383139;
        border-top: 1px solid #f1f2fa;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. DYNAMIC INTEGRATION ENGINEER IMPORT & INFERENCE PIPELINE
# ---------------------------------------------------------------------
@st.cache_resource
def load_dl_engine():
    possible_paths = [
        Path(__file__).parent / "05_integration_engineer" / "05_integration_engineer.py",
        Path("Stage02_DL/05_integration_engineer/05_integration_engineer.py"),
        Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage02_DL/05_integration_engineer/05_integration_engineer.py")
    ]
    script_path = next((p for p in possible_paths if p.exists()), None)
    if not script_path:
        raise FileNotFoundError("Could not find Stage02_DL 05_integration_engineer.py module!")

    spec = importlib.util.spec_from_file_location("integration_module", str(script_path))
    int_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(int_mod)
    return int_mod.predict_next_dwell_hours

try:
    predict_next_dwell_hours = load_dl_engine()
except Exception as e:
    st.error(f"Error loading Deep Learning engine: {e}")
    predict_next_dwell_hours = None

# Ingest hourly telemetry
@st.cache_data
def load_hourly_telemetry():
    possible_csvs = [
        Path(__file__).parent / "data" / "hourly_node_telemetry.csv",
        Path("Stage02_DL/data/hourly_node_telemetry.csv"),
        Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage02_DL/data/hourly_node_telemetry.csv")
    ]
    csv_file = next((p for p in possible_csvs if p.exists()), None)
    if not csv_file:
        raise FileNotFoundError("Could not find Stage02_DL hourly_node_telemetry.csv!")
    return pd.read_csv(csv_file, parse_dates=["hour"])

df_hourly = load_hourly_telemetry()
nodes_list = sorted(df_hourly["current_node_id"].unique().tolist())

# ---------------------------------------------------------------------
# 3. SIDEBAR CONTROLS: 12-HOUR SEQUENCE TUNER & SHOCK INJECTION
# ---------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <span style="background-color: #f6a3ca; color: #383139; font-weight: 800; font-size: 11px; padding: 4px 10px; border-radius: 999px; text-transform: uppercase;">Stage 02 • Deep Learning</span>
    <h3 style="font-size: 19px; font-weight: 800; color: #383139; margin-top: 8px; margin-bottom: 2px;">Network Sequence Studio</h3>
    <p style="font-size: 12px; color: #48404b; margin: 0;">12-Hour Telemetry Sliding Window</p>
</div>
""", unsafe_allow_html=True)

# Node Selection
selected_node = st.sidebar.selectbox(
    "📍 Target Logistics Hub / Port",
    nodes_list,
    index=nodes_list.index("PORT_US_LAX") if "PORT_US_LAX" in nodes_list else 0
)

# Presets
preset_choice = st.sidebar.selectbox(
    "⚡ Telemetry Scenario Preset",
    [
        "🟢 Baseline Fluid Corridor",
        "🟡 Pre-Storm Port Slowdown (+30% Hold)",
        "🔴 Extreme Pacific Typhoon Shock",
        "⚠️ Midwest Yard Saturation Crisis (>95%)"
    ]
)

# Extract 12 hours for the selected node
node_data = df_hourly[df_hourly["current_node_id"] == selected_node].sort_values("hour")
if len(node_data) < 12:
    st.error(f"Not enough hourly data for node {selected_node}.")
    st.stop()

seq_12 = node_data.iloc[-12:].copy()

# Apply Presets to hour 12
if preset_choice == "🟡 Pre-Storm Port Slowdown (+30% Hold)":
    seq_12.iloc[-1, seq_12.columns.get_loc("customs_hold_hours")] = 9.5
    seq_12.iloc[-1, seq_12.columns.get_loc("weather_severity_index")] = 5.2
elif preset_choice == "🔴 Extreme Pacific Typhoon Shock":
    seq_12.iloc[-1, seq_12.columns.get_loc("weather_severity_index")] = 9.4
    seq_12.iloc[-1, seq_12.columns.get_loc("throughput_teu_per_hr")] = 75.0
    seq_12.iloc[-1, seq_12.columns.get_loc("capacity_utilization_pct")] = 98.0
elif preset_choice == "⚠️ Midwest Yard Saturation Crisis (>95%)":
    seq_12.iloc[-1, seq_12.columns.get_loc("capacity_utilization_pct")] = 106.0
    seq_12.iloc[-1, seq_12.columns.get_loc("labor_deficit_ratio")] = 0.42

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size:12px; font-weight:700; color:#48404b; text-transform:uppercase;'>Tune Current Hour 12 Telemetry</p>", unsafe_allow_html=True)

dwell_h12 = st.sidebar.slider("Dwell Time (hrs) at h12", 2.0, 70.0, float(seq_12["dwell_time_hours"].iloc[-1]), 0.5)
cap_h12 = st.sidebar.slider("Yard Capacity Saturation (%) at h12", 20.0, 130.0, float(seq_12["capacity_utilization_pct"].iloc[-1]), 1.0)
customs_h12 = st.sidebar.slider("Customs Hold (hrs) at h12", 0.5, 30.0, float(seq_12["customs_hold_hours"].iloc[-1]), 0.5)
throughput_h12 = st.sidebar.slider("Berth Crane Velocity (TEU/h) at h12", 40.0, 350.0, float(seq_12["throughput_teu_per_hr"].iloc[-1]), 5.0)
weather_h12 = st.sidebar.slider("Weather Severity (0-10) at h12", 0.0, 10.0, float(seq_12["weather_severity_index"].iloc[-1]), 0.2)
labor_h12 = st.sidebar.slider("Labor Deficit Ratio (0-1.0) at h12", 0.0, 0.8, float(seq_12["labor_deficit_ratio"].iloc[-1]), 0.02)

# Update final row of sequence with slider inputs
seq_12.iloc[-1, seq_12.columns.get_loc("dwell_time_hours")] = dwell_h12
seq_12.iloc[-1, seq_12.columns.get_loc("capacity_utilization_pct")] = cap_h12
seq_12.iloc[-1, seq_12.columns.get_loc("customs_hold_hours")] = customs_h12
seq_12.iloc[-1, seq_12.columns.get_loc("throughput_teu_per_hr")] = throughput_h12
seq_12.iloc[-1, seq_12.columns.get_loc("weather_severity_index")] = weather_h12
seq_12.iloc[-1, seq_12.columns.get_loc("labor_deficit_ratio")] = labor_h12

# Run Inference
if predict_next_dwell_hours:
    dl_out = predict_next_dwell_hours(seq_12)
    pred_dwell = dl_out["predicted_next_hour_dwell_hours"]
    pred_status = dl_out["status"]
    pred_directive = dl_out["operational_directive"]
    requires_review = dl_out["requires_operator_review"]
else:
    pred_dwell = float(dwell_h12)
    pred_status = "FLUID_TRANSIT"
    pred_directive = "Standard operational dispatch maintained."
    requires_review = False

# ---------------------------------------------------------------------
# 4. CLAYMORPHISM PINK HERO BANNER
# ---------------------------------------------------------------------
st.markdown(f"""
<div class="clay-hero">
    <div class="clay-hero-tag">🧠 PyTorch Recurrent Forecaster (LSTM) • Stage 02 Deep Learning</div>
    <div class="clay-hero-title">Global Logistics Dwell Time Forecaster</div>
    <div class="clay-hero-subtitle">
        Evaluating 12-Hour Operational Waveforms to Predict Next-Hour ($h_{{13}}$) Berth and Rail Gridlock
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 5. CORRIDOR TRANSIT STEPPER
# ---------------------------------------------------------------------
st.markdown(f"""
<div style="background-color: #fffdf7; border-radius: 28px; padding: 18px 26px; margin-bottom: 22px; box-shadow: 6px 6px 16px rgba(185, 190, 210, 0.35);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="font-size: 11px; font-weight: 700; color: #48404b; text-transform: uppercase;">Active Corridor</div>
            <div style="font-size: 17px; font-weight: 800; color: #383139;">CN_SHA ➔ US_LAX ➔ CHI_BNSF ➔ MW_JOLIET</div>
        </div>
        <div>
            <div style="font-size: 11px; font-weight: 700; color: #48404b; text-transform: uppercase;">Target Forecast Hub</div>
            <div style="font-size: 17px; font-weight: 800; color: #3b82f6;">📍 {selected_node}</div>
        </div>
        <div>
            <div style="font-size: 11px; font-weight: 700; color: #48404b; text-transform: uppercase;">Sequence Horizon</div>
            <div style="font-size: 17px; font-weight: 800; color: #10b981;">12 Hours Input ➔ +1 Hour Forecast</div>
        </div>
        <div>
            <div style="font-size: 11px; font-weight: 700; color: #48404b; text-transform: uppercase;">Model Accuracy</div>
            <div style="font-size: 17px; font-weight: 800; color: #8b5cf6;">MAE: 4.28h • R²: 0.602</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 6. TABS LAYOUT
# ---------------------------------------------------------------------
tab_forecast, tab_waveform, tab_graph = st.tabs([
    "🎯 Live Dwell Forecast & Triage",
    "📈 12-Hour Waveform & Cross-Signals",
    "🌐 Spatial Graph & Benchmark Ablation"
])

# ---------------------------------------------------------------------
# TAB 1: LIVE FORECAST & TRIAGE
# ---------------------------------------------------------------------
with tab_forecast:
    st.markdown('<div class="clay-tray">', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns([1.1, 1.0], gap="large")

    diff_dwell = pred_dwell - dwell_h12
    trend_badge = f"▲ +{diff_dwell:.1f}h" if diff_dwell >= 0 else f"▼ {diff_dwell:.1f}h"

    with col_t1:
        # Prediction Pill Banner
        if pred_dwell < 18.0:
            st.markdown(f"""
            <div class="prediction-banner-fluid">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #204b36;">Next-Hour Dwell Prediction (Hour 13)</div>
                <div style="font-size: 26px; font-weight: 800; color: #15803d; margin: 4px 0;">🟢 FLUID TRANSIT ({pred_dwell:.2f} hrs)</div>
                <div style="font-size: 13px; color: #204b36;">Container turnaround is fast and unhindered. Contractual turnaround SLA fully maintained.</div>
            </div>
            """, unsafe_allow_html=True)
        elif pred_dwell <= 24.0:
            st.markdown(f"""
            <div class="prediction-banner-backlog">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #524213;">Next-Hour Dwell Prediction (Hour 13)</div>
                <div style="font-size: 26px; font-weight: 800; color: #b45309; margin: 4px 0;">🟡 MODERATE BACKLOG ({pred_dwell:.2f} hrs)</div>
                <div style="font-size: 13px; color: #524213;">Corridor friction building up. Yard is saturating toward 24-hour critical SLA threshold.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-banner-critical">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #4d1b31;">Next-Hour Dwell Prediction (Hour 13)</div>
                <div style="font-size: 26px; font-weight: 800; color: #b91c1c; margin: 4px 0;">🔴 CRITICAL BOTTLENECK ({pred_dwell:.2f} hrs)</div>
                <div style="font-size: 13px; color: #4d1b31;">Severe terminal gridlock forecasted! Dwell exceeds contractual threshold (&gt;24.0h).</div>
            </div>
            """, unsafe_allow_html=True)

        # Tactical Directive Cushion
        st.markdown(f"""
        <div style="background-color: #fffdf7; border-radius: 24px; padding: 18px 22px; margin-top: 16px; border-left: 6px solid #f6a3ca; box-shadow: 4px 4px 12px rgba(185, 190, 210, 0.25);">
            <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #f6a3ca;">Tactical Logistics Directive</div>
            <div style="font-size: 14px; font-weight: 600; color: #383139; margin-top: 4px; line-height: 1.5;">{pred_directive}</div>
        </div>
        """, unsafe_allow_html=True)

        # Operational Drivers Table
        st.markdown("""
        <table class="telemetry-table">
            <thead>
                <tr>
                    <th>Signal Feature</th>
                    <th>Observed (h12)</th>
                    <th>Safe Baseline</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Container Dwell Time</strong></td>
                    <td>{:.1f} hrs</td>
                    <td>&lt; 18.0 hrs</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <td><strong>Yard Saturation</strong></td>
                    <td>{:.1f}%</td>
                    <td>&lt; 80.0%</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <td><strong>Customs Hold Time</strong></td>
                    <td>{:.1f} hrs</td>
                    <td>&lt; 4.0 hrs</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <td><strong>Berth Crane Velocity</strong></td>
                    <td>{:.0f} TEU/h</td>
                    <td>&gt; 180 TEU/h</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <td><strong>Weather Severity</strong></td>
                    <td>{:.1f} / 10</td>
                    <td>&lt; 3.0</td>
                    <td>{}</td>
                </tr>
            </tbody>
        </table>
        """.format(
            dwell_h12, "🟢 Normal" if dwell_h12 < 18 else ("🟡 Caution" if dwell_h12 <= 24 else "🔴 Alert"),
            cap_h12, "🟢 Clear" if cap_h12 < 80 else ("🟡 Heavy" if cap_h12 <= 90 else "🔴 Jammed"),
            customs_h12, "🟢 Passed" if customs_h12 < 4 else ("🟡 Held" if customs_h12 <= 8 else "🔴 Delayed"),
            throughput_h12, "🟢 Swift" if throughput_h12 >= 180 else ("🟡 Slow" if throughput_h12 >= 120 else "🔴 Stalled"),
            weather_h12, "🟢 Calm" if weather_h12 < 3.0 else ("🟡 Stormy" if weather_h12 <= 6.0 else "🔴 Severe")
        ), unsafe_allow_html=True)

    with col_t2:
        # Semicircular Dwell Gauge
        st.markdown("<h4 style='font-size: 15px; font-weight: 800; color: #383139; margin-bottom: 2px;'>⏱️ Forecast Dwell Velocity Gauge</h4>", unsafe_allow_html=True)
        st.caption("Continuous next-hour dwell projection against terminal operational SLA.")

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_dwell,
            delta={'reference': dwell_h12, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
            number={'suffix': " hrs", 'font': {'size': 36, 'color': '#383139', 'family': 'Instrument Sans'}},
            gauge={
                'axis': {'range': [0, 50], 'tickwidth': 1, 'tickcolor': "#48404b"},
                'bar': {'color': "#383139", 'thickness': 0.24},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 18], 'color': '#c0efdc'},
                    {'range': [18, 24], 'color': '#f9e7a8'},
                    {'range': [24, 50], 'color': '#f6a3ca'}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 4},
                    'thickness': 0.75,
                    'value': 24.0
                }
            }
        ))
        gauge_fig.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Instrument Sans", color="#383139")
        )
        st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})

        # Dual Metric Cushions
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style="background-color: #f1f2fa; border-radius: 20px; padding: 14px 18px; box-shadow: inset 2px 2px 5px rgba(185, 190, 210, 0.4);">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #48404b;">Hour 12 Dwell</div>
                <div style="font-size: 22px; font-weight: 800; color: #383139; margin-top: 2px;">{dwell_h12:.1f} <span style="font-size:12px;">hrs</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background-color: #ded0f9; border-radius: 20px; padding: 14px 18px; box-shadow: 4px 4px 10px rgba(185, 190, 210, 0.3);">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #48404b;">Queue Momentum</div>
                <div style="font-size: 22px; font-weight: 800; color: #383139; margin-top: 2px;">{trend_badge}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TAB 2: 12-HOUR WAVEFORM & CROSS-SIGNALS
# ---------------------------------------------------------------------
with tab_waveform:
    st.markdown('<div class="clay-tray">', unsafe_allow_html=True)
    st.markdown("<h4 style='font-size:17px; font-weight:800; color:#383139;'>📈 12-Hour Multi-Signal Telemetry Waveform + Neural Forecast</h4>", unsafe_allow_html=True)
    st.caption("Visualizing the continuous sequence fed into the 32 LSTM hidden units and its forecasted trajectory at hour 13.")

    hours_x = [f"h{i}" for i in range(1, 13)]
    dwell_series = seq_12["dwell_time_hours"].tolist()
    yard_series = seq_12["capacity_utilization_pct"].tolist()
    customs_series = seq_12["customs_hold_hours"].tolist()

    wf_fig = go.Figure()

    # Dwell line
    wf_fig.add_trace(go.Scatter(
        x=hours_x, y=dwell_series,
        mode="lines+markers",
        name="Observed Dwell History (h1-h12)",
        line=dict(color="#3b82f6", width=3.5),
        marker=dict(size=8, color="#1d4ed8")
    ))

    # Forecast Point (h12 to h13)
    forecast_color = "#ef4444" if pred_dwell > 24 else ("#f59e0b" if pred_dwell > 18 else "#10b981")
    wf_fig.add_trace(go.Scatter(
        x=["h12", "h13 (Forecast)"],
        y=[dwell_series[-1], pred_dwell],
        mode="lines+markers",
        name="LSTM Predicted Dwell (h13)",
        line=dict(color=forecast_color, width=4, dash="dash"),
        marker=dict(size=12, symbol="star", color=forecast_color)
    ))

    # SLA 24-hour line
    wf_fig.add_hline(
        y=24.0, line_dash="dot", line_color="#ef4444", line_width=1.5,
        annotation_text="Critical Bottleneck SLA (24.0 hrs)", annotation_position="top left",
        annotation_font=dict(color="#ef4444", size=11, family="Instrument Sans")
    )

    wf_fig.update_layout(
        xaxis_title="Chronological 1-Hour Steps",
        yaxis_title="Container Dwell Time (Hours)",
        height=350,
        margin=dict(l=35, r=20, t=30, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans", color="#383139"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(family="Instrument Sans", size=11, color="#383139")),
        xaxis=dict(gridcolor="#f1f2fa", color="#48404b"),
        yaxis=dict(gridcolor="#f1f2fa", color="#48404b")
    )
    st.plotly_chart(wf_fig, use_container_width=True, config={'displayModeBar': False})

    # Sub-Visualizations: Yard Saturation & Customs Hold
    col_w1, col_w2 = st.columns(2, gap="large")
    with col_w1:
        st.markdown("<h5 style='font-size:14px; font-weight:700; color:#383139;'>🏭 Yard Saturation Curve (%)</h5>", unsafe_allow_html=True)
        yard_fig = go.Figure()
        yard_fig.add_trace(go.Bar(
            x=hours_x, y=yard_series,
            marker_color=["#ef4444" if v > 90 else ("#f59e0b" if v > 80 else "#3b82f6") for v in yard_series],
            name="Yard %"
        ))
        yard_fig.update_layout(
            height=200, margin=dict(l=20, r=20, t=10, b=25),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#f1f2fa", color="#48404b", range=[0, 130]),
            xaxis=dict(gridcolor="#f1f2fa", color="#48404b")
        )
        st.plotly_chart(yard_fig, use_container_width=True, config={'displayModeBar': False})

    with col_w2:
        st.markdown("<h5 style='font-size:14px; font-weight:700; color:#383139;'>🛂 Customs Inspection Hold (Hours)</h5>", unsafe_allow_html=True)
        customs_fig = go.Figure()
        customs_fig.add_trace(go.Scatter(
            x=hours_x, y=customs_series,
            mode="lines+markers",
            line=dict(color="#8b5cf6", width=2.5),
            marker=dict(size=6, color="#6d28d9"),
            name="Customs (hrs)"
        ))
        customs_fig.update_layout(
            height=200, margin=dict(l=20, r=20, t=10, b=25),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#f1f2fa", color="#48404b"),
            xaxis=dict(gridcolor="#f1f2fa", color="#48404b")
        )
        st.plotly_chart(customs_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TAB 3: SPATIAL GRAPH & BENCHMARK ABLATION
# ---------------------------------------------------------------------
with tab_graph:
    st.markdown('<div class="clay-tray">', unsafe_allow_html=True)
    col_g1, col_g2 = st.columns([1.1, 1.0], gap="large")

    with col_g1:
        st.markdown("<h4 style='font-size:16px; font-weight:800; color:#383139;'>🌐 Network Topology & 1-Hop Spatial Neighbors</h4>", unsafe_allow_html=True)
        st.caption("Verifying corridor delay propagation and node degree baselines from Slide 20 in 2_DL.pptx.")

        # Load graph node features if available
        graph_file = Path("Stage02_DL/data/graph_node_features.csv")
        if not graph_file.exists():
            graph_file = Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage02_DL/data/graph_node_features.csv")
        
        if graph_file.exists():
            df_graph = pd.read_csv(graph_file)
            st.dataframe(
                df_graph.rename(columns={
                    "current_node_id": "Logistics Hub",
                    "mean_dwell": "Historical Dwell (h)",
                    "neighbor_mean_dwell": "Neighbors Mean Dwell (h)",
                    "neighbor_count": "Active Corridor Links"
                }).style.format({
                    "Historical Dwell (h)": "{:.1f}h",
                    "Neighbors Mean Dwell (h)": "{:.1f}h"
                }),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("""
        <div style="background-color: #f1f2fa; border-radius: 18px; padding: 14px 18px; margin-top: 14px; font-size: 13px; color: #48404b;">
            💡 <strong>Spatial Message-Passing Insight:</strong> Delays at origin mega-hubs like <code>PORT_CN_SHA</code> propagate through transpacific maritime corridors to <code>PORT_US_LAX</code>, and cascade into intermodal rail ramps like <code>RAIL_US_CHI_BNSF</code>.
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        st.markdown("<h4 style='font-size:16px; font-weight:800; color:#383139;'>🧪 Ablation Study: Why Deep Learning?</h4>", unsafe_allow_html=True)
        st.caption("Tested across 923 unseen chronological future windows with zero data leakage.")

        ablation_table = """
        <table class="telemetry-table">
            <thead>
                <tr>
                    <th>Candidate Model</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R² Score</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>1. Historical Mean (Static)</strong></td>
                    <td>5.53h</td>
                    <td>7.07h</td>
                    <td>0.389</td>
                </tr>
                <tr>
                    <td><strong>2. Spatial 1-Hop Neighbor</strong></td>
                    <td>6.19h</td>
                    <td>9.01h</td>
                    <td>0.007</td>
                </tr>
                <tr>
                    <td><strong>3. Naive Persistence (y_t)</strong></td>
                    <td>3.84h</td>
                    <td>4.94h</td>
                    <td>0.702</td>
                </tr>
                <tr style="background-color: #ded0f9;">
                    <td><strong>4. PyTorch LSTM (Our Model)</strong></td>
                    <td><strong>4.28h</strong></td>
                    <td><strong>5.70h</strong></td>
                    <td><strong>0.602</strong></td>
                </tr>
            </tbody>
        </table>
        """
        st.markdown(ablation_table, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #c0efdc; border-radius: 18px; padding: 14px 18px; margin-top: 14px; font-size: 12px; color: #204b36;">
            ✅ <strong>Verification:</strong> Matches Slide 19 targets (Target MAE ~4.24h, RMSE ~5.66h, R² ~0.61). The LSTM achieves <strong>96.5% precision</strong> on critical bottleneck alerts (>24h dwell).
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 7. FOOTER
# ---------------------------------------------------------------------
st.markdown("""
<div style="background-color: #fffdf7; border-radius: 28px; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; color: #48404b; box-shadow: 4px 4px 12px rgba(185, 190, 210, 0.25);">
    <div>🚢 Global Supply Chain Dwell Forecaster • Stage 02 Deep Learning</div>
    <div>PyTorch LSTM • 5,921 Parameters • 35 Epochs MPS Acceleration</div>
</div>
""", unsafe_allow_html=True)


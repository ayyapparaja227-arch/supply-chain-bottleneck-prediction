#!/usr/bin/env python3
"""
Interactive Live Demonstration of Stage 02 Deep Learning Forecaster
Runs real 12-hour historical sequences from global shipping hubs through the trained PyTorch LSTM.
"""

from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np
import torch

# Dynamically import integration engineer module
int_file = Path("Stage02_DL/05_integration_engineer/05_integration_engineer.py")
if not int_file.exists():
    int_file = Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage02_DL/05_integration_engineer/05_integration_engineer.py")

spec = importlib.util.spec_from_file_location("integration_module", str(int_file))
int_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(int_mod)
predict_next_dwell_hours = int_mod.predict_next_dwell_hours

def main():
    print("=" * 80)
    print("🚢 STAGE 02 (DEEP LEARNING) — LIVE MODEL INFERENCE DEMO")
    print("   Architecture: PyTorch LSTM Forecaster (8 Signals, 12-Hour Sliding Window)")
    print("=" * 80)

    data_dir = Path("Stage02_DL/data")
    hourly_file = data_dir / "hourly_node_telemetry.csv"
    if not hourly_file.exists():
        hourly_file = Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage02_DL/data/hourly_node_telemetry.csv")

    df = pd.read_csv(hourly_file, parse_dates=["hour"])

    test_scenarios = [
        ("PORT_US_LAX", "Port of Los Angeles (Major Pacific Gateway)"),
        ("PORT_CN_SHA", "Port of Shanghai (Origin Mega-Hub)"),
        ("RAIL_US_CHI_BNSF", "Chicago BNSF Rail Ramp (Intermodal Chokepoint)"),
        ("DC_US_MW_JOLIET", "Midwest DC Fulfillment Hub (Final Mile Inbound)")
    ]

    for node_id, description in test_scenarios:
        node_df = df[df["current_node_id"] == node_id].sort_values("hour")
        if len(node_df) < 12:
            continue

        # Take the most recent 12 hours of real telemetry
        recent_12h = node_df.iloc[-12:]
        history_dwell = recent_12h["dwell_time_hours"].tolist()
        last_hour_time = recent_12h["hour"].iloc[-1]
        
        # Run through trained PyTorch Deep Learning model
        res = predict_next_dwell_hours(recent_12h)
        pred_dwell = res["predicted_next_hour_dwell_hours"]
        status = res["status"]
        directive = res["operational_directive"]

        print(f"\n📍 HUB: {node_id} — {description}")
        print(f"   • Observation Cutoff: {last_hour_time}")
        print("   • 12-Hour Dwell History (h1 -> h12):")
        dwell_history_str = " -> ".join([f"{h:.1f}h" for h in history_dwell])
        print(f"     [{dwell_history_str}]")
        print(f"   • Current Hour 12 Dwell: {history_dwell[-1]:.1f} hours")
        print(f"   ------------------------------------------------------------")
        print(f"   🎯 DEEP LEARNING PREDICTION FOR NEXT HOUR (h13):")
        print(f"      ▶ Forecasted Dwell:  {pred_dwell:.2f} hours")
        delta = pred_dwell - history_dwell[-1]
        trend_arrow = "▲ INCREASING (+{:.2f}h)".format(delta) if delta > 0 else "▼ DECREASING ({:.2f}h)".format(delta)
        print(f"      ▶ Congestion Trend:  {trend_arrow}")
        print(f"      ▶ Congestion Status: {status}")
        print(f"      ▶ Tactical Action:   {directive}")
        print("-" * 80)

if __name__ == "__main__":
    main()

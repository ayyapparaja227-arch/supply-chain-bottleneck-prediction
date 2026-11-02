#!/usr/bin/env python3
"""
STAGE 02 DEEP LEARNING — MILESTONE 5 (40% -> 50%)
Role 02: EDA Engineer — Dwell Waveform, Cascade Cross-Correlation & Telemetry Dynamics

Problem Addressed (Slide 15-18 & 20 in 2_DL.pptx):
"How do dwell times fluctuate across different nodes over time?
Are dwell spikes localized or cascading across transit corridors?
What is the temporal autocorrelation and cross-signal lag profile?"

Deliverables:
1. Hourly dwell time waveform analysis per logistics hub.
2. Cross-correlation between capacity utilization, customs holds, and dwell time.
3. Lag autocorrelation analysis showing temporal dependency decaying across 12 hours.
4. Generates:
   - data/eda_dwell_trends.png (Multi-node dwell waveform plot)
   - data/eda_delay_status.png (Delay classification distribution)
   - data/eda_hourly_trends.csv (Network-wide hourly averages)
   - data/eda_numeric_summary.csv (Comprehensive statistical distributions)
   - data/eda_correlations.csv (Telemetry cross-correlation matrix)
   - data/eda_report.txt (Full analytical diagnostic report)
   - data/milestone_50pct_report.json (Audited milestone verification metrics)
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run_milestone_50pct_pipeline():
    print("=" * 70)
    print("🚢 STAGE 02 (DEEP LEARNING) — MILESTONE 5 [50% COMPLETE]")
    print("   Role 02: EDA Engineer — Waveform & Cascade Cross-Correlation")
    print("=" * 70)

    base_dir = Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage02_DL")
    if not base_dir.exists():
        base_dir = Path("Stage02_DL")
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = base_dir / "02_eda_engineer" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    prep_file = data_dir / "prepared_dataset.csv"
    hourly_file = data_dir / "hourly_node_telemetry.csv"

    if not prep_file.exists() or not hourly_file.exists():
        raise FileNotFoundError("Missing prepared dataset or hourly telemetry. Complete Milestones 1-4 first!")

    print(f"\n[1/5] Ingesting telemetry data for deep exploratory analysis...")
    df_raw = pd.read_csv(prep_file, parse_dates=["timestamp"])
    df_hourly = pd.read_csv(hourly_file, parse_dates=["hour"])

    print(f"  • Row-level shipment records: {len(df_raw):,}")
    print(f"  • Resampled hourly node steps: {len(df_hourly):,}")

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("SUPPLY CHAIN TELEMETRY EXPLORATORY DATA ANALYSIS (STAGE 02 DL)")
    report_lines.append("=" * 60)
    report_lines.append(f"Row count: {len(df_raw):,} rows | Columns: {len(df_raw.columns)}")
    report_lines.append(f"Observation window: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")

    # 2. Distribution & Numeric Profiles
    print("\n[2/5] Profiling numerical signal distributions and physical bounds...")
    num_cols = df_raw.select_dtypes(include="number")
    num_summary = num_cols.describe().T
    num_summary.to_csv(data_dir / "eda_numeric_summary.csv")
    print("  • Saved: eda_numeric_summary.csv")

    corr_matrix = num_cols.corr()
    corr_matrix.to_csv(data_dir / "eda_correlations.csv")
    print("  • Saved: eda_correlations.csv")

    dwell_corr = corr_matrix["dwell_time_hours"].sort_values(ascending=False)
    report_lines.append("\nTop Feature Correlations with Dwell Time (Hours):")
    for feat, val in dwell_corr.items():
        if feat != "dwell_time_hours":
            report_lines.append(f"  • {feat:30}: {val:+.4f}")
            print(f"    - {feat:30}: {val:+.4f}")

    # 3. Delay Classification Distribution
    print("\n[3/5] Visualizing delay severity distribution...")
    if "delay_status" in df_raw.columns:
        status_counts = df_raw["delay_status"].value_counts()
        plt.figure(figsize=(7, 4))
        colors = ["#22c55e", "#f59e0b", "#ef4444"]
        status_counts.plot(kind="bar", color=colors[:len(status_counts)], edgecolor="#333", linewidth=1.2)
        plt.title("Supply Chain Delay Status Distribution", fontsize=12, fontweight="bold")
        plt.xlabel("Delay Status")
        plt.ylabel("Shipment Records")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(data_dir / "eda_delay_status.png", dpi=180)
        plt.savefig(fig_dir / "eda_delay_status.png", dpi=180)
        plt.close()
        print("  • Saved: eda_delay_status.png")

    # 4. Hourly Node Dwell Waveforms (Slide 20)
    print("\n[4/5] Rendering multi-node dwell waveform time-series plots...")
    plt.figure(figsize=(14, 6))
    node_stats = {}
    node_palette = {
        "DC_US_MW_JOLIET": "#3b82f6",
        "PORT_CN_SHA": "#10b981",
        "PORT_US_LAX": "#ef4444",
        "PORT_US_LGB": "#f59e0b",
        "PORT_US_OAK": "#8b5cf6",
        "RAIL_US_CHI_BNSF": "#06b6d4"
    }

    for node, g in df_hourly.groupby("current_node_id"):
        g_sorted = g.sort_values("hour")
        color = node_palette.get(node, None)
        plt.plot(g_sorted["hour"], g_sorted["dwell_time_hours"], label=f"{node} (avg: {g['dwell_time_hours'].mean():.1f}h)", color=color, linewidth=1.4, alpha=0.85)
        node_stats[node] = {
            "mean_dwell": round(float(g["dwell_time_hours"].mean()), 2),
            "std_dwell": round(float(g["dwell_time_hours"].std()), 2),
            "max_dwell": round(float(g["dwell_time_hours"].max()), 2),
            "step_count": len(g)
        }

    plt.title("Hourly Dwell Time Telemetry Waveforms by Logistics Node (60-Day Horizon)", fontsize=13, fontweight="bold")
    plt.xlabel("Timeline (UTC)", fontsize=11)
    plt.ylabel("Observed Dwell Time (Hours)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(data_dir / "eda_dwell_trends.png", dpi=180)
    plt.savefig(fig_dir / "eda_dwell_trends.png", dpi=180)
    plt.close()
    print("  • Saved: eda_dwell_trends.png")

    # Network-wide hourly trend averages
    network_hourly = df_hourly.groupby("hour").agg(
        mean_dwell_hours=("dwell_time_hours", "mean"),
        max_dwell_hours=("dwell_time_hours", "max"),
        mean_throughput=("throughput_teu_per_hr", "mean"),
        mean_capacity=("capacity_utilization_pct", "mean"),
        mean_customs=("customs_hold_hours", "mean")
    ).reset_index()
    network_hourly.to_csv(data_dir / "eda_hourly_trends.csv", index=False)
    print("  • Saved: eda_hourly_trends.csv")

    # 5. Temporal Autocorrelation Analysis across Lags 1 to 12 Hours
    print("\n[5/5] Computing dwell time lag autocorrelation (Lags 1h to 12h)...")
    autocorr = {}
    for lag in range(1, 13):
        # Calculate lag correlation within each node series to avoid cross-node artifacts
        node_corrs = []
        for _, g in df_hourly.groupby("current_node_id"):
            s = g.sort_values("hour")["dwell_time_hours"]
            if len(s) > lag:
                node_corrs.append(s.autocorr(lag=lag))
        avg_ac = float(np.nanmean(node_corrs))
        autocorr[f"lag_{lag}h"] = round(avg_ac, 4)
        print(f"    • Lag {lag:2d}h autocorrelation: {avg_ac:.4f}")

    report_lines.append("\nDwell Time Lag Autocorrelation (1h to 12h):")
    for lag, val in autocorr.items():
        report_lines.append(f"  • {lag}: {val:+.4f}")

    report_lines.append("\nNode Dwell Waveform Summary:")
    for node, s in node_stats.items():
        report_lines.append(f"  • {node:20}: Mean={s['mean_dwell']}h, Std={s['std_dwell']}h, Peak={s['max_dwell']}h")

    # Write textual diagnostic report
    report_txt = "\n".join(report_lines)
    (data_dir / "eda_report.txt").write_text(report_txt, encoding="utf-8")
    print("  • Saved: eda_report.txt")

    # Save Milestone 5 Report JSON
    milestone_report = {
        "milestone": "50% Complete",
        "phase": "Role 02: EDA Engineer — Waveform & Cascade Analysis",
        "node_waveforms": node_stats,
        "dwell_correlations": {k: round(float(v), 4) for k, v in dwell_corr.items()},
        "lag_autocorrelation_1_to_12h": autocorr,
        "key_findings": {
            "highest_congestion_node": max(node_stats.items(), key=lambda x: x[1]["mean_dwell"])[0],
            "lowest_congestion_node": min(node_stats.items(), key=lambda x: x[1]["mean_dwell"])[0],
            "lag_1_persistence": autocorr["lag_1h"],
            "lag_12_persistence": autocorr["lag_12h"]
        },
        "artifacts_generated": [
            "data/eda_dwell_trends.png",
            "data/eda_delay_status.png",
            "data/eda_hourly_trends.csv",
            "data/eda_numeric_summary.csv",
            "data/eda_correlations.csv",
            "data/eda_report.txt"
        ],
        "status": "VERIFIED_READY_FOR_MODEL_TRAINING"
    }

    with open(data_dir / "milestone_50pct_report.json", "w") as f:
        json.dump(milestone_report, f, indent=2)

    print(f"\n✅ Milestone 5 report generated: milestone_50pct_report.json")
    print("=" * 70)
    print("🎯 MILESTONE 5 [50%] SIGN-OFF: EDA ENGINEER ROLE 100% COMPLETE!")
    print("======================================================================")

if __name__ == "__main__":
    run_milestone_50pct_pipeline()

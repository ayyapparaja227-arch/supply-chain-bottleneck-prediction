"""ROLE 2 — EDA ENGINEER: quality, drift, time coverage, and disruption patterns."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA = Path("data") if Path("data").exists() else Path("../data"); df=pd.read_csv(DATA/"prepared_dataset.csv",parse_dates=["timestamp"])
report=[]
report.append(f"Rows={len(df)}; columns={len(df.columns)}")
report.append(f"Time range={df.timestamp.min()} to {df.timestamp.max()}")
report.append("Missing values:\n"+df.isna().sum().to_string())
report.append("Duplicate rows="+str(int(df.duplicated().sum())))
for c,lo,hi in [("dwell_time_hours",0,None),("capacity_utilization_pct",0,100),("equipment_availability_pct",0,100),("labor_deficit_ratio",0,1)]:
    if c in df:
        bad=(df[c]<lo) if hi is None else ((df[c]<lo)|(df[c]>hi))
        report.append(f"Range review {c}: {int(bad.sum())} suspicious rows")
if "delay_status" in df:
    dist=df.delay_status.value_counts(dropna=False)
    report.append("Delay status counts:\n"+dist.to_string())
    dist.plot(kind="bar",title="Shipment delay status"); plt.tight_layout(); plt.savefig(DATA/"eda_delay_status.png",dpi=150); plt.close()
nums=df.select_dtypes(include="number")
nums.describe().T.to_csv(DATA/"eda_numeric_summary.csv")
nums.corr().to_csv(DATA/"eda_correlations.csv")
hourly=pd.read_csv(DATA/"hourly_node_telemetry.csv",parse_dates=["hour"])
if not hourly.empty:
    hourly.groupby("hour").agg(mean_dwell_hours=("dwell_time_hours","mean"),mean_throughput=("throughput_teu_per_hr","mean")).to_csv(DATA/"eda_hourly_trends.csv")
    for node,g in hourly.groupby("current_node_id"):
        plt.plot(g.hour,g.dwell_time_hours,label=node)
    plt.title("Dwell time over time by logistics node"); plt.xlabel("Time"); plt.ylabel("Mean dwell time (hours)"); plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(DATA/"eda_dwell_trends.png",dpi=150); plt.close()
(DATA/"eda_report.txt").write_text("\n\n".join(report),encoding="utf-8")
print("\n\n".join(report)); print("EDA artifacts saved under data/")

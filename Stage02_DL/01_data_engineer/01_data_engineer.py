"""ROLE 1 — DATA ENGINEER: validate raw telemetry and build hourly node sequences."""
from pathlib import Path
import pandas as pd
DATA = Path("data") if Path("data").exists() else Path("../data"); DATA.mkdir(exist_ok=True)
raw=pd.read_csv(DATA/"shipment_telemetry_raw.csv")
raw.columns=[c.strip().lower() for c in raw.columns]
required=["timestamp","current_node_id","dwell_time_hours","throughput_teu_per_hr","capacity_utilization_pct","customs_hold_hours","weather_severity_index","labor_deficit_ratio","equipment_availability_pct","historical_delay_rate"]
missing=[c for c in required if c not in raw.columns]
if missing: raise ValueError(f"Required columns missing: {missing}")
raw["timestamp"]=pd.to_datetime(raw["timestamp"],errors="coerce",utc=True)
raw=raw.dropna(subset=["timestamp","current_node_id"]).drop_duplicates().sort_values("timestamp")
num=required[2:]
for c in num: raw[c]=pd.to_numeric(raw[c],errors="coerce")
# Preserve cleaned row-level data for audit and build hourly aggregates for sequence learning.
raw.to_csv(DATA/"prepared_dataset.csv",index=False)
raw["hour"] = raw["timestamp"].dt.floor("h")
agg={c:"mean" for c in num}; agg["dwell_time_hours"]="mean"
hourly=raw.groupby(["current_node_id","hour"],as_index=False).agg(agg).sort_values(["current_node_id","hour"])
hourly.to_csv(DATA/"hourly_node_telemetry.csv",index=False)
# Edges are derived only from observed origin/current/destination relationships.
edge_rows=[]
for _,r in raw[["origin_port","current_node_id","destination_port"]].dropna().drop_duplicates().iterrows():
    nodes=[r["origin_port"],r["current_node_id"],r["destination_port"]]
    for a,b in zip(nodes,nodes[1:]):
        if a!=b: edge_rows.append((a,b))
edges=pd.DataFrame(sorted(set(edge_rows)),columns=["source","target"])
edges.to_csv(DATA/"logistics_edges.csv",index=False)
print(f"Raw cleaned data: {raw.shape}; hourly node sequence rows: {hourly.shape}; network edges: {len(edges)}")
print("Saved prepared_dataset.csv, hourly_node_telemetry.csv, logistics_edges.csv")

#!/usr/bin/env python3
"""
STAGE 02 DEEP LEARNING — MILESTONE 3 (20% -> 30%)
Role 01: Data Engineer — Logistics Graph Topology & Edge Extraction

Problem Addressed (Slide 20 of PDF):
"The Network Part: Graph Features: A Baseline, Not a GNN.
A node averages its neighbours' dwell times. That is message passing, not a trained GNN."

Deliverables:
1. Ingests raw multi-carrier shipment corridor routes (origin -> current_node -> destination).
2. Derives physical network adjacency graph (logistics_edges.csv).
3. Computes neighbor connections and degree connectivity across the 6 hubs.
4. Performs 1-hop spatial message passing (neighbor_mean_dwell) to capture cascading network congestion.
5. Saves:
   - data/logistics_edges.csv
   - data/graph_node_features.csv
   - data/milestone_30pct_report.json
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

def run_milestone_30pct_pipeline():
    print("=" * 70)
    print("🚢 STAGE 02 (DEEP LEARNING) — MILESTONE 3 [30% COMPLETE]")
    print("   Role 01: Data Engineer — Logistics Graph Topology & Edge Extraction")
    print("=" * 70)

    # Resolve Data Directory
    base_dir = Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage02_DL")
    if not base_dir.exists():
        base_dir = Path("Stage02_DL")
    data_dir = base_dir / "data"

    prep_csv = data_dir / "prepared_dataset.csv"
    if not prep_csv.exists():
        prep_csv = data_dir / "shipment_telemetry_raw.csv"

    print(f"\n[1/3] Ingesting multi-node shipment corridor telemetry from: {prep_csv.name}")
    raw = pd.read_csv(prep_csv)

    # 1. Derive Edge Connections
    print("\n[2/3] Deriving network topology from multi-hop transit paths...")
    edge_rows = []
    route_cols = ["origin_port", "current_node_id", "destination_port"]
    available_cols = [c for c in route_cols if c in raw.columns]

    for _, r in raw[available_cols].dropna().drop_duplicates().iterrows():
        nodes = [r[c] for c in available_cols]
        for a, b in zip(nodes, nodes[1:]):
            if a != b:
                edge_rows.append((a, b))

    edges_df = pd.DataFrame(sorted(set(edge_rows)), columns=["source", "target"])
    edges_csv = data_dir / "logistics_edges.csv"
    edges_df.to_csv(edges_csv, index=False)
    print(f"  • Derived {len(edges_df)} unique corridor transit edges -> {edges_csv.name}")

    # 2. Build Adjacency & Message Passing Baseline
    print("\n[3/3] Performing 1-hop spatial message-passing over network hubs...")
    node_stats = raw.groupby("current_node_id").agg(
        mean_dwell=("dwell_time_hours", "mean"),
        mean_capacity=("capacity_utilization_pct", "mean"),
        mean_throughput=("throughput_teu_per_hr", "mean")
    ).reset_index()

    dwell_lookup = {r.current_node_id: r.mean_dwell for r in node_stats.itertuples()}
    neighbors = {n: [] for n in dwell_lookup}

    for e in edges_df.itertuples():
        if e.source in neighbors and e.target in neighbors:
            neighbors[e.source].append(e.target)
            neighbors[e.target].append(e.source)

    # Deduplicate neighbor lists
    for k in neighbors:
        neighbors[k] = sorted(list(set(neighbors[k])))

    node_stats["neighbor_mean_dwell"] = [
        float(np.mean([dwell_lookup[n] for n in neighbors.get(k, [])])) if neighbors.get(k) else np.nan
        for k in node_stats.current_node_id
    ]
    node_stats["neighbor_count"] = [len(neighbors.get(k, [])) for k in node_stats.current_node_id]

    graph_csv = data_dir / "graph_node_features.csv"
    node_stats.to_csv(graph_csv, index=False)
    print(f"  • Saved spatial graph node features -> {graph_csv.name}\n")

    print("Network Graph Verification (Verifying Slide 20 in PDF):")
    print("-" * 75)
    print(f"{'Logistics Hub':20} | {'Mean Dwell (h)':14} | {'Neighbors Mean Dwell':20} | {'Links':5}")
    print("-" * 75)

    graph_summary = {}
    for r in node_stats.itertuples():
        hub = r.current_node_id
        d_mean = round(r.mean_dwell, 1)
        n_mean = round(r.neighbor_mean_dwell, 1)
        links = r.neighbor_count
        graph_summary[hub] = {
            "mean_dwell_hours": d_mean,
            "neighbor_mean_dwell_hours": n_mean,
            "links_degree": links,
            "connected_neighbors": neighbors.get(hub, [])
        }
        print(f"📍 {hub:18} | {d_mean:10.1f} h    | {n_mean:14.1f} h          | {links:2d} links")
    print("-" * 75)

    # Generate Milestone 3 Report
    report = {
        "milestone": "30% Complete",
        "phase": "Role 01: Logistics Graph Topology & Edge Extraction",
        "total_network_edges": len(edges_df),
        "total_active_nodes": len(node_stats),
        "network_topology": graph_summary,
        "message_passing_type": "1-hop mean neighbor aggregation",
        "artifacts_generated": [
            str(edges_csv.name),
            str(graph_csv.name)
        ],
        "status": "APPROVED — READY FOR 40% MILESTONE (ROLE 01: 12-HOUR SLIDING WINDOW SEQUENCE TENSORS)"
    }

    report_file = data_dir / "milestone_30pct_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Milestone 3 report generated: {report_file}")
    print("\n" + "=" * 70)
    print("🎯 MILESTONE 3 [30%] SIGN-OFF: LOGISTICS GRAPH TOPOLOGY COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    run_milestone_30pct_pipeline()


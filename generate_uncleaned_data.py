#!/usr/bin/env python3
"""
generate_uncleaned_data.py
Creates an authentic 'uncleaned' version of the shipment telemetry data
simulating real-world port IoT sensor dropouts, carrier log discrepancies,
RFID double-scans, out-of-order telemetry, and stochastic real-world delay noise.
"""

import pandas as pd
import numpy as np

def inject_real_world_noise(input_path: str, output_path: str, seed: int = 42):
    np.random.seed(seed)
    df = pd.read_csv(input_path)
    n = len(df)
    print(f"Original clean dataset loaded: {n} rows, {df.shape[1]} columns.")

    # 1. Inject Missing Values (Sensor dropouts & network packet loss)
    missing_rates = {
        'dwell_time_hours': 0.05,            # 5% missing GPS/yard RFID pings
        'throughput_teu_per_hr': 0.04,       # 4% crane telemetry dropouts
        'capacity_utilization_pct': 0.06,    # 6% terminal yard feed lags
        'customs_hold_hours': 0.08,          # 8% unrecorded customs holds
        'weather_severity_index': 0.03,      # 3% meteorology feed missing
        'labor_deficit_ratio': 0.05,         # 5% shift report missing
        'equipment_availability_pct': 0.04   # 4% maintenance log lag
    }
    for col, rate in missing_rates.items():
        mask = np.random.rand(n) < rate
        df.loc[mask, col] = np.nan

    # 2. Inject Sensor Glitches & Negative / Extreme Outliers
    glitch_idx_1 = np.random.choice(n, size=int(n * 0.015), replace=False)
    df.loc[glitch_idx_1, 'dwell_time_hours'] = -99.0  # Common IoT sensor reset error code

    glitch_idx_2 = np.random.choice(n, size=int(n * 0.01), replace=False)
    df.loc[glitch_idx_2, 'capacity_utilization_pct'] = np.random.uniform(115.0, 180.0, size=len(glitch_idx_2)) # False sensor readings > 100%

    glitch_idx_3 = np.random.choice(n, size=int(n * 0.01), replace=False)
    df.loc[glitch_idx_3, 'throughput_teu_per_hr'] = 0.0 # Crane telemetry stuck on zero

    # 3. Inject String & Formatting Inconsistencies
    carrier_variations = {
        'MSK': ['MSK', 'msk', 'MAERSK', ' Maersk ', 'MSK_LINE'],
        'MSC': ['MSC', 'msc', 'MED_SHIPPING', 'MSC_CORP '],
        'HAPAG': ['HAPAG', 'hapag', 'HAPAG_LLOYD', 'HL_AG '],
        'ONE': ['ONE', 'one', 'OCEAN_NET', ' ONE '],
        'COSCO': ['COSCO', 'cosco', 'COSCO_SHP', 'COSCO ']
    }
    def perturb_carrier(c):
        if c in carrier_variations and np.random.rand() < 0.25:
            return np.random.choice(carrier_variations[c])
        return c
    df['carrier_code'] = df['carrier_code'].apply(perturb_carrier)

    # Lowercase or strip random port IDs
    port_cols = ['origin_port', 'destination_port', 'current_node_id']
    for p_col in port_cols:
        dirty_mask = np.random.rand(n) < 0.10
        df.loc[dirty_mask, p_col] = df.loc[dirty_mask, p_col].str.lower()
        dirty_pad = np.random.rand(n) < 0.05
        df.loc[dirty_pad, p_col] = df.loc[dirty_pad, p_col] + " "

    # 4. Inject Timestamp Formatting Inconsistencies & Jitter
    def format_ts(ts):
        r = np.random.rand()
        if r < 0.15:
            return ts.replace(' ', 'T') + 'Z'  # ISO-8601 format
        elif r < 0.25:
            return ts + '.000'                  # Milliseconds format
        return ts
    df['timestamp'] = df['timestamp'].apply(format_ts)

    # 5. Inject Latent Stochasticity (Eliminating the 100% Accuracy Synthetic Rule Trap)
    # In real logistics, unobserved latent variables (paperwork errors, crane mechanic shift changes,
    # gate queue jams) create genuine uncertainty near boundary conditions.
    boundary_mask = (df['dwell_time_hours'] > 18) & (df['dwell_time_hours'] < 30)
    flip_candidates = df[boundary_mask].index
    num_flips = int(len(flip_candidates) * 0.08) # 8% stochastic latent variance
    flips = np.random.choice(flip_candidates, size=num_flips, replace=False)
    for idx in flips:
        curr_status = df.loc[idx, 'delay_status']
        if curr_status == 'ON_TIME':
            df.loc[idx, 'delay_status'] = 'AT_RISK'
        elif curr_status == 'AT_RISK':
            df.loc[idx, 'delay_status'] = np.random.choice(['ON_TIME', 'CRITICAL_DELAY'])
        elif curr_status == 'CRITICAL_DELAY':
            df.loc[idx, 'delay_status'] = 'AT_RISK'

    # 6. Inject Duplicate Records (RFID gate reader multiple scans)
    dup_sample = df.sample(n=int(n * 0.03), random_state=seed)
    df_uncleaned = pd.concat([df, dup_sample], ignore_index=True)

    # Shuffle slightly to simulate asynchronous multi-carrier network arrival
    df_uncleaned = df_uncleaned.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df_uncleaned.to_csv(output_path, index=False)
    print(f"Successfully generated realistic uncleaned dataset with defects: {output_path}")
    print(f"New dataset shape: {df_uncleaned.shape} ({len(df_uncleaned) - n} duplicate scans added)")
    print(f"Total missing values: {df_uncleaned.isnull().sum().sum()}")

if __name__ == '__main__':
    inject_real_world_noise(
        input_path="/Users/ayyapparaja/pahse 2/data/sample_raw_logistics.csv",
        output_path="/Users/ayyapparaja/pahse 2/data/uncleaned_raw_telemetry.csv"
    )


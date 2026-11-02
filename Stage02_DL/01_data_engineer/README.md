# Role 01: Data Engineer (Stage 02 - Deep Learning)
### Mission Objective
Transform irregular, discrete container event timestamps into continuous, synchronized 1-hour node telemetry sequences and sliding window tensors.

### Responsibilities:
- `01_data_engineer.py`: Ingest raw shipment feeds, resample to hourly intervals per node.
- Impute missing hourly gaps using forward-fill/physics limits.
- Compute network graph adjacency links (`network_edges.csv`).
- Generate (N, 12, 8) sliding sequence windows for LSTM consumption.
- Chronological split: 75% train (2,700 windows) / 25% test (923 windows).
- Normalize features using training-set-only scalers (`scaler.pkl`).

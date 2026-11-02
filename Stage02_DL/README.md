# Stage 02: Deep Learning Engine
## Project 07: Forecasting Dwell Time Across a Global Logistics Network

### Mission Architecture
Stage 02 shifts the paradigm from individual snapshot classification (Stage 01) to **temporal & spatial sequence forecasting**:
Given the past 12 hours of 8 operational telemetry streams across 6 interconnected logistics hubs, forecast the **next-hour dwell time (h13)** before network gridlock cascades.

```
Stage02_DL/
|-- 01_data_engineer/        # Resampling, regular hourly aggregation, graph topology, sliding sequence tensors
|-- 02_eda_engineer/         # Time-series trend decomposition, autocorrelation, cross-node delay propagation
|-- 03_dl_engineer/          # PyTorch LSTM Forecaster (32 hidden units), training loop (35 epochs), graph baseline
|-- 04_evaluation_engineer/  # Holdout benchmark on 923 unseen future hours (MAE ~4.24h, RMSE ~5.66h, R2 ~0.61)
|-- 05_integration_engineer/ # Real-time sequence inference service & operator escalation triggers
`-- data/                    # Hourly telemetry, graph edges, sliding window tensors (.pt), and PyTorch model weights
```

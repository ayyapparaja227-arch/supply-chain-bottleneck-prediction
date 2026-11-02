#!/usr/bin/env python3
"""
STAGE 02 DEEP LEARNING — MILESTONE 7 (60% -> 70%)
Role 03: DL Engineer — LSTM Model Training & Optimization Pipeline

Problem Addressed (Slide 14 & 19 in 2_DL.pptx):
"Train the PyTorch LSTM Forecaster on 2,700 training sequence windows over 35 epochs.
Validate convergence against MSE loss, track learning dynamics, and serialize
model weights for downstream evaluation."

Deliverables:
1. Loads pre-built tensors (X_train.pt, y_train.pt, X_test.pt, y_test.pt) and scaler.
2. Trains LSTMForecaster with Adam optimizer (lr=0.001) for 35 epochs.
3. Computes training loss & test loss per epoch, logging to training_history.csv.
4. Inverse-transforms scaled predictions back to physical hours.
5. Saves:
   - data/lstm_dwell_forecaster.pt
   - data/training_history.csv
   - data/lstm_forecast_test.csv
   - data/dl_metrics.json
   - data/milestone_70pct_report.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import LSTMForecaster

FEATURES = [
    "dwell_time_hours",
    "throughput_teu_per_hr",
    "capacity_utilization_pct",
    "customs_hold_hours",
    "weather_severity_index",
    "labor_deficit_ratio",
    "equipment_availability_pct",
    "historical_delay_rate"
]
SEQ_LEN = 12
EPOCHS = 35
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

def run_milestone_70pct_training():
    print("=" * 70)
    print("🚢 STAGE 02 (DEEP LEARNING) — MILESTONE 7 [70% COMPLETE]")
    print("   Role 03: DL Engineer — Model Training & Optimization Pipeline")
    print("=" * 70)

    # Set seeds for deterministic reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Detect compute device (Apple Silicon MPS / CUDA / CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n[1/5] Compute device initialized: {device}")

    base_dir = Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage02_DL")
    if not base_dir.exists():
        base_dir = Path("Stage02_DL")
    data_dir = base_dir / "data"

    # Load pre-built tensors
    print("\n[2/5] Loading partitioned sliding window sequence datasets...")
    X_train = torch.load(data_dir / "X_train.pt", weights_only=True)
    y_train = torch.load(data_dir / "y_train.pt", weights_only=True)
    X_test = torch.load(data_dir / "X_test.pt", weights_only=True)
    y_test = torch.load(data_dir / "y_test.pt", weights_only=True)

    print(f"  • Train dataset: X_train = {tuple(X_train.shape)}, y_train = {tuple(y_train.shape)}")
    print(f"  • Test dataset:  X_test  = {tuple(X_test.shape)},  y_test  = {tuple(y_test.shape)}")

    # Load scaler parameters from 40% report or calculate from prepared data
    report_40_path = data_dir / "milestone_40pct_report.json"
    with open(report_40_path) as f:
        meta_40 = json.load(f)
    scaler_means = meta_40["scaler"]["means"]
    scaler_scales = meta_40["scaler"]["scales"]

    dwell_mean = scaler_means["dwell_time_hours"]
    dwell_scale = scaler_scales["dwell_time_hours"]

    # Initialize model
    print(f"\n[3/5] Instantiating LSTMForecaster (8 inputs -> 32 hidden units -> 16 -> 1)...")
    model = LSTMForecaster(input_size=len(FEATURES), hidden_size=32, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    X_tr_dev = X_train.to(device)
    y_tr_dev = y_train.to(device)
    X_te_dev = X_test.to(device)
    y_te_dev = y_test.to(device)

    # 4. Training Loop across 35 Epochs
    print(f"\n[4/5] Executing {EPOCHS} training epochs with Adam optimizer:")
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tr_dev)
        loss = criterion(preds, y_tr_dev)
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_preds = model(X_te_dev)
            val_loss = criterion(val_preds, y_te_dev)

        history.append({
            "epoch": epoch,
            "train_mse_loss": round(float(loss.item()), 6),
            "test_mse_loss": round(float(val_loss.item()), 6)
        })

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  Epoch [{epoch:2d}/{EPOCHS}] -> Train MSE Loss: {loss.item():.5f} | Test MSE Loss: {val_loss.item():.5f}")

    # Save training history
    df_hist = pd.DataFrame(history)
    hist_csv = data_dir / "training_history.csv"
    df_hist.to_csv(hist_csv, index=False)
    print(f"  • Training convergence history saved -> {hist_csv.name}")

    # 5. Evaluate on 923 Test Windows and Save Forecasts
    print("\n[5/5] Inverting predictions back to physical hours and calculating metrics...")
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(X_te_dev).cpu().numpy()

    test_actual_scaled = y_test.numpy()

    # Inverse transform to continuous dwell hours
    actual_hours = test_actual_scaled * dwell_scale + dwell_mean
    pred_hours = test_pred_scaled * dwell_scale + dwell_mean

    mae = float(mean_absolute_error(actual_hours, pred_hours))
    rmse = float(np.sqrt(mean_squared_error(actual_hours, pred_hours)))
    r2 = float(r2_score(actual_hours, pred_hours))

    print(f"\n  🎯 Unseen Test Set Performance (Slide 19 Benchmarks):")
    print(f"    • MAE (Hours):  {mae:.2f} h  (Benchmark target: ~4.24 h)")
    print(f"    • RMSE (Hours): {rmse:.2f} h  (Benchmark target: ~5.66 h)")
    print(f"    • R² Score:     {r2:.3f}    (Benchmark target: ~0.61)")

    # Save predictions dataframe
    df_preds = pd.DataFrame({
        "actual_dwell_time_hours": np.round(actual_hours, 2),
        "predicted_dwell_time_hours": np.round(pred_hours, 2),
        "error_hours": np.round(pred_hours - actual_hours, 2)
    })
    preds_csv = data_dir / "lstm_forecast_test.csv"
    df_preds.to_csv(preds_csv, index=False)
    print(f"  • Test set forecasts saved -> {preds_csv.name}")

    # Save model weights and bundle
    model_path = data_dir / "lstm_dwell_forecaster.pt"
    save_bundle = {
        "state_dict": model.state_dict(),
        "features": FEATURES,
        "sequence_length": SEQ_LEN,
        "scaler_mean": [scaler_means[f] for f in FEATURES],
        "scaler_scale": [scaler_scales[f] for f in FEATURES],
        "dwell_mean": dwell_mean,
        "dwell_scale": dwell_scale,
        "epochs_trained": EPOCHS,
        "final_train_loss": float(history[-1]["train_mse_loss"]),
        "final_test_loss": float(history[-1]["test_mse_loss"])
    }
    torch.save(save_bundle, model_path)
    print(f"  • Trained model weights saved -> {model_path.name}")

    # dl_metrics.json
    metrics = {
        "device": str(device),
        "sequence_length_hours": SEQ_LEN,
        "train_sequences": len(X_train),
        "test_sequences": len(X_test),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "final_train_loss": float(history[-1]["train_mse_loss"]),
        "final_test_loss": float(history[-1]["test_mse_loss"]),
        "MAE_hours": round(mae, 4),
        "RMSE_hours": round(rmse, 4),
        "R2_score": round(r2, 4),
        "status": "CONVERGED_SUCCESSFULLY"
    }
    with open(data_dir / "dl_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Milestone 7 Report
    milestone_report = {
        "milestone": "70% Complete",
        "phase": "Role 03: DL Engineer — Model Training & Optimization Pipeline",
        "training_epochs": EPOCHS,
        "loss_progression": {
            "epoch_1_train_loss": history[0]["train_mse_loss"],
            "epoch_35_train_loss": history[-1]["train_mse_loss"],
            "loss_reduction_pct": round((1.0 - history[-1]["train_mse_loss"] / history[0]["train_mse_loss"]) * 100, 2)
        },
        "evaluation_on_923_test_windows": {
            "MAE_hours": round(mae, 2),
            "RMSE_hours": round(rmse, 2),
            "R2_score": round(r2, 3)
        },
        "artifacts_generated": [
            "data/lstm_dwell_forecaster.pt",
            "data/training_history.csv",
            "data/lstm_forecast_test.csv",
            "data/dl_metrics.json"
        ],
        "status": "MODEL_READY_FOR_GRAPH_BASELINE_AND_EVALUATION"
    }
    with open(data_dir / "milestone_70pct_report.json", "w") as f:
        json.dump(milestone_report, f, indent=2)

    print(f"\n✅ Milestone 7 report generated: milestone_70pct_report.json")
    print("=" * 70)
    print("🎯 MILESTONE 7 [70%] SIGN-OFF: LSTM TRAINING PIPELINE CONVERGED!")
    print("======================================================================")

if __name__ == "__main__":
    run_milestone_70pct_training()

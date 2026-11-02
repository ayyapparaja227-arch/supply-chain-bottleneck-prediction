#!/usr/bin/env python3
"""
STAGE 02 DEEP LEARNING — MILESTONE 6 (50% -> 60%)
Role 03: DL Engineer — PyTorch LSTM Architecture Specification

Problem Addressed (Slide 13-14 in 2_DL.pptx):
"Model Architecture:
 1. LSTM Layer: 8 continuous operational signals -> 32 hidden units.
 2. Hidden State Selection: Last time step z[:, -1, :] captures the full 12-hour momentum.
 3. Fully Connected Projection Head:
    Linear(32 -> 16) -> ReLU() -> Linear(16 -> 1) -> Squeeze()
 Output: Continuous scalar forecast of Dwell Time at hour 13."

Deliverables:
- Modular LSTMForecaster nn.Module class with shape contracts.
- Parameter count audit & layer-by-layer dimension validation.
- Unit test with synthetic batches and actual X_train tensors.
- data/milestone_60pct_report.json
"""

from pathlib import Path
import json
import torch
from torch import nn

class LSTMForecaster(nn.Module):
    """
    PyTorch Recurrent Forecaster for Port & Hub Container Dwell Time.
    
    Architecture:
      Input: (Batch, Seq_Len=12, Features=8)
      -> LSTM(input_size=8, hidden_size=32, batch_first=True)
      -> Take last step hidden state: h_T in R^32
      -> Linear(32, 16)
      -> ReLU()
      -> Linear(16, 1)
      -> Squeeze(-1) -> (Batch,)
    """
    def __init__(self, input_size: int = 8, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, 12, 8)
        Returns:
            Tensor of shape (Batch,) containing continuous next-hour dwell forecasts.
        """
        # z shape: (Batch, 12, 32)
        z, (h_n, c_n) = self.lstm(x)
        # Extract representation at last observed time step (hour 12)
        last_step = z[:, -1, :]  # Shape: (Batch, 32)
        out = self.head(last_step).squeeze(-1)  # Shape: (Batch,)
        return out

    def get_parameter_breakdown(self):
        breakdown = {}
        total = 0
        for name, param in self.named_parameters():
            num = param.numel()
            breakdown[name] = {"shape": list(param.shape), "parameters": num}
            total += num
        breakdown["total_trainable_parameters"] = total
        return breakdown

def run_milestone_60pct_verification():
    print("=" * 70)
    print("🚢 STAGE 02 (DEEP LEARNING) — MILESTONE 6 [60% COMPLETE]")
    print("   Role 03: DL Engineer — PyTorch LSTM Architecture Specification")
    print("=" * 70)

    base_dir = Path("/Users/ayyapparaja/Downloads/Supply_Chain_Bottleneck_Industry_Oriented_Project/supply-chain-bottleneck-prediction/Stage02_DL")
    if not base_dir.exists():
        base_dir = Path("Stage02_DL")
    data_dir = base_dir / "data"

    # Instantiate model
    model = LSTMForecaster(input_size=8, hidden_size=32, num_layers=1)
    params = model.get_parameter_breakdown()

    print("\n[1/3] Validating Neural Network Architecture & Parameters:")
    print(f"  • Model Type: LSTM Sequence-to-Scalar Forecaster")
    print(f"  • Input dimension:  (Batch, 12, 8)")
    print(f"  • LSTM Hidden size: 32 units")
    print(f"  • Dense MLP Head:   32 -> 16 -> 1")
    print(f"  • Parameter Breakdown:")
    for k, v in params.items():
        if k != "total_trainable_parameters":
            print(f"    - {k:25}: shape {str(v['shape']):15} ({v['parameters']} params)")
    print(f"  • Total Trainable Parameters: {params['total_trainable_parameters']:,}")

    # Forward pass on synthetic batch
    print("\n[2/3] Performing Forward Pass Shape Contract Verification:")
    dummy_input = torch.randn(16, 12, 8)
    dummy_out = model(dummy_input)
    assert dummy_out.shape == (16,), f"Output shape mismatch! Expected (16,), got {dummy_out.shape}"
    print(f"  ✅ Synthetic Batch Check Passed: Input (16, 12, 8) -> Output {tuple(dummy_out.shape)}")

    # Forward pass on real X_train sample if available
    train_pt = data_dir / "X_train.pt"
    if train_pt.exists():
        X_train = torch.load(train_pt, weights_only=True)
        sample_batch = X_train[:32]
        real_out = model(sample_batch)
        assert real_out.shape == (32,), f"Real batch mismatch! Got {real_out.shape}"
        print(f"  ✅ Actual Telemetry Batch Check Passed: Input {tuple(sample_batch.shape)} -> Output {tuple(real_out.shape)}")

    # Generate Milestone 6 Report
    print("\n[3/3] Generating Milestone 6 Verification Audit...")
    report = {
        "milestone": "60% Complete",
        "phase": "Role 03: DL Engineer — PyTorch Architecture Specification",
        "model_architecture": {
            "name": "LSTMForecaster",
            "sequence_length": 12,
            "input_features": 8,
            "hidden_units": 32,
            "num_layers": 1,
            "dense_head": [
                {"layer": "Linear", "in_features": 32, "out_features": 16},
                {"layer": "ReLU"},
                {"layer": "Linear", "in_features": 16, "out_features": 1}
            ],
            "total_trainable_parameters": params["total_trainable_parameters"]
        },
        "layer_parameters": params,
        "input_contract": "(Batch, 12, 8)",
        "output_contract": "(Batch,)",
        "status": "VERIFIED_READY_FOR_TRAINING"
    }

    report_path = data_dir / "milestone_60pct_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  ✅ Saved: {report_path.name}")
    print("=" * 70)
    print("🎯 MILESTONE 6 [60%] SIGN-OFF: LSTM ARCHITECTURE FULLY SPECIFIED!")
    print("======================================================================")

if __name__ == "__main__":
    run_milestone_60pct_verification()


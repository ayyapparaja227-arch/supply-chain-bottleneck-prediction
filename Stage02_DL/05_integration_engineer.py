"""ROLE 5 — INTEGRATION ENGINEER: safe local prediction interface for the trained LSTM."""
from pathlib import Path
import torch, pandas as pd
from torch import nn
DATA=Path("data"); ART=DATA/"lstm_dwell_forecaster.pt"
FEATURES=["dwell_time_hours","throughput_teu_per_hr","capacity_utilization_pct","customs_hold_hours","weather_severity_index","labor_deficit_ratio","equipment_availability_pct","historical_delay_rate"]
class LSTMForecaster(nn.Module):
    def __init__(self,nf):
        super().__init__(); self.lstm=nn.LSTM(nf,32,batch_first=True); self.head=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): z,_=self.lstm(x); return self.head(z[:,-1,:]).squeeze(-1)
def predict_next_dwell_hours(sequence):
    """sequence: list of 12 hourly dictionaries, oldest first, with all FEATURES."""
    if not ART.exists(): raise FileNotFoundError("Train the model by running 03_dl_engineer.py first")
    ckpt=torch.load(ART,map_location="cpu",weights_only=False); seq_len=ckpt["sequence_length"]
    if len(sequence)!=seq_len: raise ValueError(f"Expected {seq_len} hourly records, oldest first")
    frame=pd.DataFrame(sequence)
    missing=[f for f in FEATURES if f not in frame.columns]
    if missing: raise ValueError(f"Missing telemetry fields: {missing}")
    import numpy as np
    arr=frame[FEATURES].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    mean=np.asarray(ckpt["scaler_mean"]); scale=np.asarray(ckpt["scaler_scale"]); arr=(arr-mean)/scale
    model=LSTMForecaster(len(FEATURES)); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad(): value=float(model(torch.tensor(arr[None,:,:])).item())
    hours=max(0.0,value*scale[0]+mean[0])
    return {"predicted_next_step_dwell_time_hours":round(hours,2),"forecast_horizon":"next hourly observation","requires_operator_review":True,"automated_rerouting":False}
if __name__=="__main__": print("Integration interface ready. Call predict_next_dwell_hours() with 12 chronological hourly records.")

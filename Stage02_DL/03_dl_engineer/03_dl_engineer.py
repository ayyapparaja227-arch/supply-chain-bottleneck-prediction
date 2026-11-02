"""ROLE 3 — DL ENGINEER: LSTM next-hour dwell-time forecasting + graph message passing.
Forecast is next observed hourly step per node, not a guaranteed 7-day forecast.
"""
from pathlib import Path
import json, numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
DATA = Path("data") if Path("data").exists() else Path("../data"); torch.manual_seed(42); np.random.seed(42)
FEATURES=["dwell_time_hours","throughput_teu_per_hr","capacity_utilization_pct","customs_hold_hours","weather_severity_index","labor_deficit_ratio","equipment_availability_pct","historical_delay_rate"]
SEQ=12; EPOCHS=35; DEVICE="cuda" if torch.cuda.is_available() else "cpu"
df=pd.read_csv(DATA/"hourly_node_telemetry.csv",parse_dates=["hour"])
if df.empty: raise ValueError("No hourly telemetry. Run 01_data_engineer.py first.")
# Chronological split per node; fit scaler only on training rows to reduce leakage.
train_cut=df.hour.quantile(.75)
scaler=StandardScaler().fit(df.loc[df.hour<=train_cut,FEATURES].fillna(0))
scaled=scaler.transform(df[FEATURES].fillna(0)); df[FEATURES]=scaled
Xtr=[]; ytr=[]; Xte=[]; yte=[]; test_meta=[]
for node,g in df.groupby("current_node_id"):
    g=g.sort_values("hour"); ix=g.index.to_numpy(); arr=g[FEATURES].to_numpy(dtype=np.float32)
    for j in range(SEQ,len(g)):
        sample=arr[j-SEQ:j]; target=float(arr[j,0])
        if g.hour.iloc[j] <= train_cut: Xtr.append(sample); ytr.append(target)
        else: Xte.append(sample); yte.append(target); test_meta.append((node,str(g.hour.iloc[j])))
if len(Xtr)<5: raise ValueError("Not enough hourly history to form sequences. Reduce SEQ or add data.")
class LSTMForecaster(nn.Module):
    def __init__(self,nf):
        super().__init__(); self.lstm=nn.LSTM(nf,32,batch_first=True); self.head=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x):
        z,_=self.lstm(x); return self.head(z[:,-1,:]).squeeze(-1)
model=LSTMForecaster(len(FEATURES)).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=.001); loss_fn=nn.MSELoss()
xt=torch.tensor(np.asarray(Xtr),device=DEVICE); yt=torch.tensor(ytr,dtype=torch.float32,device=DEVICE)
for epoch in range(EPOCHS):
    model.train(); opt.zero_grad(); loss=loss_fn(model(xt),yt); loss.backward(); opt.step()
model.eval()
with torch.no_grad(): pred=model(torch.tensor(np.asarray(Xte if Xte else Xtr),device=DEVICE)).cpu().numpy()
actual=np.asarray(yte if Xte else ytr)
# Save predictions in original dwell-time hours by inverse-transforming the dwell-time feature.
mu=scaler.mean_[0]; sd=scaler.scale_[0]; actual_h=actual*sd+mu; pred_h=pred*sd+mu
pd.DataFrame({"actual_dwell_time_hours":actual_h,"predicted_dwell_time_hours":pred_h}).to_csv(DATA/"lstm_forecast_test.csv",index=False)
torch.save({"state_dict":model.state_dict(),"features":FEATURES,"sequence_length":SEQ,"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist()},DATA/"lstm_dwell_forecaster.pt")
metrics={"device":DEVICE,"sequence_length_hours":SEQ,"train_sequences":len(Xtr),"test_sequences":len(Xte),"split_timestamp":str(train_cut),"MAE_hours":float(mean_absolute_error(actual_h,pred_h)),"RMSE_hours":float(np.sqrt(mean_squared_error(actual_h,pred_h))),"note":"If test_sequences is 0, metrics are training-only and must not be treated as validation."}
(DATA/"dl_metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))

# Lightweight graph-neighbour aggregation benchmark. This is an explicit message-passing baseline,
# not a trained GNN; it demonstrates node relationships and can be extended when richer graph labels exist.
raw=pd.read_csv(DATA/"prepared_dataset.csv")
node=raw.groupby("current_node_id").agg(mean_dwell=("dwell_time_hours","mean"),mean_capacity=("capacity_utilization_pct","mean"),mean_throughput=("throughput_teu_per_hr","mean")).reset_index()
edges=pd.read_csv(DATA/"logistics_edges.csv") if (DATA/"logistics_edges.csv").exists() else pd.DataFrame(columns=["source","target"])
lookup={r.current_node_id:r.mean_dwell for r in node.itertuples()}; neighbors={n:[] for n in lookup}
for e in edges.itertuples():
    if e.source in neighbors and e.target in neighbors: neighbors[e.source].append(e.target); neighbors[e.target].append(e.source)
node["neighbor_mean_dwell"]=[float(np.mean([lookup[n] for n in neighbors.get(k,[]) ])) if neighbors.get(k) else np.nan for k in node.current_node_id]
node["neighbor_count"]=[len(neighbors.get(k,[])) for k in node.current_node_id]
node.to_csv(DATA/"graph_node_features.csv",index=False)
print("Saved LSTM model, forecast predictions, metrics, and graph neighbour features.")

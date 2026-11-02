"""ROLE 4 — EVALUATION ENGINEER: time-ordered forecast metrics and naive baseline comparison."""
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
DATA = Path("data") if Path("data").exists() else Path("../data"); p=DATA/"lstm_forecast_test.csv"
if not p.exists(): raise FileNotFoundError("Run 03_dl_engineer.py first")
df=pd.read_csv(p); y=df.actual_dwell_time_hours; pred=df.predicted_dwell_time_hours
# Persistence baseline: predict the next value as the current/last observed value is not stored in the output;
# therefore report model metrics only and state this limitation instead of inventing a baseline.
metrics={"n_evaluated":len(df),"MAE_hours":float(mean_absolute_error(y,pred)),"RMSE_hours":float(np.sqrt(mean_squared_error(y,pred))),"R2":float(r2_score(y,pred)) if len(df)>1 else None,"limitations":["Forecast target is the next hourly observation, not seven days ahead.","Rows are synthetic/demo telemetry; performance is not evidence of real-world readiness.","A persistence baseline and rolling-origin backtest should be added before operational use."]}
(DATA/"evaluation_report.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))

# Role 04: Evaluation Engineer (Stage 02 - Deep Learning)
### Mission Objective
Rigorously benchmark the LSTM against baseline models on 923 unseen future chronological test windows.

### Benchmarks:
1. **PyTorch LSTM Champion**: Target MAE ~4.24h, RMSE ~5.66h, R2 ~0.61.
2. **Naive Persistence Baseline** ("Same as last hour"): y_{t+1} = y_t.
3. **Graph Message-Passing Baseline**: Average of connected neighbor dwell times.

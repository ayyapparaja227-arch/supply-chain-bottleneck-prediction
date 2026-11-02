# Role 05: Integration Engineer (Stage 02 - Deep Learning)
### Mission Objective
Expose the trained PyTorch LSTM model via a clean Python inference connector (`05_integration_engineer.py`) with safety bounds and operator review flags.

### Capabilities:
- Accepts arbitrary 12-hour operational sequences.
- Normalizes inputs using the frozen training scaler.
- Outputs next-hour forecast dwell (hours) and confidence bounds.
- Flags critical escalation if forecast >= 24 hours or if acceleration > 3.5 h/hr.

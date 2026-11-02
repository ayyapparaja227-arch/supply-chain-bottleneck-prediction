# Role 03: Deep Learning Engineer (Stage 02 - Deep Learning)
### Mission Objective
Design, train, and calibrate a PyTorch recurrent neural network (LSTM) for sequence regression and implement a graph message-passing neighbor baseline.

### Model Architecture:
- Input: (B, 12, 8)
- LSTM Layer: input_size=8, hidden_size=32, batch_first=True
- MLP Regressor: Linear(32, 16) -> ReLU() -> Linear(16, 1)
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss: Mean Squared Error (MSELoss)
- Epochs: 35 passes with validation loss checkpointing

import torch

from src.activity.lstm_model import ActivityLSTM


model = ActivityLSTM()

print("========== MODEL ==========")
print(model)


# Fake batch
x = torch.randn(16, 30, 34)

output = model(x)

print("\n========== OUTPUT ==========")
print("Input shape :", x.shape)
print("Output shape:", output.shape)
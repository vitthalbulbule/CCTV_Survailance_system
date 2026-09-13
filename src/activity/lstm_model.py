import torch
import torch.nn as nn


class ActivityLSTM(nn.Module):

    def __init__(
        self,
        input_size=34,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # (batch, sequence_length, features)
        # (batch, 30, 34)

        output, (hidden, cell) = self.lstm(x)

        # Take output from the last frame
        last_output = output[:, -1, :]

        # Classification
        logits = self.fc(last_output)

        return logits
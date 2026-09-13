import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ActivityDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        self.label_map = {
            "Non-Fighting": 0,
            "Fighting": 1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        sequence_path = row["sequence_path"]
        label = row["label"]

        # Load sequence
        sequence = np.load(sequence_path)

        # Expected shape: (30, 17, 2)
        if sequence.shape != (30, 17, 2):
            raise ValueError(
                f"Expected shape (30, 17, 2), "
                f"but got {sequence.shape}"
            )

        # Convert:
        # (30, 17, 2) -> (30, 34)
        sequence = sequence.reshape(30, 34)

        # Convert to PyTorch tensor
        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        label = self.label_map[label]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return sequence, label
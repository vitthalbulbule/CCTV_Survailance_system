import numpy as np
import torch

from src.activity.lstm_model import ActivityLSTM


class ActivityClassifier:

    def __init__(
        self,
        model_path="models/activity_lstm_best.pth"
    ):

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load trained model checkpoint
        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # Create same architecture used during training
        self.model = ActivityLSTM(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            num_classes=checkpoint["num_classes"]
        )

        # Load learned weights
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Classes must match training labels
        self.classes = [
            "Non-Fighting",
            "Fighting"
        ]


    def predict(self, sequence):

        # Convert to NumPy
        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        # Expected original shape
        if sequence.shape != (30, 17, 2):

            raise ValueError(
                f"Expected shape (30, 17, 2), "
                f"but got {sequence.shape}"
            )

        # Convert:
        # (30, 17, 2) → (30, 34)
        sequence = sequence.reshape(30, 34)

        # Add batch dimension:
        # (30, 34) → (1, 30, 34)
        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        sequence = sequence.to(self.device)

        # Model inference
        with torch.no_grad():

            outputs = self.model(sequence)

            # Convert logits to probabilities
            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            # Highest probability class
            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0, predicted_class
            ].item()

        prediction = self.classes[
            predicted_class
        ]

        return prediction, confidence
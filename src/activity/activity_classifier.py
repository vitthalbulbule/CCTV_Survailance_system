import numpy as np


class ActivityClassifier:

    def __init__(self):
        self.classes = [
            "Standing",
            "Walking",
            "Fighting",
            "Falling"
        ]

    def predict(self, sequence):

        sequence = np.array(sequence)

        # Check input shape
        if sequence.shape != (30, 17, 2):
            raise ValueError(
                f"Expected shape (30, 17, 2), "
                f"but got {sequence.shape}"
            )

        # Temporary prediction
        # We will replace this with the real model later
        prediction = "Standing"

        confidence = 0.90

        return prediction, confidence
import numpy as np

from src.activity.activity_classifier import ActivityClassifier


# Create classifier
classifier = ActivityClassifier()

print("Classifier loaded successfully.")
print("Classes:", classifier.classes)


# Create a test sequence
# Shape = (30, 17, 2)
sequence = np.random.rand(
    30, 17, 2
).astype(np.float32)


# Predict
prediction, confidence = classifier.predict(
    sequence
)


print("\n========== PREDICTION ==========")

print("Prediction :", prediction)
print("Confidence :", f"{confidence:.4f}")
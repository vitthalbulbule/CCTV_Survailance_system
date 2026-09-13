import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.activity.activity_dataset import ActivityDataset
from src.activity.lstm_model import ActivityLSTM


# =========================
# CONFIGURATION
# =========================

TRAIN_CSV = "dataset/splits/train.csv"
VAL_CSV = "dataset/splits/validation.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "activity_lstm_best.pth")

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

PATIENCE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# DATASET
# =========================

train_dataset = ActivityDataset(TRAIN_CSV)
val_dataset = ActivityDataset(VAL_CSV)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# MODEL
# =========================

model = ActivityLSTM().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# TRAINING
# =========================

best_f1 = 0.0
patience_counter = 0

os.makedirs(MODEL_DIR, exist_ok=True)

print("================================")
print("        LSTM TRAINING")
print("================================")

print("Device:", DEVICE)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    train_loss = 0.0

    for sequences, labels in train_loader:

        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(sequences)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for sequences, labels in val_loader:

            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(sequences)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    val_loss /= len(val_loader)

    # -------------------------
    # METRICS
    # -------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )


    print(
        f"\nEpoch [{epoch + 1:02d}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f}"
    )

    print(
        f"Accuracy:   {accuracy:.4f}"
    )

    print(
        f"Precision:  {precision:.4f}"
    )

    print(
        f"Recall:     {recall:.4f}"
    )

    print(
        f"F1 Score:   {f1:.4f}"
    )


    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if f1 > best_f1:

        best_f1 = f1

        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_f1": best_f1,
                "input_size": 34,
                "hidden_size": 64,
                "num_layers": 2,
                "num_classes": 2
            },
            MODEL_PATH
        )

        print(
            f"✓ Best model saved "
            f"(F1 = {best_f1:.4f})"
        )

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    # -------------------------
    # EARLY STOPPING
    # -------------------------

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


print("\n================================")
print("        TRAINING COMPLETE")
print("================================")

print(
    f"Best Validation F1: {best_f1:.4f}"
)

print(
    f"Model saved at: {MODEL_PATH}"
)
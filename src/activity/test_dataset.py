from torch.utils.data import DataLoader

from src.activity.activity_dataset import ActivityDataset


train_dataset = ActivityDataset(
    "dataset/splits/train.csv"
)

validation_dataset = ActivityDataset(
    "dataset/splits/validation.csv"
)

test_dataset = ActivityDataset(
    "dataset/splits/test.csv"
)


print("========== DATASET ==========")

print("Train:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Test:", len(test_dataset))


sequence, label = train_dataset[0]

print("\n========== FIRST SAMPLE ==========")

print("Sequence shape:", sequence.shape)
print("Label:", label)
print("Sequence dtype:", sequence.dtype)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

sequences, labels = next(iter(train_loader))

print("\n========== BATCH ==========")

print("Batch sequence shape:", sequences.shape)
print("Batch labels shape:", labels.shape)
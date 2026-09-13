import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


SEQUENCE_DIR = "dataset/sequence"
SPLIT_DIR = "dataset/splits"

RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def get_video_id(filename):
    """
    Extract source video ID.

    Examples:
        fi10_xvid_person_2_seq_000.npy -> fi10
        no111_xvid_person_2_seq_004.npy -> no111
    """
    match = re.match(r"^(fi\d+|no\d+)_", filename)

    if match:
        return match.group(1)

    return None


def collect_sequences():

    data = []

    for label in ["Fighting", "Non-Fighting"]:

        if label == "Fighting":
            folder = os.path.join(SEQUENCE_DIR, "fighting")
        else:
            folder = os.path.join(SEQUENCE_DIR, "non_fighting")

        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        for filename in os.listdir(folder):

            if not filename.endswith(".npy"):
                continue

            video_id = get_video_id(filename)

            if video_id is None:
                print(f"Could not extract video ID: {filename}")
                continue

            sequence_path = os.path.join(folder, filename)

            data.append({
                "sequence_path": sequence_path,
                "label": label,
                "video_id": video_id
            })

    return pd.DataFrame(data)


def split_videos(df):

    # One row per source video
    videos = df[["video_id", "label"]].drop_duplicates()

    fighting = videos[videos["label"] == "Fighting"]
    non_fighting = videos[videos["label"] == "Non-Fighting"]

    def split_class(class_df):

        # 70% train, 30% temporary
        train, temp = train_test_split(
            class_df,
            test_size=0.30,
            random_state=RANDOM_SEED
        )

        # Split remaining 30% into 15% validation + 15% test
        validation, test = train_test_split(
            temp,
            test_size=0.50,
            random_state=RANDOM_SEED
        )

        return train, validation, test

    f_train, f_val, f_test = split_class(fighting)
    n_train, n_val, n_test = split_class(non_fighting)

    train_videos = pd.concat([f_train, n_train])
    val_videos = pd.concat([f_val, n_val])
    test_videos = pd.concat([f_test, n_test])

    return train_videos, val_videos, test_videos


def create_split(df, videos):

    video_ids = set(videos["video_id"])

    return df[df["video_id"].isin(video_ids)].copy()


def main():

    os.makedirs(SPLIT_DIR, exist_ok=True)

    df = collect_sequences()

    if df.empty:
        print("No sequences found.")
        return

    print("\n========== SOURCE VIDEOS ==========")

    for label in ["Fighting", "Non-Fighting"]:

        count = df[df["label"] == label]["video_id"].nunique()

        print(f"{label}: {count} videos")

    # Split at SOURCE VIDEO level
    train_videos, val_videos, test_videos = split_videos(df)

    # Convert video splits into sequence splits
    train_df = create_split(df, train_videos)
    val_df = create_split(df, val_videos)
    test_df = create_split(df, test_videos)

    # Shuffle sequences
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED)
    val_df = val_df.sample(frac=1, random_state=RANDOM_SEED)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED)

    # Save CSV files
    train_path = os.path.join(SPLIT_DIR, "train.csv")
    val_path = os.path.join(SPLIT_DIR, "validation.csv")
    test_path = os.path.join(SPLIT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nCreated: {train_path}")
    print(f"Sequences: {len(train_df)}")

    print(f"\nCreated: {val_path}")
    print(f"Sequences: {len(val_df)}")

    print(f"\nCreated: {test_path}")
    print(f"Sequences: {len(test_df)}")

    print("\n========== SPLIT SUMMARY ==========")

    for name, split in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]:

        print(f"\n{name}:")
        print(f"  Sequences: {len(split)}")
        print(f"  Videos: {split['video_id'].nunique()}")

        print(split["label"].value_counts().to_string())

    # Verify no video leakage
    train_ids = set(train_df["video_id"])
    val_ids = set(val_df["video_id"])
    test_ids = set(test_df["video_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\n========== LEAKAGE CHECK ==========")
    print("Train ∩ Validation: 0 videos")
    print("Train ∩ Test:       0 videos")
    print("Validation ∩ Test:  0 videos")
    print("\nDataset split is SAFE.")


if __name__ == "__main__":
    main()
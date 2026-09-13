import pandas as pd

path = "dataset/sequence/sequence_summary.csv"

df = pd.read_csv(path)

print("\n===== SEQUENCE SUMMARY =====\n")

for class_name in df["class"].unique():

    class_df = df[df["class"] == class_name]

    print(f"\n{class_name.upper()}")

    print(
        class_df[
            ["video", "sequences"]
        ].to_string(index=False)
    )

    print(
        f"\nTotal sequences: "
        f"{class_df['sequences'].sum()}"
    )
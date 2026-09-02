import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from data_pipeline import (
    prepare_dataframe,
    create_dataset,
)


# Load verified split
df = pd.read_csv(
    "results/split_data_verified.csv"
)

# Training images only
train_df = df[
    df["Split"] == "train"
].copy()

train_df = prepare_dataframe(
    train_df
)

# Take a small sample
sample_df = (
    train_df
    .groupby("Class", group_keys=False)
    .head(2)
    .reset_index(drop=True)
)

dataset = create_dataset(
    sample_df,
    shuffle=False
)

images, labels = next(
    iter(dataset)
)

class_names = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TB",
]

plt.figure(figsize=(12, 8))

for i in range(len(images)):

    plt.subplot(2, 4, i + 1)

    plt.imshow(images[i])

    plt.title(
        class_names[int(labels[i])]
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results/preprocessing_preview.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print(
    "\nSaved:"
    "\nresults/preprocessing_preview.png"
)
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

SPLIT_PATH = PROJECT_DIR / "results" / "split_data_verified.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 32

CLASS_NAMES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TB",
]

CLASS_TO_LABEL = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# DETERMINISTIC PREPROCESSING
# ============================================================

class ResizeWithPadding:
    """
    Resize an image while preserving aspect ratio,
    then pad it to IMAGE_SIZE x IMAGE_SIZE.
    """

    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):

        width, height = image.size

        scale = min(
            self.size / width,
            self.size / height,
        )

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        padded = Image.new(
            "RGB",
            (self.size, self.size),
            (0, 0, 0),
        )

        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2

        padded.paste(
            image,
            (left, top),
        )

        return padded


# ============================================================
# TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([
    ResizeWithPadding(IMAGE_SIZE),

    # Controlled training augmentation only.
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=7
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03),
    ),

    transforms.ToTensor(),

    # ImageNet normalization used by pretrained EfficientNet.
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

eval_transform = transforms.Compose([
    ResizeWithPadding(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# DATASET
# ============================================================

class ChestXRayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None,
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["File Path"]

        label = int(row["Label"])

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not load image:\n"
                f"{image_path}\n"
                f"Error: {e}"
            )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# PREPARE DATAFRAME
# ============================================================

def prepare_dataframe(dataframe):

    df = dataframe.copy()

    df["Label"] = (
        df["Class"]
        .map(CLASS_TO_LABEL)
    )

    if df["Label"].isna().any():

        unknown_classes = (
            df.loc[
                df["Label"].isna(),
                "Class"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown classes: {unknown_classes}"
        )

    df["Label"] = df["Label"].astype(
        "int64"
    )

    return df


# ============================================================
# LOAD VERIFIED SPLIT
# ============================================================

def load_split_dataframe():

    if not SPLIT_PATH.exists():

        raise FileNotFoundError(
            f"Split file not found:\n"
            f"{SPLIT_PATH}"
        )

    df = pd.read_csv(
        SPLIT_PATH
    )

    required_columns = {
        "File Path",
        "Class",
        "File Hash",
        "Group ID",
        "Split",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    return prepare_dataframe(df)


# ============================================================
# CREATE LOADERS
# ============================================================

def create_dataloaders():

    df = load_split_dataframe()

    train_df = df[
        df["Split"] == "train"
    ].copy()

    validation_df = df[
        df["Split"] == "validation"
    ].copy()

    test_df = df[
        df["Split"] == "test"
    ].copy()

    train_dataset = ChestXRayDataset(
        train_df,
        transform=train_transform,
    )

    validation_dataset = ChestXRayDataset(
        validation_df,
        transform=eval_transform,
    )

    test_dataset = ChestXRayDataset(
        test_df,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# VERIFY DUPLICATE-GROUP LEAKAGE
# ============================================================

def verify_group_leakage(df):

    group_split_counts = (
        df
        .groupby("Group ID")["Split"]
        .nunique()
    )

    leaking_groups = (
        group_split_counts > 1
    ).sum()

    print(
        "Duplicate-group leakage:",
        leaking_groups,
    )

    if leaking_groups != 0:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED."
        )


# ============================================================
# VERIFY PIPELINE
# ============================================================

def verify_pipeline():

    print("=" * 60)
    print("FINAL PYTORCH DATA PIPELINE VERIFICATION")
    print("=" * 60)

    df = load_split_dataframe()

    print(
        "\nTotal images:",
        len(df),
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["Class"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nSplit distribution:"
    )

    print(
        df["Split"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nClass distribution by split:"
    )

    print(
        pd.crosstab(
            df["Class"],
            df["Split"],
        )
    )

    verify_group_leakage(df)

    (
        train_loader,
        validation_loader,
        test_loader,
        train_df,
        validation_df,
        test_df,
    ) = create_dataloaders()

    print("\nDataset sizes:")
    print(
        "Train:",
        len(train_df),
    )

    print(
        "Validation:",
        len(validation_df),
    )

    print(
        "Test:",
        len(test_df),
    )

    # --------------------------------------------------------
    # Training batch
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    print(
        "\nTraining batch:"
    )

    print(
        "Image shape:",
        images.shape,
    )

    print(
        "Label shape:",
        labels.shape,
    )

    print(
        "Image dtype:",
        images.dtype,
    )

    print(
        "Pixel minimum:",
        images.min().item(),
    )

    print(
        "Pixel maximum:",
        images.max().item(),
    )

    # --------------------------------------------------------
    # Validation batch
    # --------------------------------------------------------

    val_images, val_labels = next(
        iter(validation_loader)
    )

    print(
        "\nValidation batch:"
    )

    print(
        "Image shape:",
        val_images.shape,
    )

    print(
        "Label shape:",
        val_labels.shape,
    )

    # --------------------------------------------------------
    # Test batch
    # --------------------------------------------------------

    test_images, test_labels = next(
        iter(test_loader)
    )

    print(
        "\nTest batch:"
    )

    print(
        "Image shape:",
        test_images.shape,
    )

    print(
        "Label shape:",
        test_labels.shape,
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    expected_channels = 3

    if images.shape[1] != expected_channels:

        raise ValueError(
            "Images do not have 3 RGB channels."
        )

    if images.shape[2] != IMAGE_SIZE:

        raise ValueError(
            "Incorrect image height."
        )

    if images.shape[3] != IMAGE_SIZE:

        raise ValueError(
            "Incorrect image width."
        )

    if images.dtype != torch.float32:

        raise ValueError(
            "Images are not float32."
        )

    if labels.dtype != torch.int64:

        raise ValueError(
            "Labels are not int64."
        )

    unique_labels = set(
        labels.tolist()
    )

    if not unique_labels.issubset(
        set(range(len(CLASS_NAMES)))
    ):

        raise ValueError(
            "Invalid class labels."
        )

    print(
        "\nClass mapping:"
    )

    for class_name, label in CLASS_TO_LABEL.items():

        print(
            f"{label}: {class_name}"
        )

    print(
        "\n============================================================"
    )

    print(
        "FINAL PYTORCH PIPELINE VERIFICATION: PASSED"
    )

    print(
        "============================================================"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    verify_pipeline()
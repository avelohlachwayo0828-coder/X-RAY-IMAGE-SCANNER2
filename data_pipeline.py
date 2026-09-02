from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

SPLIT_PATH = PROJECT_DIR / "results" / "split_data_verified.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

CLASS_NAMES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TB",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================
#
# Preprocessing is performed dynamically when each image
# is loaded. No processed image files are created.
#
# Pipeline:
#
#   Load
#     ↓
#   RGB conversion
#     ↓
#   Resize while preserving aspect ratio
#     ↓
#   Pad to 224 x 224
#     ↓
#   Convert to tensor
#     ↓
#   ImageNet normalization
#
# Training additionally receives augmentation.
# Validation and test remain deterministic.
# ============================================================


class ResizeWithPad:
    """
    Resize an image while preserving its aspect ratio,
    then pad it to IMAGE_SIZE x IMAGE_SIZE.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image):

        width, height = image.size

        scale = min(
            self.size / width,
            self.size / height
        )

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR
        )

        padded = Image.new(
            "RGB",
            (self.size, self.size),
            (0, 0, 0)
        )

        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2

        padded.paste(
            image,
            (left, top)
        )

        return padded


# ============================================================
# NORMALIZATION
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    # Explicitly convert every image to RGB.
    transforms.Lambda(
        lambda image: image.convert("RGB")
    ),

    # Preserve aspect ratio and pad.
    ResizeWithPad(IMAGE_SIZE),

    # Mild augmentation for training only.
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=7
    ),

    transforms.ToTensor(),

    # Required for ImageNet-pretrained ResNet50.
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


evaluation_transform = transforms.Compose([

    # Explicit RGB conversion.
    transforms.Lambda(
        lambda image: image.convert("RGB")
    ),

    # Deterministic resize + padding.
    ResizeWithPad(IMAGE_SIZE),

    transforms.ToTensor(),

    # ImageNet normalization.
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ============================================================
# DATASET
# ============================================================


class ChestXrayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
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

        class_name = row["Class"]

        label = CLASS_TO_INDEX[class_name]

        # --------------------------------------------
        # Load image
        # --------------------------------------------

        try:

            image = Image.open(
                image_path
            )

            # ----------------------------------------
            # Explicit RGB conversion
            # ----------------------------------------

            image = image.convert("RGB")

        except Exception as error:

            raise RuntimeError(
                f"Could not load image:\n"
                f"{image_path}\n"
                f"Error: {error}"
            )

        # --------------------------------------------
        # Dynamic preprocessing
        # --------------------------------------------

        if self.transform is not None:

            image = self.transform(image)

        return image, label


# ============================================================
# LOAD VERIFIED DATAFRAME
# ============================================================


def load_split_dataframe():

    if not SPLIT_PATH.exists():

        raise FileNotFoundError(
            f"Verified split file not found:\n"
            f"{SPLIT_PATH}"
        )

    df = pd.read_csv(
        SPLIT_PATH
    )

    required_columns = {
        "File Path",
        "Class",
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

    # Verify classes.

    unexpected_classes = (
        set(df["Class"].unique())
        - set(CLASS_NAMES)
    )

    if unexpected_classes:

        raise ValueError(
            f"Unexpected classes: "
            f"{unexpected_classes}"
        )

    # Verify splits.

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    unexpected_splits = (
        set(df["Split"].unique())
        - expected_splits
    )

    if unexpected_splits:

        raise ValueError(
            f"Unexpected splits: "
            f"{unexpected_splits}"
        )

    return df


# ============================================================
# CREATE DATALOADERS
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

    train_dataset = ChestXrayDataset(
        train_df,
        transform=train_transform
    )

    validation_dataset = ChestXrayDataset(
        validation_df,
        transform=evaluation_transform
    )

    test_dataset = ChestXrayDataset(
        test_df,
        transform=evaluation_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
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
# PIPELINE VERIFICATION
# ============================================================


def verify_pipeline():

    print("=" * 60)
    print("DATA PIPELINE VERIFICATION")
    print("=" * 60)

    (
        train_loader,
        validation_loader,
        test_loader,
        train_df,
        validation_df,
        test_df,
    ) = create_dataloaders()

    print()
    print("Dataset sizes:")
    print(
        f"Train:      {len(train_df)}"
    )
    print(
        f"Validation: {len(validation_df)}"
    )
    print(
        f"Test:       {len(test_df)}"
    )

    # --------------------------------------------
    # Training batch
    # --------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    print()
    print("Training batch:")
    print(
        "Image shape:",
        tuple(images.shape)
    )
    print(
        "Label shape:",
        tuple(labels.shape)
    )
    print(
        "Image dtype:",
        images.dtype
    )
    print(
        "Label dtype:",
        labels.dtype
    )

    print(
        "Image minimum:",
        float(images.min())
    )

    print(
        "Image maximum:",
        float(images.max())
    )

    # --------------------------------------------
    # Verify shape
    # --------------------------------------------

    expected_shape = (
        min(BATCH_SIZE, len(train_df)),
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(images.shape) != expected_shape:

        raise ValueError(
            f"Unexpected image shape: "
            f"{tuple(images.shape)}"
        )

    # --------------------------------------------
    # Verify channels
    # --------------------------------------------

    if images.shape[1] != 3:

        raise ValueError(
            "Images do not have 3 RGB channels."
        )

    # --------------------------------------------
    # Verify labels
    # --------------------------------------------

    if labels.min() < 0:

        raise ValueError(
            "Invalid negative label."
        )

    if labels.max() >= len(CLASS_NAMES):

        raise ValueError(
            "Invalid class label."
        )

    print()
    print(
        "Classes:"
    )

    for index, name in enumerate(CLASS_NAMES):

        print(
            f"{index}: {name}"
        )

    # --------------------------------------------
    # Verify evaluation pipeline
    # --------------------------------------------

    val_images, val_labels = next(
        iter(validation_loader)
    )

    test_images, test_labels = next(
        iter(test_loader)
    )

    print()
    print(
        "Validation image shape:",
        tuple(val_images.shape)
    )

    print(
        "Test image shape:",
        tuple(test_images.shape)
    )

    print()
    print("=" * 60)
    print("PREPROCESSING PIPELINE VERIFICATION: PASSED")
    print("=" * 60)


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    verify_pipeline()
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "resnet50_final.pth"
)

SPLIT_PATH = (
    PROJECT_DIR
    / "results"
    / "split_data_verified.csv"
)

CLASS_NAMES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TB",
]

NUM_CLASSES = 4

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESNET50 TEST SET EVALUATION")
print("=" * 70)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Split file:")
print(SPLIT_PATH)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not SPLIT_PATH.exists():

    raise FileNotFoundError(
        f"Split file not found:\n{SPLIT_PATH}"
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Device:")
print(device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print("CPU evaluation")


# ============================================================
# LOAD SPLIT DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

df = pd.read_csv(
    SPLIT_PATH
)

print()
print(
    "Total images:",
    len(df)
)

print()
print("Split distribution:")
print(
    df["Split"].value_counts()
    .sort_index()
)


# ============================================================
# SELECT TEST SET
# ============================================================

test_df = df[
    df["Split"] == "test"
].copy()

test_df = test_df.reset_index(
    drop=True
)


print()
print("Test images:")
print(len(test_df))


if len(test_df) != 1074:

    raise RuntimeError(
        "Unexpected test-set size. "
        f"Expected 1074, got {len(test_df)}."
    )


# ============================================================
# VERIFY TEST CLASS DISTRIBUTION
# ============================================================

print()
print("Test class distribution:")

print(
    test_df["Class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CLASS TO LABEL
# ============================================================

class_to_label = {
    class_name: index
    for index, class_name
    in enumerate(CLASS_NAMES)
}


test_df["Label"] = (
    test_df["Class"]
    .map(class_to_label)
)


if test_df["Label"].isna().any():

    raise RuntimeError(
        "Unknown class found in test dataframe."
    )


test_df["Label"] = (
    test_df["Label"]
    .astype("int64")
)


# ============================================================
# DATASET
# ============================================================

class ChestXRayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None,
    ):

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index,
    ):

        row = self.dataframe.iloc[
            index
        ]

        image_path = Path(
            row["File Path"]
        )

        label = int(
            row["Label"]
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n"
                f"{image_path}"
            )

        image = Image.open(
            image_path
        )

        # Explicit RGB conversion.
        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# PREPROCESSING
# ============================================================

# IMPORTANT:
#
# This uses the same normalization values used by the
# ResNet50/ImageNet preprocessing pipeline.
#
# No augmentation is used for the test set.

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

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
# CREATE TEST DATASET
# ============================================================

test_dataset = ChestXRayDataset(
    test_df,
    transform=test_transform,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


print()
print("Test DataLoader created.")

print(
    "Batches:",
    len(test_loader)
)


# ============================================================
# CREATE RESNET50
# ============================================================

print()
print("=" * 70)
print("CREATING RESNET50")
print("=" * 70)

model = models.resnet50(
    weights=None
)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    NUM_CLASSES,
)


print()
print("ResNet50 created.")

print(
    "Classifier:",
    f"{model.fc.in_features} -> {NUM_CLASSES}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINED WEIGHTS")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)


# ============================================================
# EXTRACT STATE DICT
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    elif all(
        isinstance(
            value,
            torch.Tensor
        )
        for value in checkpoint.values()
    ):

        state_dict = checkpoint

    else:

        raise RuntimeError(
            "Could not find model state_dict."
        )

else:

    raise RuntimeError(
        "Unsupported checkpoint format."
    )


# ============================================================
# CLEAN STATE DICT
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        key = key[
            len("module.") :
        ]

    clean_state_dict[
        key
    ] = value


# ============================================================
# LOAD STATE DICT
# ============================================================

missing_keys, unexpected_keys = (
    model.load_state_dict(
        clean_state_dict,
        strict=False,
    )
)


print()
print("Missing keys:")
print(missing_keys)

print()
print("Unexpected keys:")
print(unexpected_keys)


if missing_keys:

    raise RuntimeError(
        "Missing model parameters."
    )

if unexpected_keys:

    raise RuntimeError(
        "Unexpected model parameters."
    )


print()
print(
    "Model weights loaded successfully."
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    device
)

model.eval()


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 70)
print("RUNNING TEST SET EVALUATION")
print("=" * 70)

all_predictions = []

all_labels = []

all_probabilities = []

total_images = 0


with torch.no_grad():

    for batch_index, (
        images,
        labels,
    ) in enumerate(test_loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        total_images += (
            images.size(0)
        )

        print(
            f"\rProcessed: "
            f"{total_images}/{len(test_dataset)}",
            end="",
            flush=True,
        )


print()


# ============================================================
# CONVERT RESULTS
# ============================================================

y_true = np.array(
    all_labels
)

y_pred = np.array(
    all_predictions
)

probabilities = np.array(
    all_probabilities
)


# ============================================================
# VERIFY NUMBER OF PREDICTIONS
# ============================================================

if len(y_true) != len(test_df):

    raise RuntimeError(
        "Number of predictions does not "
        "match test-set size."
    )


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=list(
        range(NUM_CLASSES)
    ),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(
        range(NUM_CLASSES)
    ),
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("TEST SET RESULTS")
print("=" * 70)

print()
print(
    f"Test images: {len(y_true)}"
)

print()
print(
    f"Overall accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Overall accuracy: "
    f"{accuracy * 100:.2f}%"
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    " " * 14
    + "".join(
        f"{name[:10]:>12}"
        for name in CLASS_NAMES
    )
)

for index, class_name in enumerate(
    CLASS_NAMES
):

    row = cm[index]

    print(
        f"{class_name:<14}"
        + "".join(
            f"{value:>12}"
            for value in row
        )
    )


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print()
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

for index, class_name in enumerate(
    CLASS_NAMES
):

    actual_count = int(
        cm[index].sum()
    )

    correct_count = int(
        cm[index, index]
    )

    if actual_count > 0:

        class_accuracy = (
            correct_count
            / actual_count
        )

    else:

        class_accuracy = 0.0

    print()

    print(
        f"{class_name}:"
    )

    print(
        f"  Actual samples: "
        f"{actual_count}"
    )

    print(
        f"  Correct: "
        f"{correct_count}"
    )

    print(
        f"  Accuracy: "
        f"{class_accuracy * 100:.2f}%"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results_df = test_df.copy()

results_df[
    "Predicted Label"
] = y_pred

results_df[
    "Predicted Class"
] = [
    CLASS_NAMES[index]
    for index in y_pred
]


for index, class_name in enumerate(
    CLASS_NAMES
):

    results_df[
        f"Probability_{class_name}"
    ] = probabilities[
        :,
        index
    ]


results_df[
    "Correct"
] = (
    results_df["Label"]
    == results_df["Predicted Label"]
)


predictions_path = (
    PROJECT_DIR
    / "results"
    / "test_predictions.csv"
)


results_df.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)


cm_path = (
    PROJECT_DIR
    / "results"
    / "test_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_path
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()
print(
    "Predictions saved:"
)

print(
    predictions_path
)

print()
print(
    "Confusion matrix saved:"
)

print(
    cm_path
)

print()
print(
    "The complete 1,074-image test set "
    "was evaluated."
)

print()
print("=" * 70)
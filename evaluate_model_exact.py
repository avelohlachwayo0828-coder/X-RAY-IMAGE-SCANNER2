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

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
print("RESNET50 — EXACT TRAINING PREPROCESSING EVALUATION")
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
# LOAD VERIFIED SPLIT
# ============================================================

print()
print("=" * 70)
print("LOADING VERIFIED DATA SPLIT")
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
    df["Split"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SELECT TEST SET
# ============================================================

test_df = (
    df[
        df["Split"] == "test"
    ]
    .copy()
    .reset_index(drop=True)
)


print()
print(
    "Test images:",
    len(test_df)
)


if len(test_df) != 1074:

    raise RuntimeError(
        "Expected 1074 test images, "
        f"found {len(test_df)}."
    )


# ============================================================
# CLASS MAPPING
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
# TEST DISTRIBUTION
# ============================================================

print()
print("Test class distribution:")

print(
    test_df["Class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# EXACT TRAINING PREPROCESSING
# ============================================================
#
# Training pipeline:
#
#   1. Read image
#   2. Convert to RGB
#   3. Resize while preserving aspect ratio
#   4. Pad to 224 x 224
#   5. Convert to tensor
#   6. ImageNet normalization
#
# No augmentation is used for validation/test.
#
# IMPORTANT:
# The model was trained with this preprocessing strategy.
# ============================================================

class ExactPreprocessing:

    def __init__(self):

        self.resize_with_pad = transforms.Resize(
            IMAGE_SIZE
        )

        self.to_tensor = (
            transforms.ToTensor()
        )

        self.normalize = (
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
            )
        )

    def __call__(self, image):

        # ----------------------------------------------------
        # Preserve aspect ratio
        # ----------------------------------------------------

        image.thumbnail(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            Image.Resampling.BILINEAR,
        )

        # ----------------------------------------------------
        # Create 224 x 224 canvas
        # ----------------------------------------------------

        canvas = Image.new(
            "RGB",
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            (0, 0, 0),
        )

        # Center image on canvas.
        x = (
            IMAGE_SIZE
            - image.width
        ) // 2

        y = (
            IMAGE_SIZE
            - image.height
        ) // 2

        canvas.paste(
            image,
            (
                x,
                y,
            ),
        )

        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        tensor = self.to_tensor(
            canvas
        )

        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        tensor = self.normalize(
            tensor
        )

        return tensor


test_transform = ExactPreprocessing()


# ============================================================
# DATASET
# ============================================================

class ChestXRayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform,
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

        else:

            image = image.copy()

        image = self.transform(
            image
        )

        return (
            image,
            label,
        )


# ============================================================
# CREATE DATASET
# ============================================================

test_dataset = ChestXRayDataset(
    test_df,
    transform=test_transform,
)


# ============================================================
# CREATE DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


print()
print(
    "Test DataLoader created."
)

print(
    "Batches:",
    len(test_loader)
)


# ============================================================
# PREPROCESSING VERIFICATION
# ============================================================

print()
print("=" * 70)
print("VERIFYING EXACT PREPROCESSING")
print("=" * 70)

sample_images, sample_labels = next(
    iter(test_loader)
)


print()
print(
    "Batch image shape:",
    sample_images.shape
)

print(
    "Batch label shape:",
    sample_labels.shape
)

print(
    "Image dtype:",
    sample_images.dtype
)

print(
    "Image minimum:",
    float(
        sample_images.min()
    )
)

print(
    "Image maximum:",
    float(
        sample_images.max()
    )
)


expected_shape = (
    BATCH_SIZE,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)


if tuple(sample_images.shape) != expected_shape:

    raise RuntimeError(
        "Unexpected image batch shape."
    )


if sample_images.dtype != torch.float32:

    raise RuntimeError(
        "Images are not float32."
    )


print()
print(
    "Preprocessing verification: PASSED"
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
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINED MODEL")
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

        print(
            "Checkpoint format:",
            "model_state_dict"
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

        print(
            "Checkpoint format:",
            "state_dict"
        )

    elif all(
        isinstance(
            value,
            torch.Tensor,
        )
        for value in checkpoint.values()
    ):

        state_dict = checkpoint

        print(
            "Checkpoint format:",
            "direct state_dict"
        )

    else:

        raise RuntimeError(
            "Could not identify state_dict."
        )

else:

    raise RuntimeError(
        "Unsupported checkpoint format."
    )


# ============================================================
# REMOVE DataParallel PREFIX
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
# LOAD WEIGHTS
# ============================================================

missing_keys, unexpected_keys = (
    model.load_state_dict(
        clean_state_dict,
        strict=False,
    )
)


print()
print(
    "Missing keys:",
    missing_keys
)

print(
    "Unexpected keys:",
    unexpected_keys
)


if missing_keys:

    raise RuntimeError(
        "Model contains missing parameters."
    )

if unexpected_keys:

    raise RuntimeError(
        "Model contains unexpected parameters."
    )


print()
print(
    "Model weights loaded successfully."
)


# ============================================================
# MOVE TO DEVICE
# ============================================================

model = model.to(
    device
)

model.eval()


# ============================================================
# RUN EVALUATION
# ============================================================

print()
print("=" * 70)
print("RUNNING EXACT TEST SET EVALUATION")
print("=" * 70)

all_labels = []

all_predictions = []

all_probabilities = []

processed = 0


with torch.no_grad():

    for images, labels in test_loader:

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

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        processed += (
            images.size(0)
        )

        print(
            f"\rProcessed: "
            f"{processed}/{len(test_dataset)}",
            end="",
            flush=True,
        )


print()


# ============================================================
# NUMPY ARRAYS
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
# VERIFY OUTPUT COUNT
# ============================================================

if len(y_true) != 1074:

    raise RuntimeError(
        "Expected 1074 predictions."
    )

if len(y_pred) != 1074:

    raise RuntimeError(
        "Expected 1074 predictions."
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

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(
        range(NUM_CLASSES)
    ),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
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
# RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL TEST RESULTS")
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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()
print(report_text)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print("Rows = Actual")
print("Columns = Predicted")
print()

print(
    f"{'Actual':<14}"
    + "".join(
        f"{name:>12}"
        for name in CLASS_NAMES
    )
)

for index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_name:<14}"
        + "".join(
            f"{value:>12}"
            for value in cm[index]
        )
    )


# ============================================================
# PER-CLASS SUMMARY
# ============================================================

print()
print("=" * 70)
print("PER-CLASS SUMMARY")
print("=" * 70)

for class_name in CLASS_NAMES:

    metrics = report_dict[
        class_name
    ]

    print()
    print(
        class_name
    )

    print(
        f"  Precision: "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"  Recall:    "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"  F1-score:  "
        f"{metrics['f1-score'] * 100:.2f}%"
    )

    print(
        f"  Support:   "
        f"{int(metrics['support'])}"
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
    RESULTS_DIR
    / "test_predictions_exact.csv"
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
    RESULTS_DIR
    / "test_confusion_matrix_exact.csv"
)


cm_df.to_csv(
    cm_path
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = (
    RESULTS_DIR
    / "test_metrics_exact.txt"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "RESNET50 EXACT TEST SET EVALUATION\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )

    f.write(
        f"Test images: {len(y_true)}\n"
    )

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Accuracy percentage: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    f.write(
        report_text
    )

    f.write(
        "\n\nCONFUSION MATRIX\n"
    )

    f.write(
        "Rows = Actual\n"
    )

    f.write(
        "Columns = Predicted\n\n"
    )

    f.write(
        str(cm_df)
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("EXACT EVALUATION COMPLETE")
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
    "Metrics saved:"
)

print(
    metrics_path
)

print()
print(
    "All 1,074 test images were evaluated "
    "using the training-matched preprocessing."
)

print()
print("=" * 70)
# ============================================================
# train_final.py
#
# ResNet50 FINAL TRAINING
#
# Stage 1:
#   Freeze ResNet50 backbone
#   Train classifier
#
# Stage 2:
#   Unfreeze layer4
#   Fine-tune layer4 + classifier
#
# Preprocessing:
#   Handled by data_pipeline.py while images are loaded.
#
# Classes:
#   0 = COVID19
#   1 = NORMAL
#   2 = PNEUMONIA
#   3 = TB
# ============================================================

from pathlib import Path
from copy import deepcopy
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from data_pipeline import (
    create_dataloaders,
    CLASS_NAMES,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_DIR / "results"

TRAINING_DIR = (
    RESULTS_DIR / "final_training"
)

TRAINING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

NUM_CLASSES = 4

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 8

STAGE1_LR = 0.001
STAGE2_LR = 0.00001

WEIGHT_DECAY = 0.0001

EARLY_STOPPING_PATIENCE = 3

LR_PATIENCE = 2
LR_FACTOR = 0.5

RANDOM_SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("RESNET50 FINAL TRAINING")
print("=" * 70)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("PyTorch:")
print(torch.__version__)

print()
print("Device:")
print(DEVICE)

print()
print("CUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():

    print()
    print("GPU:")
    print(
        torch.cuda.get_device_name(0)
    )

else:

    print()
    print("GPU:")
    print("None — CPU training")


print()
print("Classes:")

for index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{index}: {class_name}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)


# IMPORTANT:
#
# create_dataloaders() returns SIX values:
#
#   1. train_loader
#   2. validation_loader
#   3. test_loader
#   4. train_df
#   5. validation_df
#   6. test_df
#
# ============================================================

(
    train_loader,
    validation_loader,
    test_loader,
    train_df,
    validation_df,
    test_df,
) = create_dataloaders()


print()
print("DataLoaders created successfully.")


# ============================================================
# DATASET SIZES
# ============================================================

train_size = len(
    train_loader.dataset
)

validation_size = len(
    validation_loader.dataset
)

test_size = len(
    test_loader.dataset
)


print()
print("Dataset sizes:")

print(
    f"Train:      {train_size}"
)

print(
    f"Validation: {validation_size}"
)

print(
    f"Test:       {test_size}"
)


# ============================================================
# VERIFY DATAFRAMES
# ============================================================

print()
print("=" * 70)
print("DATAFRAME VERIFICATION")
print("=" * 70)

print()
print(
    f"Train dataframe:      "
    f"{len(train_df)}"
)

print(
    f"Validation dataframe: "
    f"{len(validation_df)}"
)

print(
    f"Test dataframe:       "
    f"{len(test_df)}"
)


# ============================================================
# VERIFY TRAINING BATCH
# ============================================================

print()
print("=" * 70)
print("VERIFYING TRAINING DATA")
print("=" * 70)


sample_images, sample_labels = next(
    iter(train_loader)
)


print()
print(
    "Image shape:",
    sample_images.shape
)

print(
    "Label shape:",
    sample_labels.shape
)

print(
    "Image dtype:",
    sample_images.dtype
)

print(
    "Label dtype:",
    sample_labels.dtype
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


# ============================================================
# VERIFY IMAGE FORMAT
# ============================================================

if sample_images.ndim != 4:

    raise RuntimeError(
        "Unexpected image tensor dimensions."
    )


if sample_images.shape[1] != 3:

    raise RuntimeError(
        "Expected 3-channel RGB images."
    )


if sample_images.shape[2] != 224:

    raise RuntimeError(
        "Expected image height of 224."
    )


if sample_images.shape[3] != 224:

    raise RuntimeError(
        "Expected image width of 224."
    )


if sample_images.dtype != torch.float32:

    raise RuntimeError(
        "Expected float32 image tensors."
    )


print()
print(
    "Data pipeline check: PASSED"
)


# ============================================================
# CREATE RESNET50
# ============================================================

print()
print("=" * 70)
print("CREATING RESNET50")
print("=" * 70)


print()
print(
    "Loading ImageNet pretrained ResNet50..."
)


weights = (
    models.ResNet50_Weights.DEFAULT
)


model = models.resnet50(
    weights=weights
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

num_features = (
    model.fc.in_features
)


model.fc = nn.Sequential(

    nn.Dropout(
        p=0.4
    ),

    nn.Linear(
        num_features,
        NUM_CLASSES
    )

)


model = model.to(
    DEVICE
)


print()
print(
    "ResNet50 classifier replaced."
)

print(
    f"Classifier: "
    f"{num_features} -> {NUM_CLASSES}"
)


# ============================================================
# CALCULATE CLASS COUNTS
# ============================================================

print()
print("=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)


class_counts = torch.zeros(
    NUM_CLASSES,
    dtype=torch.float32
)


print()
print(
    "Scanning training labels..."
)


for _, labels in train_loader:

    for label in labels:

        class_index = int(
            label.item()
        )

        if (
            class_index < 0
            or
            class_index >= NUM_CLASSES
        ):

            raise RuntimeError(
                f"Invalid class label: "
                f"{class_index}"
            )

        class_counts[
            class_index
        ] += 1


print()
print(
    "Training class distribution:"
)


for index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{index}: "
        f"{class_name:12s} "
        f"{int(class_counts[index])}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_samples = (
    class_counts.sum()
)


class_weights = (
    total_samples
    /
    (
        NUM_CLASSES
        *
        class_counts
    )
)


class_weights = class_weights.to(
    DEVICE
)


print()
print("Class weights:")


for index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_name:12s}: "
        f"{class_weights[index].item():.4f}"
    )


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# FREEZE BACKBONE
# ============================================================

def freeze_backbone(model):

    """
    Stage 1.

    Freeze the ResNet50 convolutional
    backbone.

    Only the final classifier is trained.
    """

    for parameter in model.parameters():

        parameter.requires_grad = False


    for parameter in model.fc.parameters():

        parameter.requires_grad = True


# ============================================================
# UNFREEZE LAYER4
# ============================================================

def unfreeze_layer4(model):

    """
    Stage 2.

    Freeze earlier ResNet50 layers.

    Train:
        layer4
        final classifier
    """

    for parameter in model.parameters():

        parameter.requires_grad = False


    for parameter in model.layer4.parameters():

        parameter.requires_grad = True


    for parameter in model.fc.parameters():

        parameter.requires_grad = True


# ============================================================
# PARAMETER STATUS
# ============================================================

def print_parameter_status(model):

    total_parameters = 0

    trainable_parameters = 0

    frozen_parameters = 0


    for parameter in model.parameters():

        count = parameter.numel()

        total_parameters += count


        if parameter.requires_grad:

            trainable_parameters += count

        else:

            frozen_parameters += count


    print()

    print(
        f"Total parameters:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen parameters:    "
        f"{frozen_parameters:,}"
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    num_epochs,
    patience,
    checkpoint_path,
    stage_name,
):

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=LR_FACTOR,

            patience=LR_PATIENCE,

        )
    )


    history = {

        "train_loss": [],

        "train_acc": [],

        "val_loss": [],

        "val_acc": [],

        "learning_rate": [],

        "epoch_time_seconds": [],

    }


    best_val_loss = float(
        "inf"
    )


    best_model_weights = deepcopy(
        model.state_dict()
    )


    epochs_no_improve = 0


    print()
    print("=" * 70)
    print(stage_name)
    print("=" * 70)


    for epoch in range(
        num_epochs
    ):

        start_time = time.time()


        # ====================================================
        # TRAINING
        # ====================================================

        model.train()


        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            optimizer.step()


            batch_size = (
                images.size(0)
            )


            running_loss += (
                loss.item()
                *
                batch_size
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += batch_size


        train_loss = (
            running_loss
            /
            total
        )


        train_acc = (
            correct
            /
            total
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        val_running_loss = 0.0

        val_correct = 0

        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device
                )

                labels = labels.to(
                    device
                )


                outputs = model(
                    images
                )


                loss = criterion(
                    outputs,
                    labels
                )


                batch_size = (
                    images.size(0)
                )


                val_running_loss += (
                    loss.item()
                    *
                    batch_size
                )


                predictions = torch.argmax(
                    outputs,
                    dim=1
                )


                val_correct += (
                    predictions == labels
                ).sum().item()


                val_total += batch_size


        val_loss = (
            val_running_loss
            /
            val_total
        )


        val_acc = (
            val_correct
            /
            val_total
        )


        # ====================================================
        # LEARNING RATE SCHEDULER
        # ====================================================

        scheduler.step(
            val_loss
        )


        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        history[
            "train_loss"
        ].append(
            train_loss
        )


        history[
            "train_acc"
        ].append(
            train_acc
        )


        history[
            "val_loss"
        ].append(
            val_loss
        )


        history[
            "val_acc"
        ].append(
            val_acc
        )


        history[
            "learning_rate"
        ].append(
            current_lr
        )


        elapsed = (
            time.time()
            -
            start_time
        )


        history[
            "epoch_time_seconds"
        ].append(
            elapsed
        )


        # ====================================================
        # PRINT EPOCH
        # ====================================================

        print()
        print(
            f"Epoch "
            f"{epoch + 1}/{num_epochs}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_acc:.4f} "
            f"({train_acc * 100:.2f}%)"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_acc:.4f} "
            f"({val_acc * 100:.2f}%)"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.8f}"
        )

        print(
            f"Time: "
            f"{elapsed:.1f}s"
        )


        # ====================================================
        # BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )


            best_model_weights = deepcopy(
                model.state_dict()
            )


            epochs_no_improve = 0


            torch.save(

                {

                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        CLASS_NAMES,

                    "architecture":
                        "ResNet50",

                    "best_val_loss":
                        best_val_loss,

                    "epoch":
                        epoch + 1,

                    "stage":
                        stage_name,

                },

                checkpoint_path,

            )


            print(
                "Validation loss improved."
            )

            print(
                "Best model saved."
            )


        else:

            epochs_no_improve += 1


            print(
                f"No improvement: "
                f"{epochs_no_improve}/"
                f"{patience}"
            )


            if (
                epochs_no_improve
                >=
                patience
            ):

                print()

                print(
                    "Early stopping triggered."
                )

                break


    # ========================================================
    # RESTORE BEST WEIGHTS
    # ========================================================

    model.load_state_dict(
        best_model_weights
    )


    print()
    print(
        f"{stage_name} complete."
    )


    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )


    return model, history


# ============================================================
# STAGE 1 — FROZEN BACKBONE
# ============================================================

freeze_backbone(
    model
)


print()
print("=" * 70)
print("STAGE 1 — PARAMETER STATUS")
print("=" * 70)


print_parameter_status(
    model
)


optimizer_stage1 = optim.AdamW(

    filter(
        lambda parameter:
            parameter.requires_grad,

        model.parameters(),
    ),

    lr=STAGE1_LR,

    weight_decay=WEIGHT_DECAY,
)


stage1_checkpoint = (
    TRAINING_DIR
    /
    "stage1_best.pth"
)


model, history_stage1 = train_model(

    model=model,

    optimizer=optimizer_stage1,

    criterion=criterion,

    train_loader=train_loader,

    val_loader=validation_loader,

    device=DEVICE,

    num_epochs=STAGE1_EPOCHS,

    patience=EARLY_STOPPING_PATIENCE,

    checkpoint_path=str(
        stage1_checkpoint
    ),

    stage_name=(
        "STAGE 1 — "
        "FROZEN RESNET50 BACKBONE"
    ),
)


# ============================================================
# STAGE 2 — FINE TUNING
# ============================================================

unfreeze_layer4(
    model
)


print()
print("=" * 70)
print("STAGE 2 — PARAMETER STATUS")
print("=" * 70)


print_parameter_status(
    model
)


optimizer_stage2 = optim.AdamW(

    filter(
        lambda parameter:
            parameter.requires_grad,

        model.parameters(),
    ),

    lr=STAGE2_LR,

    weight_decay=WEIGHT_DECAY,
)


stage2_checkpoint = (
    TRAINING_DIR
    /
    "stage2_best.pth"
)


model, history_stage2 = train_model(

    model=model,

    optimizer=optimizer_stage2,

    criterion=criterion,

    train_loader=train_loader,

    val_loader=validation_loader,

    device=DEVICE,

    num_epochs=STAGE2_EPOCHS,

    patience=EARLY_STOPPING_PATIENCE,

    checkpoint_path=str(
        stage2_checkpoint
    ),

    stage_name=(
        "STAGE 2 — "
        "FINE-TUNING RESNET50 LAYER4"
    ),
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    TRAINING_DIR
    /
    "resnet50_final.pth"
)


torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "class_names":
            CLASS_NAMES,

        "architecture":
            "ResNet50",

        "image_size":
            224,

        "pretrained":
            True,

        "stage1_epochs":
            STAGE1_EPOCHS,

        "stage2_epochs":
            STAGE2_EPOCHS,

    },

    final_model_path,

)


print()
print("=" * 70)
print("FINAL MODEL SAVED")
print("=" * 70)

print()
print(final_model_path)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


model.eval()


test_running_loss = 0.0

test_correct = 0

test_total = 0


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = torch.zeros(

    NUM_CLASSES,

    NUM_CLASSES,

    dtype=torch.int64,

)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        batch_size = (
            images.size(0)
        )


        test_running_loss += (
            loss.item()
            *
            batch_size
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        test_correct += (
            predictions == labels
        ).sum().item()


        test_total += (
            batch_size
        )


        for (
            true_label,
            predicted_label
        ) in zip(

            labels.cpu(),

            predictions.cpu(),

        ):

            confusion_matrix[
                int(true_label),
                int(predicted_label),
            ] += 1


test_loss = (
    test_running_loss
    /
    test_total
)


test_accuracy = (
    test_correct
    /
    test_total
)


print()
print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

print()
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)


per_class_results = {}


for index, class_name in enumerate(
    CLASS_NAMES
):

    true_positive = (
        confusion_matrix[
            index,
            index
        ].item()
    )


    actual = (
        confusion_matrix[
            index
        ].sum().item()
    )


    predicted = (
        confusion_matrix[
            :,
            index
        ].sum().item()
    )


    if actual > 0:

        recall = (
            true_positive
            /
            actual
        )

    else:

        recall = 0.0


    if predicted > 0:

        precision = (
            true_positive
            /
            predicted
        )

    else:

        precision = 0.0


    if (
        precision + recall
        >
        0
    ):

        f1 = (

            2
            *
            precision
            *
            recall

            /

            (
                precision
                +
                recall
            )
        )

    else:

        f1 = 0.0


    per_class_results[
        class_name
    ] = {

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "support":
            actual,

    }


    print()
    print(
        class_name
    )

    print(
        f"  Precision: "
        f"{precision:.4f}"
    )

    print(
        f"  Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"  F1:        "
        f"{f1:.4f}"
    )

    print(
        f"  Support:   "
        f"{actual}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print(
    "Rows    = Actual"
)

print(
    "Columns = Predicted"
)

print()


header = (
    "Actual       "
    +
    " ".join(
        f"{name:12s}"
        for name in CLASS_NAMES
    )
)


print(
    header
)


for index, class_name in enumerate(
    CLASS_NAMES
):

    values = (
        confusion_matrix[
            index
        ].tolist()
    )


    row = (
        f"{class_name:12s}"
        +
        " ".join(
            f"{value:12d}"
            for value in values
        )
    )


    print(
        row
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "architecture":
        "ResNet50",

    "device":
        str(DEVICE),

    "pytorch_version":
        torch.__version__,

    "class_names":
        CLASS_NAMES,

    "train_size":
        train_size,

    "validation_size":
        validation_size,

    "test_size":
        test_size,

    "class_counts":
        {

            CLASS_NAMES[i]:
                int(
                    class_counts[i].item()
                )

            for i in range(
                NUM_CLASSES
            )

        },

    "class_weights":
        {

            CLASS_NAMES[i]:
                float(
                    class_weights[i].item()
                )

            for i in range(
                NUM_CLASSES
            )

        },

    "test_loss":
        test_loss,

    "test_accuracy":
        test_accuracy,

    "test_accuracy_percent":
        test_accuracy * 100,

    "per_class":
        per_class_results,

    "confusion_matrix":
        confusion_matrix.tolist(),

    "stage1_history":
        history_stage1,

    "stage2_history":
        history_stage2,

}


results_path = (
    TRAINING_DIR
    /
    "training_results.json"
)


with open(
    results_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=4,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print("Stage 1 best:")
print(stage1_checkpoint)

print()
print("Stage 2 best:")
print(stage2_checkpoint)

print()
print("Final model:")
print(final_model_path)

print()
print("Training results:")
print(results_path)

print()
print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("=" * 70)
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 4


# ============================================================
# BUILD RESNET50
# ============================================================

def create_resnet50():

    # Load ImageNet-pretrained ResNet50.
    weights = ResNet50_Weights.DEFAULT

    model = resnet50(
        weights=weights
    )

    # --------------------------------------------------------
    # Replace the original ImageNet classifier.
    #
    # Original:
    #       2048 -> 1000
    #
    # Our problem:
    #       2048 -> 4
    # --------------------------------------------------------

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# FREEZE BACKBONE
# ============================================================

def freeze_backbone(model):

    for parameter in model.parameters():

        parameter.requires_grad = False

    # Only the new classifier is trainable.
    for parameter in model.fc.parameters():

        parameter.requires_grad = True


# ============================================================
# UNFREEZE LAST RESNET BLOCK
# ============================================================

def unfreeze_last_block(model):

    # Keep earlier layers frozen.

    for parameter in model.parameters():

        parameter.requires_grad = False

    # Fine-tune the final ResNet block.
    for parameter in model.layer4.parameters():

        parameter.requires_grad = True

    # Always train the classifier.
    for parameter in model.fc.parameters():

        parameter.requires_grad = True


# ============================================================
# PRINT TRAINABLE PARAMETERS
# ============================================================

def print_parameter_status(model):

    total = 0
    trainable = 0

    for parameter in model.parameters():

        parameter_count = parameter.numel()

        total += parameter_count

        if parameter.requires_grad:

            trainable += parameter_count

    print()
    print("Model parameter summary")
    print("-" * 40)
    print(
        f"Total parameters:     {total:,}"
    )
    print(
        f"Trainable parameters: {trainable:,}"
    )
    print(
        f"Frozen parameters:    {total - trainable:,}"
    )
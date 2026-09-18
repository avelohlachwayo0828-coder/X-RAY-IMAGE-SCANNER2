from pathlib import Path

import torch
from torchvision import models


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "resnet50_final.pth"

CLASS_NAMES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TB",
]

NUM_CLASSES = 4


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RESNET50 MODEL VERIFICATION")
print("=" * 70)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("Model:")
print(MODEL_PATH)

print()
print("PyTorch:")
print(torch.__version__)

print()
print("CUDA available:")
print(torch.cuda.is_available())


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print()
print("Model file exists: YES")

print(
    "Model size:",
    f"{MODEL_PATH.stat().st_size / (1024 ** 2):.2f} MB"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print()
print("Device:")
print(device)


# ============================================================
# CREATE RESNET50 ARCHITECTURE
# ============================================================

print()
print("=" * 70)
print("CREATING RESNET50 ARCHITECTURE")
print("=" * 70)

model = models.resnet50(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    NUM_CLASSES,
)

print()
print("Architecture:")
print("ResNet50")

print()
print("Classifier:")
print(
    f"{model.fc.in_features} -> {NUM_CLASSES}"
)

print()
print("Classes:")

for index, class_name in enumerate(CLASS_NAMES):
    print(f"{index}: {class_name}")


# ============================================================
# LOAD CHECKPOINT
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

print()
print("Checkpoint type:")
print(type(checkpoint))


# ============================================================
# HANDLE DIFFERENT CHECKPOINT FORMATS
# ============================================================

if isinstance(checkpoint, dict):

    print()
    print("Checkpoint keys:")

    for key in checkpoint.keys():
        print(f" - {key}")

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

        print()
        print("Using:")
        print("checkpoint['model_state_dict']")

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

        print()
        print("Using:")
        print("checkpoint['state_dict']")

    else:

        # Check whether the dictionary itself looks
        # like a PyTorch state_dict.
        if all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):

            state_dict = checkpoint

            print()
            print("Using checkpoint directly as state_dict.")

        else:

            raise RuntimeError(
                "Could not identify model state_dict "
                "inside checkpoint."
            )

else:

    raise RuntimeError(
        "Unsupported checkpoint format."
    )


# ============================================================
# REMOVE POSSIBLE DataParallel PREFIX
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[len("module."):]

    clean_state_dict[key] = value


# ============================================================
# LOAD STATE DICT
# ============================================================

print()
print("Loading state_dict...")

missing_keys, unexpected_keys = model.load_state_dict(
    clean_state_dict,
    strict=False,
)


# ============================================================
# CHECK LOADING
# ============================================================

print()
print("Missing keys:")
print(missing_keys)

print()
print("Unexpected keys:")
print(unexpected_keys)


if missing_keys:
    raise RuntimeError(
        "Model has missing parameters."
    )

if unexpected_keys:
    raise RuntimeError(
        "Model has unexpected parameters."
    )

print()
print("Weights loaded successfully.")


# ============================================================
# PARAMETER STATUS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()
print("=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)

print()
print(
    f"Total parameters:     {total_parameters:,}"
)

print(
    f"Trainable parameters: {trainable_parameters:,}"
)


# ============================================================
# CLASSIFIER VERIFICATION
# ============================================================

print()
print("=" * 70)
print("CLASSIFIER VERIFICATION")
print("=" * 70)

classifier_weight_shape = tuple(
    model.fc.weight.shape
)

classifier_bias_shape = tuple(
    model.fc.bias.shape
)

print()
print("Classifier weight shape:")
print(classifier_weight_shape)

print()
print("Classifier bias shape:")
print(classifier_bias_shape)


expected_weight_shape = (
    NUM_CLASSES,
    2048,
)

expected_bias_shape = (
    NUM_CLASSES,
)


if classifier_weight_shape != expected_weight_shape:

    raise RuntimeError(
        "Unexpected classifier weight shape."
    )

if classifier_bias_shape != expected_bias_shape:

    raise RuntimeError(
        "Unexpected classifier bias shape."
    )

print()
print("4-class classifier verification: PASSED")


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(device)
model.eval()


# ============================================================
# DUMMY IMAGE INFERENCE
# ============================================================

print()
print("=" * 70)
print("DUMMY IMAGE INFERENCE")
print("=" * 70)

dummy_image = torch.randn(
    1,
    3,
    224,
    224,
    device=device,
)

with torch.no_grad():

    output = model(dummy_image)

print()
print("Input shape:")
print(dummy_image.shape)

print()
print("Output shape:")
print(output.shape)


expected_output_shape = (
    1,
    NUM_CLASSES,
)

if tuple(output.shape) != expected_output_shape:

    raise RuntimeError(
        f"Unexpected output shape: {output.shape}"
    )


# ============================================================
# SOFTMAX
# ============================================================

probabilities = torch.softmax(
    output,
    dim=1,
)

predicted_index = int(
    torch.argmax(
        probabilities,
        dim=1,
    ).item()
)

predicted_class = CLASS_NAMES[
    predicted_index
]


# ============================================================
# OUTPUT
# ============================================================

print()
print("Probabilities:")

for index, class_name in enumerate(CLASS_NAMES):

    probability = float(
        probabilities[0, index]
    )

    print(
        f"{class_name:<12}: "
        f"{probability:.4f}"
    )

print()
print("Predicted class from dummy input:")
print(predicted_class)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("MODEL VERIFICATION: PASSED")
print("=" * 70)

print()
print("ResNet50 architecture:      OK")
print("4-class classifier:         OK")
print("Weights loaded:             OK")
print("Parameter structure:        OK")
print("224x224 input:              OK")
print("3-channel input:            OK")
print("Inference:                  OK")

print()
print("Classes:")

for index, class_name in enumerate(CLASS_NAMES):
    print(f"{index}: {class_name}")

print()
print("Model is ready for real-image inference.")
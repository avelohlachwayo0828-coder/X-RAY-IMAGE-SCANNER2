from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "resnet50_final.pth"
)

IMAGE_PATH = (
    PROJECT_DIR
    / "clinical-support-system"
    / "static"
    / "uploads"
    / "CASE-20260829134636857670_Figure_1.png"
)

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
print("RESNET50 REAL IMAGE INFERENCE TEST")
print("=" * 70)

print()
print("Project:")
print(PROJECT_DIR)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Test image:")
print(IMAGE_PATH)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


print()
print("Model file: OK")
print("Image file: OK")


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

    print("GPU: CPU")


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
print("LOADING TRAINED MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)


# ============================================================
# FIND STATE DICTIONARY
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print()
        print(
            "Checkpoint format:",
            "model_state_dict"
        )

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

        print()
        print(
            "Checkpoint format:",
            "state_dict"
        )

    elif all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):

        state_dict = checkpoint

        print()
        print(
            "Checkpoint format:",
            "direct state_dict"
        )

    else:

        raise RuntimeError(
            "Could not find model state_dict."
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

    if key.startswith("module."):

        key = key[len("module."):]

    clean_state_dict[key] = value


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
print("Missing keys:")
print(missing_keys)

print()
print("Unexpected keys:")
print(unexpected_keys)


if missing_keys:

    raise RuntimeError(
        "Model contains missing parameters."
    )

if unexpected_keys:

    raise RuntimeError(
        "Model contains unexpected parameters."
    )


print()
print("Model weights loaded successfully.")


# ============================================================
# MODEL MODE
# ============================================================

model = model.to(device)

model.eval()


# ============================================================
# PREPROCESSING
# ============================================================

print()
print("=" * 70)
print("PREPROCESSING IMAGE")
print("=" * 70)

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
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
# LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
)

print()
print("Original image mode:")
print(image.mode)

print()
print("Original image size:")
print(image.size)


# ============================================================
# RGB CONVERSION
# ============================================================

if image.mode != "RGB":

    image = image.convert(
        "RGB"
    )

    print()
    print("Converted image to RGB.")


# ============================================================
# APPLY PREPROCESSING
# ============================================================

image_tensor = transform(
    image
)

image_tensor = image_tensor.unsqueeze(
    0
)

image_tensor = image_tensor.to(
    device
)


print()
print("Processed tensor shape:")
print(image_tensor.shape)

print()
print("Processed tensor dtype:")
print(image_tensor.dtype)

print()
print("Processed minimum:")
print(
    float(
        image_tensor.min()
    )
)

print()
print("Processed maximum:")
print(
    float(
        image_tensor.max()
    )
)


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 70)
print("RUNNING MODEL INFERENCE")
print("=" * 70)

with torch.no_grad():

    outputs = model(
        image_tensor
    )


# ============================================================
# SOFTMAX
# ============================================================

probabilities = torch.softmax(
    outputs,
    dim=1,
)

predicted_index = int(
    torch.argmax(
        probabilities,
        dim=1,
    ).item()
)

predicted_class = (
    CLASS_NAMES[
        predicted_index
    ]
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)

print()
print(
    "Predicted class:"
)

print(
    predicted_class
)

print()
print(
    "Predicted class index:"
)

print(
    predicted_index
)

print()
print(
    "Class probabilities:"
)

for index, class_name in enumerate(
    CLASS_NAMES
):

    probability = float(
        probabilities[
            0,
            index
        ]
    )

    print(
        f"{class_name:<12}: "
        f"{probability:.4f} "
        f"({probability * 100:.2f}%)"
    )


# ============================================================
# CONFIDENCE
# ============================================================

confidence = float(
    probabilities[
        0,
        predicted_index
    ]
)


print()
print(
    "Model confidence:"
)

print(
    f"{confidence * 100:.2f}%"
)


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("INFERENCE TEST: PASSED")
print("=" * 70)

print()
print("The trained ResNet50 successfully:")

print("1. Loaded the model weights")
print("2. Loaded the real image")
print("3. Converted the image to RGB")
print("4. Resized the image to 224 x 224")
print("5. Applied ImageNet normalization")
print("6. Ran inference")
print("7. Produced four-class probabilities")

print()
print("Prediction:")
print(predicted_class)

print()
print("=" * 70)
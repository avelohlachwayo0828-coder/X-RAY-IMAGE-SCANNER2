import json

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "resnet50_final.pth"
)

DEPLOYMENT_CONFIG_PATH = (
    PROJECT_DIR
    / "deployment_config.json"
)


# ============================================================
# CLASSES
# ============================================================
# Read from deployment_config.json (written by the training
# notebook as sorted(train_df['label'].unique())) instead of
# hardcoding, so this can never silently drift out of sync
# with what the model was actually trained on.

with open(DEPLOYMENT_CONFIG_PATH) as f:

    _deployment_config = json.load(f)


CLASS_NAMES = _deployment_config["classes"]


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# BORDER CROP FRACTION
# ============================================================
# Must match margin_frac used in training exactly.

CROP_MARGIN_FRAC = _deployment_config.get(
    "crop_margin_frac",
    0.05
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# BORDER CROP
# ============================================================
# Exact same crop applied during training/evaluation. This
# removes a fixed-fraction margin from each edge before the
# image is resized, so the model sees the same framing in
# production that it saw during training.

def crop_margin(image, margin_frac=CROP_MARGIN_FRAC):

    width, height = image.size

    left = int(width * margin_frac)
    top = int(height * margin_frac)
    right = int(width * (1 - margin_frac))
    bottom = int(height * (1 - margin_frac))

    return image.crop(
        (left, top, right, bottom)
    )


# ============================================================
# EXACT TRAINING / EVALUATION PREPROCESSING
# ============================================================
# NOTE: training resized directly to (IMAGE_SIZE, IMAGE_SIZE)
# after the border crop -- it does NOT preserve aspect ratio
# and pad. Do not reintroduce aspect-preserving padding here;
# it would feed the model a differently-shaped input than the
# one it was trained and evaluated on.

transform = transforms.Compose([

    transforms.Lambda(
        lambda image:
        image.convert("RGB")
    ),

    transforms.Lambda(
        lambda image:
        crop_margin(image)
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])


# ============================================================
# CREATE RESNET50
# ============================================================

def create_model():

    model = models.resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_trained_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )


    model = create_model()


    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )


    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

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
                "Could not find model weights "
                "in checkpoint."
            )

    else:

        raise RuntimeError(
            "Unsupported model checkpoint format."
        )


    # --------------------------------------------------------
    # Remove DataParallel prefix if necessary
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )
    )


    if missing_keys:

        raise RuntimeError(
            "Missing model keys:\n"
            + "\n".join(
                missing_keys
            )
        )


    if unexpected_keys:

        raise RuntimeError(
            "Unexpected model keys:\n"
            + "\n".join(
                unexpected_keys
            )
        )


    model = model.to(
        DEVICE
    )

    model.eval()


    return model


# ============================================================
# LOAD ONCE
# ============================================================

model = load_trained_model()


# ============================================================
# PREDICTION
# ============================================================

def predict_tb(image):

    """
    Run the trained four-class ResNet50.

    Returns:

        diagnosis
        predicted_class
        confidence
        probabilities
        covid19_probability
        normal_probability
        pneumonia_probability
        tb_probability
    """


    # --------------------------------------------------------
    # Validate PIL image
    # --------------------------------------------------------

    if not isinstance(
        image,
        Image.Image
    ):

        raise TypeError(
            "Input must be a PIL Image."
        )


    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Apply exact preprocessing
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )


    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]


    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_index = int(
        torch.argmax(
            probabilities
        ).item()
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )


    # --------------------------------------------------------
    # Individual probabilities
    # --------------------------------------------------------
    # Indexed by position in CLASS_NAMES (from
    # deployment_config.json) rather than assumed fixed slots,
    # so this stays correct even if the class order in the
    # config ever changes.

    class_probabilities = {

        class_name: round(
            float(probabilities[i].item()) * 100,
            2
        )

        for i, class_name in enumerate(CLASS_NAMES)

    }


    # --------------------------------------------------------
    # Human-readable diagnosis
    # --------------------------------------------------------

    if predicted_class == "TB":

        diagnosis = (
            "Tuberculosis (TB) "
            "prediction"
        )

    elif predicted_class == "COVID19":

        diagnosis = (
            "COVID-19 prediction"
        )

    elif predicted_class == "PNEUMONIA":

        diagnosis = (
            "Pneumonia prediction"
        )

    else:

        diagnosis = (
            "Normal chest X-ray prediction"
        )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "diagnosis": diagnosis,

        "predicted_class":
            predicted_class,

        "predicted_class_index":
            predicted_index,

        "confidence": round(
            confidence * 100,
            2
        ),

        "probabilities": class_probabilities,

        "covid19_probability": class_probabilities.get(
            "COVID19"
        ),

        "normal_probability": class_probabilities.get(
            "NORMAL"
        ),

        "pneumonia_probability": class_probabilities.get(
            "PNEUMONIA"
        ),

        "tb_probability": class_probabilities.get(
            "TB"
        )
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info():

    return {

        "model":
            "ResNet50",

        "model_path":
            str(MODEL_PATH),

        "device":
            str(DEVICE),

        "classes":
            CLASS_NAMES,

        "image_size":
            IMAGE_SIZE,

        "crop_margin_frac":
            CROP_MARGIN_FRAC,

        "preprocessing":
            "RGB -> crop_margin(0.05) -> Resize(224x224) -> "
            "ToTensor -> ImageNet normalization"

    }
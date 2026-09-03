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


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TB",
]


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

class ResizeWithPad:

    def __init__(self, size):

        self.size = size


    def __call__(self, image):

        width, height = image.size

        scale = min(
            self.size / width,
            self.size / height
        )

        new_width = max(
            1,
            round(width * scale)
        )

        new_height = max(
            1,
            round(height * scale)
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.BILINEAR
        )

        padded = Image.new(
            "RGB",
            (
                self.size,
                self.size
            ),
            (0, 0, 0)
        )

        left = (
            self.size - new_width
        ) // 2

        top = (
            self.size - new_height
        ) // 2

        padded.paste(
            image,
            (
                left,
                top
            )
        )

        return padded


# ============================================================
# EXACT TRAINING / EVALUATION PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Lambda(
        lambda image:
        image.convert("RGB")
    ),

    ResizeWithPad(
        IMAGE_SIZE
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
        4
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

    covid_probability = float(
        probabilities[0].item()
    )

    normal_probability = float(
        probabilities[1].item()
    )

    pneumonia_probability = float(
        probabilities[2].item()
    )

    tb_probability = float(
        probabilities[3].item()
    )


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

        "probabilities": {

            "COVID19": round(
                covid_probability * 100,
                2
            ),

            "NORMAL": round(
                normal_probability * 100,
                2
            ),

            "PNEUMONIA": round(
                pneumonia_probability * 100,
                2
            ),

            "TB": round(
                tb_probability * 100,
                2
            )
        },

        "covid19_probability": round(
            covid_probability * 100,
            2
        ),

        "normal_probability": round(
            normal_probability * 100,
            2
        ),

        "pneumonia_probability": round(
            pneumonia_probability * 100,
            2
        ),

        "tb_probability": round(
            tb_probability * 100,
            2
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

        "preprocessing":
            "RGB -> ResizeWithPad(224) -> "
            "ToTensor -> ImageNet normalization"

    }
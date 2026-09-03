from pathlib import Path

import numpy as np
import cv2

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "resnet50_final.pth"
)


CLASS_NAMES = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TB",
]


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
# IMAGE NORMALIZATION
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
# RESIZE + PAD
#
# THIS MATCHES THE PROJECT DATA PIPELINE
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
# EXACT EVALUATION TRANSFORM
# ============================================================

evaluation_transform = transforms.Compose([

    transforms.Lambda(
        lambda image:
        image.convert("RGB")
    ),

    ResizeWithPad(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

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
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model):

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"ResNet50 model not found:\n"
            f"{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------------
    # Determine checkpoint format
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
                "Unable to find model state_dict "
                "inside checkpoint."
            )

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )


    # --------------------------------------------------------
    # Remove DataParallel prefix if present
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
    # Load
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


    return model


# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = create_model()

model = load_checkpoint(
    model
)

model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    image,
    target_class=None
):

    """
    Generate a Grad-CAM heatmap for the
    trained PyTorch ResNet50.

    The final convolutional block is used:

        model.layer4[-1]

    Gradients are calculated with respect to
    the selected class.
    """


    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    if not isinstance(
        image,
        Image.Image
    ):

        raise TypeError(
            "image must be a PIL.Image.Image"
        )


    image = image.convert(
        "RGB"
    )


    input_tensor = (
        evaluation_transform(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Storage for activations and gradients
    # --------------------------------------------------------

    activations = []

    gradients = []


    # --------------------------------------------------------
    # Hooks
    # --------------------------------------------------------

    target_layer = (
        model.layer4[-1]
    )


    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(
            output
        )


    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0]
        )


    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )


    try:

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )


        output = model(
            input_tensor
        )


        # ----------------------------------------------------
        # Select target class
        # ----------------------------------------------------

        if target_class is None:

            target_class = int(
                torch.argmax(
                    output,
                    dim=1
                ).item()
            )


        if not (
            0
            <= target_class
            < len(CLASS_NAMES)
        ):

            raise ValueError(
                f"Invalid target class: "
                f"{target_class}"
            )


        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        score = output[
            0,
            target_class
        ]


        score.backward()


        # ----------------------------------------------------
        # Get activation + gradients
        # ----------------------------------------------------

        if not activations:

            raise RuntimeError(
                "Grad-CAM activation hook "
                "did not capture output."
            )


        if not gradients:

            raise RuntimeError(
                "Grad-CAM gradient hook "
                "did not capture gradients."
            )


        activation = (
            activations[0]
        )


        gradient = (
            gradients[0]
        )


        # ----------------------------------------------------
        # Move to CPU
        # ----------------------------------------------------

        activation = (
            activation
            .detach()
            .cpu()
        )


        gradient = (
            gradient
            .detach()
            .cpu()
        )


        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )


        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = (
            weights * activation
        ).sum(
            dim=1
        )


        cam = torch.relu(
            cam
        )


        cam = cam[
            0
        ].numpy()


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam_min = cam.min()

        cam_max = cam.max()


        if (
            cam_max - cam_min
        ) > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam
            )


        # ----------------------------------------------------
        # Resize to 224 x 224
        # ----------------------------------------------------

        cam = cv2.resize(
            cam,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # Final safety normalization
        # ----------------------------------------------------

        cam = np.clip(
            cam,
            0.0,
            1.0
        )


        return cam, target_class


    finally:

        forward_handle.remove()

        backward_handle.remove()

        model.zero_grad(
            set_to_none=True
        )


# ============================================================
# GENERATE OVERLAY
# ============================================================

def generate_gradcam(
    image,
    target_class=None
):

    """
    Generate a JPEG-ready RGB Grad-CAM overlay.

    Returns:
        numpy.ndarray
        shape = (224, 224, 3)
        dtype = uint8
    """


    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original = image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    heatmap, predicted_class = (
        make_gradcam_heatmap(
            original,
            target_class
        )
    )


    # --------------------------------------------------------
    # Prepare original image using
    # the SAME resize/padding geometry
    # --------------------------------------------------------

    processed_image = (
        evaluation_transform.transforms[1](
            original
        )
    )


    original_array = np.asarray(
        processed_image
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Convert heatmap to uint8
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )


    # --------------------------------------------------------
    # OpenCV colour map
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )


    # --------------------------------------------------------
    # OpenCV uses BGR.
    # PIL image is RGB.
    # --------------------------------------------------------

    original_bgr = cv2.cvtColor(
        original_array,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay_bgr = cv2.addWeighted(
        original_bgr,
        0.60,
        heatmap_color,
        0.40,
        0
    )


    # --------------------------------------------------------
    # Convert back to RGB
    # --------------------------------------------------------

    overlay_rgb = cv2.cvtColor(
        overlay_bgr,
        cv2.COLOR_BGR2RGB
    )


    return overlay_rgb


# ============================================================
# OPTIONAL INFORMATION FUNCTION
# ============================================================

def get_gradcam_model_info():

    return {
        "model": "ResNet50",
        "model_path": str(
            MODEL_PATH
        ),
        "device": str(
            DEVICE
        ),
        "target_layer":
            "layer4[-1]",
        "image_size":
            IMAGE_SIZE,
        "classes":
            CLASS_NAMES,
        "preprocessing":
            [
                "RGB conversion",
                "ResizeWithPad(224)",
                "ImageNet normalization",
            ],
    }
"""
Shared helpers for NeuroSeg.

Everything that must be identical between training (train.py) and inference
(app.py) lives here: image size, ImageNet normalisation, the model definition,
the brain-pixel rule and the default file locations. Keeping it in one place
guarantees the API feeds the network exactly what it saw during training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Paths (all relative to this file, so scripts work from any working directory)
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
WEIGHTS_DIR = BACKEND_DIR / "weights"
OUTPUTS_DIR = BACKEND_DIR / "outputs"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
SAMPLES_DIR = PROJECT_DIR / "frontend" / "public" / "samples"

# --------------------------------------------------------------------------- #
# Model / preprocessing constants
# --------------------------------------------------------------------------- #
IMG_SIZE = 256
ENCODER_NAME = "resnet34"

# ImageNet statistics used by the pretrained ResNet34 encoder (RGB order).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# A pixel counts as "brain" when its brightest channel exceeds this value
# (0-255 scale). The LGG slices have a pure black background, so a small
# epsilon is enough and also tolerates a bit of JPEG noise in user uploads.
BRAIN_INTENSITY_EPS = 10

DEFAULT_THRESHOLD = 0.5


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalise an RGB uint8 image (H, W, 3) with ImageNet mean/std.

    Returns a float32 array of the same shape. Used by both the training
    dataset and the API so preprocessing can never drift apart.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image of shape (H, W, 3), got {image.shape}")
    scaled = image.astype(np.float32) / 255.0
    return (scaled - IMAGENET_MEAN) / IMAGENET_STD


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Normalise an RGB uint8 image and convert it to a (3, H, W) float tensor."""
    normalized = normalize_image(image)
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def count_brain_pixels(image: np.ndarray, eps: int = BRAIN_INTENSITY_EPS) -> int:
    """Number of non-background pixels in an RGB uint8 image."""
    return int((image.max(axis=2) > eps).sum())


def build_model(pretrained: bool = True) -> torch.nn.Module:
    """Create the U-Net (ResNet34 encoder, 3 input channels, 1 output logit map).

    pretrained=True downloads ImageNet encoder weights (training only).
    The API calls this with pretrained=False and then loads best.pt.
    """
    # Imported lazily so `import common` stays cheap for tooling.
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
    )


def get_device() -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device name for logs."""
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"

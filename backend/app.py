"""
NeuroSeg inference API (FastAPI).

Run from the backend folder:
    uvicorn app:app --port 8000

Endpoints
---------
GET  /health    -> service status and whether the model is loaded
POST /predict   -> multipart upload (field "file"); returns the probability map
                   as a grayscale PNG so the browser can threshold it live.
"""

from __future__ import annotations

import base64
import io
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from common import (
    DEFAULT_THRESHOLD,
    IMG_SIZE,
    WEIGHTS_PATH,
    build_model,
    count_brain_pixels,
    describe_device,
    get_device,
    image_to_tensor,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB is far more than any MRI slice needs
MISSING_WEIGHTS_MESSAGE = (
    f"Model weights not found at {WEIGHTS_PATH}. "
    "Train the model first: open a terminal in the backend folder and run "
    "'python train.py' (or 'python train.py --max-images 300 --epochs 1' for a quick test), "
    "then restart the API."
)

# Populated once at startup by the lifespan handler.
state: dict = {"model": None, "device": None, "error": None, "checkpoint": {}}


def load_model() -> None:
    """Load weights/best.pt once. If it is missing, remember a helpful error instead of crashing."""
    device = get_device()
    state["device"] = device
    if not WEIGHTS_PATH.exists():
        state["error"] = MISSING_WEIGHTS_MESSAGE
        print(f"[NeuroSeg] WARNING: {MISSING_WEIGHTS_MESSAGE}")
        return
    try:
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        model = build_model(pretrained=False)  # architecture only; weights come from best.pt
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
    except Exception as exc:  # corrupt or incompatible checkpoint
        state["error"] = f"Could not load {WEIGHTS_PATH}: {exc}. Re-run train.py to regenerate it."
        print(f"[NeuroSeg] ERROR: {state['error']}")
        return
    state["model"] = model
    state["error"] = None
    state["checkpoint"] = {k: checkpoint.get(k) for k in ("epoch", "val_dice", "pretrained")}
    print(f"[NeuroSeg] Model loaded from {WEIGHTS_PATH} on {describe_device(device)} "
          f"(epoch {checkpoint.get('epoch')}, val Dice {checkpoint.get('val_dice', 0):.4f})")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="NeuroSeg API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def png_base64(array: np.ndarray) -> str:
    """Encode a uint8 array (H, W) or (H, W, 3) as a base64 PNG string (no data: prefix)."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_upload(data: bytes) -> np.ndarray:
    """Decode an uploaded image to an RGB uint8 array resized to IMG_SIZE x IMG_SIZE."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                # 16-bit / float images: rescale to 0-255 before converting.
                arr = np.asarray(img, dtype=np.float32)
                lo, hi = float(arr.min()), float(arr.max())
                arr = (arr - lo) / (hi - lo + 1e-8) * 255.0
                img = Image.fromarray(arr.astype(np.uint8))
            rgb = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            return np.array(rgb)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Unsupported or corrupted image. Please upload a PNG, JPG or TIF MRI slice.",
        ) from exc


@app.get("/health")
def health() -> dict:
    device = state["device"]
    return {
        "status": "ok" if state["model"] is not None else "model_missing",
        "model_loaded": state["model"] is not None,
        "device": describe_device(device) if device is not None else None,
        "weights_path": str(WEIGHTS_PATH),
        "checkpoint": state["checkpoint"],
        "detail": state["error"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail=state["error"] or MISSING_WEIGHTS_MESSAGE)

    if file.content_type and not (
        file.content_type.startswith("image/") or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400,
            detail=f"'{file.filename}' is not an image ({file.content_type}). Upload a PNG, JPG or TIF.",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")

    image = decode_upload(data)
    started = time.perf_counter()
    x = image_to_tensor(image)[None].to(state["device"])  # same normalisation as training
    with torch.inference_mode():
        prob = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
    elapsed_ms = (time.perf_counter() - started) * 1000

    # 8-bit quantisation (1/255 steps) is far finer than the 0.05 slider steps.
    prob_u8 = np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)
    return {
        "prob_map": png_base64(prob_u8),
        "image": png_base64(image),
        "brain_pixels": count_brain_pixels(image),
        "default_threshold": DEFAULT_THRESHOLD,
        "width": IMG_SIZE,
        "height": IMG_SIZE,
        "inference_ms": round(elapsed_ms, 1),
        "filename": file.filename,
    }

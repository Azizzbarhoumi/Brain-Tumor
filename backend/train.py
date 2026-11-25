"""
NeuroSeg training script.

Trains a U-Net (ResNet34 encoder) to segment low-grade gliomas on the
"LGG MRI Segmentation" dataset (Kaggle: mateuszbuda/lgg-mri-segmentation).

Examples
--------
    python train.py --max-images 300 --epochs 1     # quick smoke test
    python train.py                                 # full training (10 epochs)
    python train.py --data D:/datasets/lgg          # use an already-downloaded copy
    python train.py --no-pretrained                 # random-init encoder for comparison

Outputs
-------
    weights/best.pt              best checkpoint by validation Dice
    outputs/curves.png           loss and Dice curves
    outputs/predictions.png      6 test tumour slices: MRI / ground truth / prediction
    outputs/metrics.json         test Dice / IoU, split and training history
    ../frontend/public/samples/  3 tumour + 1 no-tumour test slices for the web demo

With --no-pretrained the checkpoint and outputs go to weights/best_no_pretrained.pt
and outputs/no_pretrained/ so the demo model is not overwritten.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import (
    IMG_SIZE,
    OUTPUTS_DIR,
    SAMPLES_DIR,
    WEIGHTS_DIR,
    WEIGHTS_PATH,
    build_model,
    count_brain_pixels,
    describe_device,
    get_device,
    image_to_tensor,
)

DATASET_ID = "mateuszbuda/lgg-mri-segmentation"
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)  # train / val / test, by patient
EVAL_THRESHOLD = 0.5


# =========================================================================== #
# Command line
# =========================================================================== #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a U-Net (ResNet34) for brain tumour segmentation on LGG MRI slices."
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to an already-downloaded dataset folder (the folder that "
                             "contains kaggle_3m, or kaggle_3m itself). If omitted, the dataset "
                             "is downloaded with kagglehub.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default 10).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default 16).")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (default 3e-4).")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Use at most this many slices in total (split 80/10/10) for a quick test run.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Random-init encoder (encoder_weights=None) for a pretrained vs. scratch comparison.")
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader worker processes (default: 2 on Windows, 4 elsewhere). "
                             "Use 0 if you hit multiprocessing errors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split and training (default 42).")
    return parser.parse_args()


# =========================================================================== #
# Dataset discovery and download
# =========================================================================== #
def kaggle_credentials_present() -> bool:
    """True if kagglehub is likely to find Kaggle credentials."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "kaggle.json").exists() or (config_dir / "access_token").exists()


def print_kaggle_help(error: Exception | None = None) -> None:
    """Explain how to set up Kaggle credentials or pass --data."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    lines = [
        "",
        "=" * 72,
        "Could not download the dataset from Kaggle.",
    ]
    if error is not None:
        lines.append(f"Reason: {type(error).__name__}: {error}")
    lines += [
        "",
        "Option A - set up Kaggle credentials (one time):",
        "  1. Sign in at https://www.kaggle.com and open Settings -> API -> 'Create New Token'.",
        "     This downloads a file called kaggle.json.",
        f"  2. Move it to: {kaggle_json}",
        "     (or set the environment variables KAGGLE_USERNAME and KAGGLE_KEY).",
        "  3. Run this script again.",
        "",
        "Option B - download manually:",
        f"  1. Download https://www.kaggle.com/datasets/{DATASET_ID} and unzip it.",
        "  2. Run:  python train.py --data <path-to-the-unzipped-folder>",
        "=" * 72,
        "",
    ]
    print("\n".join(lines), file=sys.stderr)


def download_dataset() -> Path:
    """Download (or reuse the cached copy of) the dataset with kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run:  pip install kagglehub", file=sys.stderr)
        print_kaggle_help()
        sys.exit(1)

    if not kaggle_credentials_present():
        print("Note: no Kaggle credentials found; trying an anonymous download of the public dataset...")
    print(f"Downloading '{DATASET_ID}' with kagglehub (cached after the first run)...")
    try:
        path = kagglehub.dataset_download(DATASET_ID)
    except Exception as exc:  # network, auth or API errors all end up here
        print_kaggle_help(exc)
        sys.exit(1)
    print(f"Dataset available at: {path}")
    return Path(path)


def find_kaggle_3m(root: Path) -> Path:
    """Locate the folder whose sub-folders hold <patient>_<n>.tif / _mask.tif pairs."""
    root = root.expanduser().resolve()
    if not root.exists():
        sys.exit(f"Data folder not found: {root}")

    def has_slices(folder: Path) -> bool:
        return folder.is_dir() and next(folder.glob("*/*_mask.tif"), None) is not None

    if has_slices(root):
        return root
    # The Kaggle archive contains kaggle_3m (and sometimes a nested duplicate);
    # prefer the shallowest match.
    candidates = sorted((p for p in root.rglob("kaggle_3m") if p.is_dir()), key=lambda p: len(p.parts))
    for candidate in candidates:
        if has_slices(candidate):
            return candidate
    sys.exit(
        f"Could not find the kaggle_3m folder with *_mask.tif files under {root}.\n"
        "Pass --data pointing at the unzipped dataset (the folder containing kaggle_3m)."
    )


# =========================================================================== #
# Slices, split and sub-sampling
# =========================================================================== #
@dataclass(frozen=True)
class Slice:
    patient: str
    image_path: Path
    mask_path: Path
    tumor_pixels: int

    @property
    def has_tumor(self) -> bool:
        return self.tumor_pixels > 0


def slice_number(path: Path) -> int:
    """Return n from <patient>_<n>.tif (used for natural sorting)."""
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def read_image(path: Path) -> np.ndarray:
    """Read an MRI slice as RGB uint8 (H, W, 3), resized to IMG_SIZE if needed."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(img)


def read_mask(path: Path) -> np.ndarray:
    """Read a mask as binary uint8 (H, W) with values {0, 1}."""
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        return (np.array(img) > 127).astype(np.uint8)


def collect_slices(data_root: Path) -> dict[str, list[Slice]]:
    """Scan kaggle_3m/<patient>/ and return {patient: [Slice, ...]} sorted by slice number."""
    patients: dict[str, list[Slice]] = {}
    patient_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    print(f"Scanning {len(patient_dirs)} patient folders (reading masks)...")
    for patient_dir in patient_dirs:
        slices = []
        for mask_path in patient_dir.glob("*_mask.tif"):
            image_path = mask_path.with_name(mask_path.name[: -len("_mask.tif")] + ".tif")
            if not image_path.exists():
                continue
            tumor_pixels = int(read_mask(mask_path).sum())
            slices.append(Slice(patient_dir.name, image_path, mask_path, tumor_pixels))
        if slices:
            slices.sort(key=lambda s: slice_number(s.image_path))
            patients[patient_dir.name] = slices
    if not patients:
        sys.exit(f"No image/mask pairs found in {data_root}")
    return patients


def split_patients(patients: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    """Shuffle patients with a fixed seed and split 80/10/10 so no patient is shared."""
    ordered = sorted(patients)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    n_val = max(1, round(n * SPLIT_FRACTIONS[1])) if n >= 3 else 0
    n_test = max(1, round(n * SPLIT_FRACTIONS[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    return ordered[:n_train], ordered[n_train:n_train + n_val], ordered[n_train + n_val:]


def subsample(slices: list[Slice], limit: int, rng: random.Random) -> list[Slice]:
    """Pick up to `limit` slices, balancing tumour / no-tumour so tiny runs still learn something."""
    if limit >= len(slices):
        return list(slices)
    tumor = [s for s in slices if s.has_tumor]
    healthy = [s for s in slices if not s.has_tumor]
    rng.shuffle(tumor)
    rng.shuffle(healthy)
    n_tumor = min(len(tumor), limit // 2)
    n_healthy = min(len(healthy), limit - n_tumor)
    n_tumor = min(len(tumor), limit - n_healthy)
    return tumor[:n_tumor] + healthy[:n_healthy]


# =========================================================================== #
# PyTorch dataset and augmentation
# =========================================================================== #
def build_train_augmentations():
    """Flips, rotate90, shift/scale/rotate and brightness/contrast (albumentations)."""
    import albumentations as A

    with warnings.catch_warnings():
        # albumentations 2.x marks ShiftScaleRotate as deprecated but still provides it.
        warnings.simplefilter("ignore")
        if hasattr(A, "ShiftScaleRotate"):
            shift_scale_rotate = A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5
            )
        else:  # future albumentations without ShiftScaleRotate: equivalent Affine
            shift_scale_rotate = A.Affine(
                translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-15, 15), border_mode=0, p=0.5
            )
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        shift_scale_rotate,
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])


class SliceDataset(Dataset):
    """Loads (image, mask) pairs from disk; normalisation comes from common.py."""

    def __init__(self, slices: list[Slice], augment=None):
        self.slices = slices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int):
        item = self.slices[index]
        image = read_image(item.image_path)
        mask = read_mask(item.mask_path)
        if self.augment is not None:
            out = self.augment(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        x = image_to_tensor(image)
        y = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None]
        return x, y


def make_loader(slices, augment, batch_size, shuffle, workers, device) -> DataLoader:
    return DataLoader(
        SliceDataset(slices, augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


# =========================================================================== #
# Loss and metrics
# =========================================================================== #
class BceDiceLoss(nn.Module):
    """BCE-with-logits + soft Dice loss."""

    def __init__(self):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.dice(logits, target)


def autocast_context(use_amp: bool):
    """Mixed precision on CUDA only; a no-op elsewhere."""
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def slice_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice for one slice; an empty prediction on an empty mask counts as 1."""
    inter = float((pred * target).sum())
    total = float(pred.sum() + target.sum())
    return 1.0 if total == 0 else 2.0 * inter / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp) -> dict:
    """Loss, dataset-level Dice/IoU and mean per-slice Dice at threshold 0.5."""
    model.eval()
    total_loss, n_items = 0.0, 0
    inter_sum = pred_sum = true_sum = 0.0
    per_slice, per_tumor_slice = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(use_amp):
            logits = model(x)
        logits = logits.float()
        total_loss += loss_fn(logits, y).item() * x.size(0)
        n_items += x.size(0)

        pred = (torch.sigmoid(logits) > EVAL_THRESHOLD).float()
        inter = (pred * y).sum(dim=(1, 2, 3))
        p_sum = pred.sum(dim=(1, 2, 3))
        t_sum = y.sum(dim=(1, 2, 3))
        inter_sum += inter.sum().item()
        pred_sum += p_sum.sum().item()
        true_sum += t_sum.sum().item()
        for i, p, t in zip(inter.tolist(), p_sum.tolist(), t_sum.tolist()):
            d = 1.0 if p + t == 0 else 2.0 * i / (p + t)
            per_slice.append(d)
            if t > 0:
                per_tumor_slice.append(d)

    denom = pred_sum + true_sum
    union = denom - inter_sum
    return {
        "loss": total_loss / max(n_items, 1),
        "dice": 1.0 if denom == 0 else 2.0 * inter_sum / denom,
        "iou": 1.0 if union == 0 else inter_sum / union,
        "dice_per_slice_mean": float(np.mean(per_slice)) if per_slice else float("nan"),
        "dice_tumor_slices_mean": float(np.mean(per_tumor_slice)) if per_tumor_slice else float("nan"),
        "num_slices": n_items,
    }


# =========================================================================== #
# Training
# =========================================================================== #
def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, use_amp,
                    epoch, epochs, estimate_epoch_time, val_batches) -> float:
    """One pass over the training set. On the first epoch, prints a time estimate."""
    model.train()
    total_loss, n_items = 0.0, 0
    n_batches = len(loader)
    epoch_start = time.perf_counter()
    timer_start = None
    estimate_after = min(10, n_batches)  # time a handful of batches, then extrapolate

    for batch_idx, (x, y) in enumerate(loader, start=1):
        if batch_idx == 2:
            # Skip batch 1 in the estimate: it includes worker start-up and CUDA warm-up.
            timer_start = time.perf_counter()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        n_items += x.size(0)

        if estimate_epoch_time and batch_idx == estimate_after:
            if timer_start is not None and batch_idx > 1:
                per_batch = (time.perf_counter() - timer_start) / (batch_idx - 1)
            else:
                per_batch = time.perf_counter() - epoch_start
            # Validation is forward-only, roughly a third of a training step.
            estimate = per_batch * n_batches + per_batch * val_batches / 3
            print(f"\n  Estimated time per epoch: ~{format_seconds(estimate)} "
                  f"(~{format_seconds(estimate * epochs)} for {epochs} epoch(s))")

        print(f"\r  Epoch {epoch}/{epochs}  batch {batch_idx}/{n_batches}  "
              f"loss {total_loss / n_items:.4f}", end="", flush=True)
    print()
    return total_loss / max(n_items, 1)


# =========================================================================== #
# Reporting: plots, predictions, samples
# =========================================================================== #
def save_curves(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "o-", label="validation")
    axes[0].set_title("Loss (BCE + Dice)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_dice"] for h in history], "o-", label="val Dice")
    axes[1].plot(epochs, [h["val_iou"] for h in history], "o-", label="val IoU")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def red_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a red mask over an RGB image (for predictions.png)."""
    out = image.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array([255, 0, 0], dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


@torch.no_grad()
def predict_mask(model, image: np.ndarray, device) -> np.ndarray:
    model.eval()
    x = image_to_tensor(image)[None].to(device)
    prob = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
    return (prob > EVAL_THRESHOLD).astype(np.uint8)


def pick_tumor_slices(slices: list[Slice], count: int) -> list[Slice]:
    """Largest-tumour slice of each patient first (variety), then other large ones."""
    tumor = [s for s in slices if s.has_tumor]
    best_per_patient: dict[str, Slice] = {}
    for s in tumor:
        if s.patient not in best_per_patient or s.tumor_pixels > best_per_patient[s.patient].tumor_pixels:
            best_per_patient[s.patient] = s
    chosen = sorted(best_per_patient.values(), key=lambda s: -s.tumor_pixels)[:count]
    if len(chosen) < count:
        rest = sorted((s for s in tumor if s not in chosen), key=lambda s: -s.tumor_pixels)
        chosen += rest[: count - len(chosen)]
    return chosen


def save_predictions(model, slices: list[Slice], device, path: Path, count: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = pick_tumor_slices(slices, count)
    if not chosen:
        print("  No tumour slices in the test set; skipping predictions.png")
        return
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)), squeeze=False)
    for row, item in enumerate(chosen):
        image = read_image(item.image_path)
        truth = read_mask(item.mask_path)
        pred = predict_mask(model, image, device)
        dice = slice_dice(pred, truth)
        panels = [
            (image, f"MRI  {item.image_path.stem}"),
            (red_overlay(image, truth), "Ground truth"),
            (red_overlay(image, pred), f"Prediction  Dice {dice:.3f}"),
        ]
        for col, (panel, title) in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(panel)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def export_samples(test_slices: list[Slice], samples_dir: Path) -> None:
    """Write 3 tumour slices and 1 no-tumour slice from TEST patients as PNG for the frontend."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    tumor = pick_tumor_slices(test_slices, 3)
    healthy = [s for s in test_slices if not s.has_tumor]
    exported = []
    for i, item in enumerate(tumor, start=1):
        out = samples_dir / f"sample_tumor_{i}.png"
        Image.fromarray(read_image(item.image_path)).save(out)
        exported.append(out.name)
    if healthy:
        # The healthy slice with the most brain tissue is the most convincing example.
        best = max(healthy, key=lambda s: count_brain_pixels(read_image(s.image_path)))
        out = samples_dir / "sample_no_tumor.png"
        Image.fromarray(read_image(best.image_path)).save(out)
        exported.append(out.name)
    print(f"Exported {len(exported)} sample images to {samples_dir}: {', '.join(exported)}")


# =========================================================================== #
# Main
# =========================================================================== #
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    workers = args.workers if args.workers is not None else (2 if os.name == "nt" else 4)
    workers = max(0, min(workers, os.cpu_count() or 1))
    pretrained = not args.no_pretrained

    weights_path = WEIGHTS_PATH if pretrained else WEIGHTS_DIR / "best_no_pretrained.pt"
    outputs_dir = OUTPUTS_DIR if pretrained else OUTPUTS_DIR / "no_pretrained"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {describe_device(device)}  |  mixed precision: {'on' if use_amp else 'off'}  "
          f"|  workers: {workers}")

    # ---- data -------------------------------------------------------------
    data_root = find_kaggle_3m(Path(args.data) if args.data else download_dataset())
    print(f"Using data from: {data_root}")
    patients = collect_slices(data_root)
    train_p, val_p, test_p = split_patients(list(patients), args.seed)
    splits = {
        "train": [s for p in train_p for s in patients[p]],
        "val": [s for p in val_p for s in patients[p]],
        "test": [s for p in test_p for s in patients[p]],
    }
    assert not (set(train_p) & set(val_p) or set(train_p) & set(test_p) or set(val_p) & set(test_p))

    # Samples always come from the full test split, so they don't depend on --max-images.
    export_samples(splits["test"], SAMPLES_DIR)

    if args.max_images:
        rng = random.Random(args.seed)
        quotas = [max(1, int(args.max_images * f)) for f in SPLIT_FRACTIONS]
        quotas[0] = max(1, args.max_images - quotas[1] - quotas[2])
        for (name, items), quota in zip(list(splits.items()), quotas):
            splits[name] = subsample(items, quota, rng)
        print(f"--max-images {args.max_images}: using a balanced subset of the patient-level split.")

    for name, ids in (("train", train_p), ("val", val_p), ("test", test_p)):
        items = splits[name]
        n_tumor = sum(s.has_tumor for s in items)
        print(f"  {name:5s}: {len(ids):3d} patients, {len(items):4d} slices ({n_tumor} with tumour)")
    if not splits["train"] or not splits["val"] or not splits["test"]:
        sys.exit("Not enough patients to build train/val/test splits.")

    train_loader = make_loader(splits["train"], build_train_augmentations(), args.batch_size, True, workers, device)
    val_loader = make_loader(splits["val"], None, args.batch_size, False, workers, device)
    test_loader = make_loader(splits["test"], None, args.batch_size, False, workers, device)

    # ---- model ------------------------------------------------------------
    print(f"Building U-Net / resnet34 (encoder weights: {'imagenet' if pretrained else 'random init'})")
    try:
        model = build_model(pretrained=pretrained).to(device)
    except Exception as exc:  # usually no internet access to fetch the ImageNet weights
        sys.exit(
            f"Could not create the model with ImageNet encoder weights ({type(exc).__name__}: {exc}).\n"
            "The pretrained ResNet34 weights are downloaded on first use; check your internet connection,\n"
            "or train from scratch with:  python train.py --no-pretrained"
        )
    loss_fn = BceDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- training loop ----------------------------------------------------
    history, best_dice, best_epoch = [], -1.0, 0
    run_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, use_amp,
            epoch, args.epochs, estimate_epoch_time=(epoch == 1), val_batches=len(val_loader),
        )
        val = evaluate(model, val_loader, loss_fn, device, use_amp)
        scheduler.step(val["dice"])
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_dice": val["dice"], "val_iou": val["iou"], "lr": lr, "seconds": elapsed})

        improved = val["dice"] > best_dice
        if improved:
            best_dice, best_epoch = val["dice"], epoch
            torch.save({
                "model_state": model.state_dict(),
                "encoder": "resnet34",
                "pretrained": pretrained,
                "img_size": IMG_SIZE,
                "epoch": epoch,
                "val_dice": val["dice"],
            }, weights_path)
        print(f"  epoch {epoch}: train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"val Dice {val['dice']:.4f} | val IoU {val['iou']:.4f} | lr {lr:.1e} | "
              f"{format_seconds(elapsed)}{'  <- best, saved' if improved else ''}")

    print(f"Training finished in {format_seconds(time.perf_counter() - run_start)}. "
          f"Best val Dice {best_dice:.4f} at epoch {best_epoch} -> {weights_path}")

    # ---- test evaluation with the best checkpoint --------------------------
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, test_loader, loss_fn, device, use_amp)
    print("\nTEST set (unseen patients):")
    print(f"  Dice {test['dice']:.4f}  |  IoU {test['iou']:.4f}  |  "
          f"mean per-slice Dice {test['dice_per_slice_mean']:.4f}  |  "
          f"mean Dice on tumour slices {test['dice_tumor_slices_mean']:.4f}")

    save_curves(history, outputs_dir / "curves.png")
    save_predictions(model, splits["test"], device, outputs_dir / "predictions.png")

    def clean(value):
        return None if isinstance(value, float) and math.isnan(value) else value

    metrics = {
        "test": {k: clean(v) for k, v in test.items()},
        "best_val_dice": best_dice,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "pretrained_encoder": pretrained,
        "max_images": args.max_images,
        "device": describe_device(device),
        "threshold": EVAL_THRESHOLD,
        "seed": args.seed,
        "split": {
            "train_patients": train_p, "val_patients": val_p, "test_patients": test_p,
            "train_slices": len(splits["train"]), "val_slices": len(splits["val"]),
            "test_slices": len(splits["test"]),
        },
        "history": history,
    }
    (outputs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved curves.png, predictions.png and metrics.json to {outputs_dir}")
    if pretrained:
        print("Next: start the API with  uvicorn app:app --port 8000")


if __name__ == "__main__":
    # The __main__ guard is required on Windows, where DataLoader workers re-import this file.
    main()

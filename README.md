# NeuroSeg: brain tumor segmentation demo

Upload a brain MRI slice and get the tumor highlighted in red, a *Tumor detected / Not detected* verdict, the tumor area as a percentage of the brain, a confidence heatmap, and a threshold slider that updates everything live in the browser.

> **Educational demo, not a medical device.** Do not use it for diagnosis.

```
Brain-Tumor/
├── backend/
│   ├── common.py          shared preprocessing (ImageNet normalisation), model, paths
│   ├── train.py           downloads the data, trains the U-Net, evaluates, exports samples
│   ├── app.py             FastAPI inference server (/predict, /health)
│   ├── requirements.txt
│   ├── weights/           best.pt is written here by train.py
│   └── outputs/           curves.png, predictions.png, metrics.json
└── frontend/              Vite + React + Tailwind app
    └── public/samples/    sample slices exported by train.py
```

- **Model:** `segmentation_models_pytorch.Unet`, ResNet34 encoder (ImageNet weights), 3 input channels, 1 output.
- **Data:** [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (about 110 patients, 3,900 slices of 256×256), split **by patient** 80/10/10 with a fixed seed.
- **Training:** BCE + Dice loss, AdamW (lr 3e-4), ReduceLROnPlateau, 10 epochs, batch size 16, flips / rotate90 / shift-scale-rotate / brightness-contrast augmentation, mixed precision on CUDA.

## Requirements

- Python 3.10, 3.11 or 3.12
- Node.js 18 or newer (includes npm)
- About 2 GB of free disk space for the dataset and packages
- A Kaggle account for the automatic download, or a manually downloaded copy of the dataset (see step 2)

The commands below are for **Windows PowerShell**. macOS/Linux equivalents are shown where they differ.

---

## 1. Install the backend dependencies

Open PowerShell in the project folder:

```powershell
cd C:\Users\azizb\Downloads\Brain-Tumor\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell refuses to run `Activate.ps1`, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**NVIDIA GPU (recommended):** install the CUDA build of PyTorch first. Without this step pip installs the CPU-only build on Windows.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then install everything else (on a CPU-only machine or a Mac, this is the only install command you need):

```powershell
pip install -r requirements.txt
```

macOS/Linux: `source .venv/bin/activate` instead of `.\.venv\Scripts\Activate.ps1`.

## 2. Kaggle access (for the automatic download)

`train.py` downloads the dataset with `kagglehub` and caches it after the first run. If the download fails, the script prints these instructions:

1. On https://www.kaggle.com open **Settings → API → Create New Token**. This downloads `kaggle.json`.
2. Move it to `C:\Users\<you>\.kaggle\kaggle.json` (macOS/Linux: `~/.kaggle/kaggle.json`).

Alternatively, download and unzip the dataset yourself and pass its folder with `--data`:

```powershell
python train.py --data C:\path\to\lgg-mri-segmentation
```

`--data` accepts the folder that contains `kaggle_3m` or `kaggle_3m` itself.

## 3. Quick test run (a few minutes, even on CPU)

With the virtual environment active, from `backend/`:

```powershell
python train.py --max-images 300 --epochs 1
```

This trains on a 300-slice subset of the patient split, so the whole pipeline runs end to end quickly. It writes `weights/best.pt`, the files in `outputs/`, and the 4 sample slices in `frontend/public/samples/`. The model will be weak; it only checks that everything works.

## 4. Full training

```powershell
python train.py
```

This uses all patients, 10 epochs and batch size 16. The script prints the device it found (CUDA, Apple MPS or CPU) and an estimated time per epoch after the first few batches. A recent NVIDIA GPU needs roughly 1 minute per epoch; a laptop CPU can take 10 to 30 minutes per epoch.

At the end it reports **Dice and IoU on the test patients** and saves:

| File | Content |
| --- | --- |
| `weights/best.pt` | checkpoint with the best validation Dice |
| `outputs/curves.png` | training/validation loss and validation Dice/IoU per epoch |
| `outputs/predictions.png` | 6 test tumor slices: MRI / ground truth / prediction, with the Dice of each slice |
| `outputs/metrics.json` | test metrics, the patient split and the training history |
| `../frontend/public/samples/*.png` | 3 tumor slices and 1 tumor-free slice from the test patients |

Useful options:

```powershell
python train.py --epochs 20            # train longer
python train.py --batch-size 8         # if the GPU runs out of memory
python train.py --workers 0            # if Windows shows DataLoader/multiprocessing errors
python train.py --no-pretrained        # random-init encoder, for a pretrained vs. scratch comparison
```

`--no-pretrained` writes to `weights/best_no_pretrained.pt` and `outputs/no_pretrained/`, so it never overwrites the demo model. Compare `outputs/metrics.json` with `outputs/no_pretrained/metrics.json`.

## 5. Start the API

From `backend/` with the virtual environment active:

```powershell
uvicorn app:app --port 8000
```

- Health check: http://localhost:8000/health
- Interactive docs: http://localhost:8000/docs

The model is loaded once at startup. If `weights/best.pt` is missing, `/predict` answers with an error telling you to run `train.py` first; restart uvicorn after training.

## 6. Install and start the frontend

Open a **second** PowerShell window:

```powershell
cd C:\Users\azizb\Downloads\Brain-Tumor\frontend
npm install
npm run dev
```

Open http://localhost:5173. The header shows whether the API is online and whether the model is loaded. Drop an MRI slice on the upload zone, click to browse, or click one of the sample slices.

The API address defaults to `http://localhost:8000`. To change it, copy `.env.example` to `.env` and edit `VITE_API_URL`:

```powershell
Copy-Item .env.example .env
```

The frontend must run on port 5173, because the API only allows CORS from `http://localhost:5173`.

---

## How the demo works

1. The browser uploads the image to `POST /predict`.
2. The API resizes it to 256×256, normalises it with the ImageNet mean/std from `common.py` (the same function used in training), runs the U-Net and applies a sigmoid.
3. It returns the probability map as a grayscale PNG (`prob_map`), the resized MRI (`image`), the number of non-background pixels (`brain_pixels`) and `default_threshold` (0.5).
4. The browser draws both PNGs into canvases and does all pixel work itself:
   - **mask** = probability > threshold
   - **overlay**: semi-transparent red fill plus a thin outline
   - **heatmap**: inferno colormap blended 50% over the MRI, with a legend
   - **verdict**: *TUMOR DETECTED* when the mask has at least 30 pixels
   - **area** = mask pixels / brain pixels × 100, plus pixel count and mean confidence inside the mask

Because thresholding happens in the browser, the slider updates everything instantly without calling the API again.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Backend unreachable" in the app | Start the API (step 5) and check http://localhost:8000/health |
| "Model not trained yet" | Run `python train.py`, then restart uvicorn |
| Sample tiles say "not exported yet" | Run `train.py` once (even the quick test run exports them) |
| Kaggle download fails | Set up `kaggle.json` (step 2) or use `--data` |
| ImageNet weights fail to download | Check your internet connection, or use `--no-pretrained` |
| `CUDA out of memory` | `python train.py --batch-size 8` |
| Errors mentioning workers or multiprocessing on Windows | `python train.py --workers 0` |
| Training says `Device: CPU` on an NVIDIA machine | Reinstall PyTorch with the CUDA command in step 1 |

## Dataset credit

Buda, M., Saha, A., Mazurowski, M. A. *Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm.* Computers in Biology and Medicine, 2019. Images from The Cancer Imaging Archive (TCIA).

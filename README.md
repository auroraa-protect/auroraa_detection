# Auroraa Forensics — Training Repo

Image manipulation detection via ELA, EXIF, Noise & FFT analysis.  
This repo is the **standalone training environment** — no API, no routes, just the core forensics engine + a Jupyter notebook pipeline.

---

## Directory Layout

```
auroraa vault/
├── app/
│   └── image_forensics/
│       ├── core/
│       │   ├── ela.py          ← Error Level Analysis
│       │   ├── exif.py         ← EXIF metadata forensics
│       │   ├── fft.py          ← Frequency domain analysis
│       │   └── noise.py        ← Noise consistency map
│       ├── analyzer.py         ← Orchestrator (runs all 4 signals)
│       ├── report.py           ← Visual report generator
│       └── translate.py        ← Plain-language finding generator
├── notebooks/
│   └── training_pipeline.ipynb ← Main training notebook (START HERE)
├── samples/
│   ├── original/               ← AUTHENTIC images (auto-generated)
│   └── tampered/               ← MANIPULATED images (auto-generated)
├── models/                     ← Saved model output (.joblib)
├── reports/                    ← Charts and evaluation plots
├── audit.py                    ← Batch audit & weight recommender
├── run_audit_test.py           ← CLI evaluation runner
├── generate_training_samples.py← Synthetic sample generator
├── install.bat                 ← One-click GPU environment setup
└── requirements.txt
```

---

## Step 1 — Set Up the Environment

> **Requires:** Python 3.10+, an NVIDIA RTX GPU (RTX 2050 supported), CUDA 12.x drivers installed.

### Option A — One-click (recommended)

Double-click `install.bat` OR run it from PowerShell:

```powershell
.\install.bat
```

This will:
1. Install all Python dependencies from `requirements.txt`
2. Install **PyTorch with CUDA 12.1** (for RTX 2050 GPU acceleration)
3. Print a GPU verification check at the end

### Option B — Manual

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# 2. Install core + ML deps
pip install -r requirements.txt

# 3. Install PyTorch CUDA (RTX 2050 = CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Verify GPU is detected
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA: True | NVIDIA GeForce RTX 2050
```

---

## Step 2 — Generate Training Samples

The repo ships with a synthetic image generator so you can start immediately **without any real dataset**.

```powershell
venv\Scripts\python.exe generate_training_samples.py
```

This creates:
- `samples/original/authentic_00.jpg … authentic_19.jpg` — 20 authentic images
- `samples/tampered/tampered_00.jpg  … tampered_19.jpg`  — 20 tampered images

with 4 scene types × 4 manipulation types (splice, double-compress, copy-move, AI pattern).

> **Want more data?** Just increase `N = 20` at the top of `generate_training_samples.py` and re-run.

> **Have real images?** Drop your own `.jpg`/`.png` files into `samples/original/` and `samples/tampered/` — the notebook picks them up automatically.

---

## Step 3 — Launch the Notebook

```powershell
venv\Scripts\jupyter notebook notebooks\training_pipeline.ipynb
```

Your browser will open automatically. The notebook has **6 sections — run them top to bottom:**

| Cell | What it does |
|------|-------------|
| **0. Setup** | Imports, path config, GPU detection |
| **1. Feature Extraction** | Runs ELA/EXIF/FFT/Noise on every sample, builds a feature table |
| **2. Data Exploration** | Plots feature distributions (authentic vs tampered) + correlation heatmap |
| **3. Model Training** | Trains Logistic Regression, Random Forest, Gradient Boosting — picks best via 5-fold CV |
| **4. Evaluation** | Confusion matrix, ROC curve, feature importance chart |
| **5. Signal Separation** | Shows which forensic signals discriminate best (guides weight tuning) |
| **6. Export** | Saves trained model to `models/forensics_model.joblib` |

---

## Step 4 — Apply the Trained Weights Back to the System

After running the notebook, check the **Signal Separation** chart (Cell 5).  
It shows the mean separation score per feature. Use this to update `WEIGHTS` in `analyzer.py`:

```python
# analyzer.py — update based on notebook Cell 5 output
WEIGHTS = {
    "ela_suspicious_pixels": 0.30,   # ← adjust up/down
    "ela_regional_variance":  0.15,
    "exif_risk":              0.30,
    "noise_suspicious_tiles": 0.15,
    "fft_high_freq_ratio":    0.10,
}
```

**Rule of thumb from the audit system:**
- Feature separation > 0.15 → increase weight
- Feature separation < 0.05 → decrease weight (signal is unreliable)

---

## Step 5 — Run the CLI Audit Test

To get an automated weight recommendation without opening the notebook:

```powershell
# Use all available samples
venv\Scripts\python.exe run_audit_test.py

# Or limit to N images per class
venv\Scripts\python.exe run_audit_test.py --original 10 --tampered 10
```

Output includes:
- Accuracy / Precision / Recall
- Per-signal diagnosis (over-performing / under-performing / unreliable)
- Proposed new `WEIGHTS` values (copy-paste ready)
- JSON report saved to `reports/audit_report.json`

---

## GPU Notes (RTX 2050)

| Task | GPU used? |
|------|-----------|
| Image generation (`generate_training_samples.py`) | CPU only (fast enough) |
| Feature extraction (ELA/Noise/FFT) | CPU (numpy/scipy) |
| FFT in notebook | **GPU via `torch.fft`** if CUDA is available |
| ML training (sklearn RF/GB) | CPU (sklearn is CPU-only) |
| Future: CNN/ViT model | **Full GPU** — extend notebook Cell 3 |

To check GPU status at any point:
```powershell
venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_properties(0))"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA: False` | Install CUDA 12.x drivers from nvidia.com, then re-run `install.bat` |
| `ModuleNotFoundError: torch` | Run `install.bat` or the manual pip command in Step 1 |
| `No images found` | Run Step 2 first (`generate_training_samples.py`) |
| Notebook kernel crashes | Make sure you're using `venv\Scripts\jupyter`, not a system jupyter |
| `Invalid quality setting` | Already fixed — pull latest code |

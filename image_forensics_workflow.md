# Auroraa Vault Image Forensics Workflow

This document explains the end-to-end workflow of the image forensics pipeline implemented in the Auroraa Vault codebase. The system is designed to analyze document images for signs of digital manipulation (splicing, AI generation, metadata alteration) and translate complex forensic data into actionable, plain-language insights.

## Overall Architecture

The forensics engine is exposed as a FastAPI REST API (`app.py` / `main.py`). The core analysis is orchestrated by `analyzer.py`, which runs the image through four distinct forensic modules. The numeric results are then scored to produce a final verdict, translated into human-readable findings, and compiled into both visual (PNG) and machine-readable (JSON) reports.

## The Analysis Pipeline

### 1. Upload & Preprocessing (`routes.py`)
- **Endpoint:** `POST /api/analyze`
- **Process:** The system accepts an image upload (JPEG, PNG, WebP, etc.), validates its MIME type, and saves it to a local `uploads/` directory with a unique UUID.
- The path is then passed to the main orchestrator (`analyzer.py`).

### 2. Forensic Modules (`app/image_forensics/core/`)

The image undergoes four parallel analyses to detect different types of manipulation:

#### 2.1 Error Level Analysis (ELA) - `ela.py`
- **Concept:** JPEG compression is lossy. If an image is spliced together from different sources and saved, the different regions will have different compression histories.
- **Workflow:** 
  1. The original image is intentionally re-saved at a known JPEG quality (e.g., 75).
  2. The pixel-by-pixel absolute difference between the original and the re-saved image is computed.
  3. The difference is amplified to create a visible "ELA map".
  4. The module calculates the percentage of "suspicious pixels" and "regional variance" to flag spatially inconsistent compression (a strong indicator of splicing).

#### 2.2 EXIF Metadata Analysis - `exif.py`
- **Concept:** Genuine photos contain metadata directly from the camera (Make, Model, GPS, original timestamp). Forgeries often have missing data, timestamp mismatches, or traces of editing software.
- **Workflow:**
  1. Extracts the EXIF tags from the image.
  2. Checks the `Software` field against a known list of editing tools (Photoshop, GIMP, Midjourney, etc.).
  3. Compares `DateTimeOriginal` with `DateTime` to see if the file was re-saved long after capture.
  4. Checks for the absence of standard camera fields (Make/Model) which might indicate a synthetic or screenshot image.

#### 2.3 Noise Consistency - `noise.py`
- **Concept:** A real photograph has a uniform sensor noise pattern across the entire frame. Spliced or AI-generated regions will disrupt this uniform noise.
- **Workflow:**
  1. The image is divided into small 32×32 pixel tiles.
  2. A Laplacian-based estimator quickly calculates the noise standard deviation (sigma) for each tile.
  3. The module computes the median noise across the whole image. Tiles whose noise deviates significantly from the median are flagged as "suspicious tiles".

#### 2.4 Frequency Domain (FFT) Analysis - `fft.py`
- **Concept:** AI-generated images (GANs, Diffusion models) often leave invisible, periodic artifacts caused by their neural network architecture (e.g., upsampling blocks). These artifacts appear as unnatural spikes in the high-frequency spectrum.
- **Workflow:**
  1. Converts the image to grayscale and computes the 2D Fast Fourier Transform (FFT).
  2. Computes the radially-averaged power spectrum to see how energy is distributed across frequencies.
  3. Flags images with an unusually high ratio of high-frequency energy or periodic spectral bands (cruciform patterns).

### 3. Verdict Scoring (`analyzer.py`)

Once the four modules finish, their individual statistics are combined into a final forgery score.
- **Weights:** ELA suspicious pixels (30%), ELA regional variance (15%), EXIF risk (30%), Noise suspicious tiles (15%), FFT high-freq ratio (10%).
- **Score:** The signals yield a normalized score between 0.0 and 1.0.
- **Verdict Thresholds:**
  - `0.00 – 0.30`: **LIKELY AUTHENTIC** (Green)
  - `0.30 – 0.55`: **INCONCLUSIVE** (Amber)
  - `0.55 – 1.00`: **LIKELY MANIPULATED** (Red)

### 4. Plain-Language Translation (`translate.py`)

Because the target users (e.g., fraud managers at an NBFC) may not understand "Laplacian noise sigma" or "Nyquist frequencies", a translation layer acts as an interpreter.
- It takes the raw numeric stats and the verdict.
- It generates a list of actionable `findings`, categorized by severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`).
- Example: An EXIF timestamp mismatch of 3 hours is translated into: *"Metadata creation date does not match document date — file was modified after original capture."*

### 5. Report Generation & Response (`api.py`)

- **Visual Report:** The pipeline generates a comprehensive PNG (`report.py`) showing the original image next to the ELA heatmap, Noise heatmap, and FFT spectrum.
- **JSON Output:** All raw stats and translated findings are saved to a JSON file.
- **API Response:** The `/api/analyze` endpoint returns the translated findings, the overall verdict/score, the relative URL to the JSON report, and the raw stats for developer use.

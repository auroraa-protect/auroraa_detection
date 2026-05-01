"""
ela.py — Error Level Analysis
──────────────────────────────
Theory:
  JPEG compression is lossy. Every time a JPEG is saved, each 8×8 pixel block
  is re-compressed, which introduces quantisation error. If an image is saved
  once uniformly, all blocks should have roughly the same error level.

  If a region was spliced or copy-pasted from another source, it was compressed
  independently (possibly at a different quality or a different number of times).
  When we force-resave the whole image at a known quality and compute the
  per-pixel absolute difference, spliced regions "light up" because their error
  level is different from the background.

  ELA map = |original_pixels - resaved_pixels| × amplification_factor
"""

import io
import numpy as np
from PIL import Image


def compute_ela(
    image_path: str,
    resave_quality: int = 75,
    amplify: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute the ELA map for a given image.

    Parameters
    ----------
    image_path    : path to the original image (JPEG or PNG)
    resave_quality: JPEG quality used for the resave step (default 75)
    amplify       : multiply the raw difference to make it visible (default 15)

    Returns
    -------
    ela_map       : amplified per-pixel absolute difference as uint8 (H×W×3)
    original_rgb  : original image as numpy array (H×W×3)
    stats         : dict with numeric diagnostics
    """
    # ── 1. Load original ──────────────────────────────────────
    original = Image.open(image_path).convert("RGB")
    orig_arr = np.array(original, dtype=np.float32)

    # ── 2. Resave at known quality ────────────────────────────
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=resave_quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")
    resaved_arr = np.array(resaved, dtype=np.float32)

    # ── 3. Compute difference ─────────────────────────────────
    diff = np.abs(orig_arr - resaved_arr)

    # ── 4. Amplify for visibility ─────────────────────────────
    ela_map = np.clip(diff * amplify, 0, 255).astype(np.uint8)

    # ── 5. Diagnostics ────────────────────────────────────────
    diff_gray = diff.mean(axis=2)          # H×W average across channels
    stats = {
        "mean_ela":        float(diff_gray.mean()),
        "max_ela":         float(diff_gray.max()),
        "std_ela":         float(diff_gray.std()),

        # Regional variance: split into 4 quadrants and compare std across them.
        # Authentic images are spatially consistent; tampered images show spikes
        # in specific regions.
        "regional_variance": _regional_variance(diff_gray),

        # High-suspicion threshold: pixels with error > mean + 2 std
        "suspicious_pixel_pct": _suspicious_pixel_pct(diff_gray),

        "resave_quality":  resave_quality,
        "amplify_factor":  amplify,
        "image_size":      original.size,   # (width, height)
    }

    return ela_map, np.array(original, dtype=np.uint8), stats


def _regional_variance(diff_gray: np.ndarray) -> float:
    """
    Split the image into a 3×3 grid and return the coefficient of variation
    of mean ELA values across tiles. High variance = spatially inconsistent
    error levels = possible splicing.
    """
    h, w = diff_gray.shape
    tile_means = []
    rows, cols = 3, 3
    for r in range(rows):
        for c in range(cols):
            tile = diff_gray[
                r * h // rows : (r + 1) * h // rows,
                c * w // cols : (c + 1) * w // cols,
            ]
            tile_means.append(tile.mean())
    tile_means = np.array(tile_means)
    # Coefficient of variation: std / mean  (scale-free)
    cv = float(tile_means.std() / (tile_means.mean() + 1e-6))
    return round(cv, 4)


def _suspicious_pixel_pct(diff_gray: np.ndarray) -> float:
    """Percentage of pixels whose error exceeds mean + 2×std."""
    threshold = diff_gray.mean() + 2 * diff_gray.std()
    suspicious = (diff_gray > threshold).sum()
    return round(float(suspicious) / diff_gray.size * 100, 2)
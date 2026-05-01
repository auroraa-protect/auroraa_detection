"""
noise.py — Noise Consistency
──────────────────────────────────────────────────────────

1. NOISE CONSISTENCY MAP
   Real photographs have spatially consistent sensor noise across the whole
   frame. If a region was spliced in from another image (or AI-generated
   and pasted), its noise texture will differ from the background.

   Method: estimate noise per non-overlapping 32×32 tile using the
   Laplacian-based estimator (fast, no training needed). Build a noise map
   and flag tiles that deviate significantly from the median.

"""

import numpy as np
from PIL import Image
from skimage import color
from scipy import ndimage


# ── 1. Noise Consistency ──────────────────────────────────────────────────

def _laplacian_sigma(gray_tile: np.ndarray) -> float:
    """
    Noise sigma estimate via Laplacian residuals.
    Apply a 3×3 Laplacian kernel and return std of the result,
    scaled by a constant derived from the MAD estimator.
    """
    kernel = np.array([[0, 1, 0],
                        [1,-4, 1],
                        [0, 1, 0]], dtype=np.float32)
    lap = ndimage.convolve(gray_tile.astype(np.float32), kernel)
    # Robust sigma via median absolute deviation
    median = np.median(np.abs(lap))
    return float(median / 1.4826)   # MAD → Gaussian sigma scale factor


def noise_consistency_map(
    image_path: str,
    tile_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute a per-tile noise map and flag inconsistent regions.

    Returns
    -------
    noise_map   : 2D array of per-tile sigma estimates (shape = grid tiles)
    heatmap     : noise_map upscaled to original image size for visualisation
    stats       : dict with mean, std, suspicious tile info
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    gray = color.rgb2gray(arr)  # H × W

    h, w = gray.shape
    rows = h // tile_size
    cols = w // tile_size

    noise_map = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            tile = gray[
                r * tile_size : (r + 1) * tile_size,
                c * tile_size : (c + 1) * tile_size,
            ]
            # Laplacian-based noise estimation — no extra dependencies.
            # Applies the 3×3 Laplacian kernel and estimates noise from
            # the standard deviation of the residuals (fast & robust).
            sigma = _laplacian_sigma(tile)
            noise_map[r, c] = float(sigma)

    # Upscale to original image dims for overlay
    zoom_r = h / rows
    zoom_c = w / cols
    heatmap = ndimage.zoom(noise_map, (zoom_r, zoom_c), order=1)
    heatmap = np.clip(heatmap, 0, None)

    # ── Forensic stats ────────────────────────────────────────
    median_sigma = float(np.median(noise_map))
    std_sigma    = float(noise_map.std())
    # Tiles more than 2 std above the median are suspicious
    threshold    = median_sigma + 2 * std_sigma
    suspicious   = np.argwhere(noise_map > threshold)

    stats = {
        "median_noise_sigma": round(median_sigma, 5),
        "std_noise_sigma":    round(std_sigma, 5),
        "suspicious_tiles":   len(suspicious),
        "total_tiles":        rows * cols,
        "suspicious_pct":     round(len(suspicious) / (rows * cols) * 100, 2),
        "noise_range":        (round(float(noise_map.min()), 5),
                               round(float(noise_map.max()), 5)),
        "tile_size":          tile_size,
        "grid_shape":         (rows, cols),
    }

    return noise_map, heatmap, stats
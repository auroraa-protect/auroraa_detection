"""
generate_samples.py
────────────────────
Creates two test images for the forensics pipeline:
  1. authentic.jpg  — simulated camera photo with consistent noise
  2. tampered.jpg   — same base with a spliced region (different compression history)
"""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

rng = np.random.default_rng(42)

W, H = 640, 480


def make_base() -> np.ndarray:
    """Synthetic landscape-like image with gradients and texture."""
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(H // 2):
        r = int(30  + (y / (H // 2)) * 60)
        g = int(100 + (y / (H // 2)) * 80)
        b = int(200 - (y / (H // 2)) * 40)
        img[y, :] = [r, g, b]

    # Ground gradient
    for y in range(H // 2, H):
        t = (y - H // 2) / (H // 2)
        r = int(60  + t * 40)
        g = int(120 + t * 20)
        b = int(40  + t * 10)
        img[y, :] = [r, g, b]

    # Add natural per-pixel noise (camera sensor)
    noise = rng.integers(-12, 12, size=(H, W, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def save_jpeg(arr: np.ndarray, path: str, quality: int = 92) -> None:
    Image.fromarray(arr).save(path, format="JPEG", quality=quality)


# ── 1. Authentic ─────────────────────────────────────────────
base = make_base()
save_jpeg(base, "samples/authentic.jpg", quality=92)
print("Saved samples/authentic.jpg")


# ── 2. Tampered ──────────────────────────────────────────────
# Step 1: create the splice patch from a DIFFERENT base (different noise seed)
rng2 = np.random.default_rng(999)
patch_arr = rng2.integers(40, 200, size=(120, 160, 3), dtype=np.uint8)

# Save patch at lower quality (different compression history)
buf = io.BytesIO()
Image.fromarray(patch_arr).save(buf, format="JPEG", quality=60)
buf.seek(0)
patch = np.array(Image.open(buf))

# Step 2: paste into base image
tampered = base.copy()
py, px = 150, 200      # top-left position of splice
ph, pw = patch.shape[:2]
tampered[py : py + ph, px : px + pw] = patch

# Step 3: save tampered image (whole-image quality = 92)
save_jpeg(tampered, "samples/tampered.jpg", quality=92)
print("Saved samples/tampered.jpg")
print("Spliced region: pixels [150:270, 200:360] (different compression history)")
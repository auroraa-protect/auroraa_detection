"""
generate_training_samples.py
─────────────────────────────
Auto-generates 20 AUTHENTIC + 20 TAMPERED sample images for training.

Each image is a fully synthetic scene so the pipeline can run without any
real photographs. Run from the project root:

    python generate_training_samples.py

Output
──────
  samples/original/  ← 20 authentic JPEGs (authentic_00.jpg … authentic_19.jpg)
  samples/tampered/  ← 20 tampered JPEGs  (tampered_00.jpg  … tampered_19.jpg)
"""

import io
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import docx


ROOT   = Path(__file__).parent
ORIG   = ROOT / "samples" / "original"
TAMP   = ROOT / "samples" / "tampered"
ORIG.mkdir(parents=True, exist_ok=True)
TAMP.mkdir(parents=True, exist_ok=True)

N       = 20        # images per class
W, H    = 640, 480  # canvas size

# ─────────────────────────────────────────────────────────────────────────────
# Scene generators — each returns an (H, W, 3) uint8 array
# ─────────────────────────────────────────────────────────────────────────────

def _clip(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def scene_sky_ground(rng, seed):
    """Gradient sky + ground with per-pixel sensor noise."""
    img = np.zeros((H, W, 3), dtype=np.int16)
    sky_r = rng.integers(20, 80)
    sky_g = rng.integers(80, 160)
    sky_b = rng.integers(160, 240)
    for y in range(H // 2):
        t = y / (H // 2)
        img[y] = [sky_r + int(t * 40), sky_g + int(t * 30), sky_b - int(t * 40)]
    gnd_r = rng.integers(40, 100)
    gnd_g = rng.integers(90, 150)
    gnd_b = rng.integers(20, 60)
    for y in range(H // 2, H):
        t = (y - H // 2) / (H // 2)
        img[y] = [gnd_r + int(t * 20), gnd_g + int(t * 10), gnd_b + int(t * 5)]
    noise = rng.integers(-10, 10, size=(H, W, 3), dtype=np.int16)
    return _clip(img + noise)


def scene_gradient_texture(rng, seed):
    """Diagonal gradient + uniform mid-freq noise texture."""
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    diag = (xx + yy) / 2
    r = _clip(diag * rng.integers(120, 220) + rng.integers(-15, 15, (H, W)))
    g = _clip((1 - diag) * rng.integers(80, 180) + rng.integers(-15, 15, (H, W)))
    b = _clip(diag * rng.integers(60, 200) + rng.integers(-15, 15, (H, W)))
    return np.stack([r, g, b], axis=2)


def scene_striped(rng, seed):
    """Horizontal stripes with noise (simulates textured document)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    stripe_h = rng.integers(20, 60)
    for y in range(H):
        phase = (y // stripe_h) % 2
        base  = rng.integers(60, 180, 3) if phase else rng.integers(120, 230, 3)
        img[y] = base
    noise = rng.integers(-8, 8, size=(H, W, 3), dtype=np.int16)
    return _clip(img.astype(np.int16) + noise)


def scene_radial(rng, seed):
    """Radial gradient from a random centre point."""
    cx = rng.integers(W // 4, 3 * W // 4)
    cy = rng.integers(H // 4, 3 * H // 4)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / dist.max()
    col  = rng.integers(50, 200, 3).astype(float)
    img  = (1 - dist[:, :, None]) * col[None, None, :]
    noise = rng.integers(-12, 12, size=(H, W, 3), dtype=np.int16).astype(np.float64)
    return _clip(img + noise)


SCENE_FNS = [scene_sky_ground, scene_gradient_texture, scene_striped, scene_radial]


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))


def save_all_formats(arr, path, quality=92):
    """Saves the array as JPEG, PDF, and DOCX."""
    pil_img = _to_pil(arr)
    
    # 1. Save JPEG
    pil_img.save(str(path), format="JPEG", quality=quality)
    
    # 2. Save PDF
    pdf_path = path.with_suffix(".pdf")
    pil_img.save(str(pdf_path), "PDF", resolution=100.0)
    
    # 3. Save DOCX
    docx_path = path.with_suffix(".docx")
    doc = docx.Document()
    # Use BytesIO to insert image into docx without extra temp files
    img_buf = io.BytesIO()
    pil_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    doc.add_picture(img_buf, width=docx.shared.Inches(5))
    doc.save(str(docx_path))


# ─────────────────────────────────────────────────────────────────────────────
# Authentic image generator
# ─────────────────────────────────────────────────────────────────────────────

def make_authentic(i):
    """
    Authentic = one consistent compression pass at high quality.
    Noise is spatially uniform (simulates camera sensor).
    """
    rng   = np.random.default_rng(i * 100)
    scene = SCENE_FNS[i % len(SCENE_FNS)](rng, i)

    path  = ORIG / f"authentic_{i:02d}.jpg"
    save_all_formats(scene, path, quality=int(rng.integers(88, 95)))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Tampered image generators — four distinct manipulation types
# ─────────────────────────────────────────────────────────────────────────────

def _splice_patch(base_arr, rng, patch_quality=55):
    """Paste a region compressed at a different quality (classic ELA tell)."""
    rng2        = np.random.default_rng(int(rng.integers(0, 9999)))
    ph          = int(rng.integers(80, 160))
    pw          = int(rng.integers(100, 200))
    py          = int(rng.integers(20, H - ph - 20))
    px          = int(rng.integers(20, W - pw - 20))

    patch_scene = SCENE_FNS[int(rng.integers(0, len(SCENE_FNS)))](rng2, 999)
    patch_crop  = patch_scene[py: py + ph, px: px + pw]

    buf = io.BytesIO()
    Image.fromarray(patch_crop).save(buf, format="JPEG", quality=int(patch_quality))
    buf.seek(0)
    patch = np.array(Image.open(buf).convert("RGB"))

    result = base_arr.copy()
    result[py: py + ph, px: px + pw] = patch
    return result


def _double_compress(base_arr, rng):
    """
    Double JPEG compression: save at low quality then re-save at high quality.
    Classic ELA / noise inconsistency signal.
    """
    buf = io.BytesIO()
    Image.fromarray(base_arr).save(buf, format="JPEG", quality=int(rng.integers(40, 60)))
    buf.seek(0)
    degraded = np.array(Image.open(buf).convert("RGB"))
    return degraded


def _copy_move(base_arr, rng):
    """Copy a rectangular region from one location and paste it elsewhere."""
    ph = rng.integers(60, 130)
    pw = rng.integers(80, 160)
    src_y = rng.integers(0, H // 2 - ph)
    src_x = rng.integers(0, W // 2 - pw)
    dst_y = rng.integers(H // 2, H - ph)
    dst_x = rng.integers(W // 2, W - pw)
    result = base_arr.copy()
    result[dst_y: dst_y + ph, dst_x: dst_x + pw] = \
        base_arr[src_y: src_y + ph, src_x: src_x + pw]
    return result


def _add_ai_pattern(base_arr, rng):
    """
    Simulate an AI-generated region: overwrite a block with a regular
    sinusoidal pattern (mimics diffusion model periodic artifacts).
    """
    ph = rng.integers(100, 180)
    pw = rng.integers(120, 220)
    py = rng.integers(0, H - ph)
    px = rng.integers(0, W - pw)
    freq = rng.uniform(4, 12)
    yy, xx = np.mgrid[0:ph, 0:pw]
    wave   = (np.sin(xx * freq / pw * 2 * np.pi) *
               np.sin(yy * freq / ph * 2 * np.pi))
    col    = rng.integers(80, 200, 3).astype(float)
    patch  = _clip(col[None, None, :] + wave[:, :, None] * 40)
    result = base_arr.copy()
    result[py: py + ph, px: px + pw] = patch
    return result


TAMPER_FNS = [_splice_patch, _double_compress, _copy_move, _add_ai_pattern]


def make_tampered(i):
    rng   = np.random.default_rng(i * 100 + 1)
    scene = SCENE_FNS[i % len(SCENE_FNS)](rng, i)

    tamper_fn = TAMPER_FNS[i % len(TAMPER_FNS)]

    # _splice_patch and _add_ai_pattern need an rng; _double_compress and
    # _copy_move also need one — pass rng to all via keyword-safe call
    try:
        manipulated = tamper_fn(scene, rng)
    except TypeError:
        manipulated = tamper_fn(scene, rng)

    path = TAMP / f"tampered_{i:02d}.jpg"
    save_all_formats(manipulated, path, quality=int(rng.integers(85, 94)))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

print(f"Generating {N} authentic images -> {ORIG}")
for i in range(N):
    p = make_authentic(i)
    print(f"  [OK]  {p.name}")

print(f"\nGenerating {N} tampered images  -> {TAMP}")
for i in range(N):
    p = make_tampered(i)
    print(f"  [X]   {p.name}")

print(f"\nDone. {N * 2 * 3} files total (JPG, PDF, DOCX).")
print(f"  {ORIG} -- {N * 3} files")
print(f"  {TAMP} -- {N * 3} files")

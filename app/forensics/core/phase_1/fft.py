"""
fft.py — Frequency Domain Analysis
──────────────────────────────────────────────────────────

1. FREQUENCY DOMAIN (FFT) ANALYSIS
   AI-generated images (diffusion models, GANs) leave spectral fingerprints:
   - Diffusion UNets produce periodic grid artifacts at Nyquist frequencies
   - GAN upsampling creates peaks at regular intervals in the power spectrum
   - Inpainting fills show soft-boundary transitions visible as low-frequency
     blobs in the difference spectrum

   Method: compute the 2D FFT of the luminance channel, shift zero-frequency
   to center, and compute the radially-averaged power spectrum. Spikes in the
   high-frequency band (> Nyquist/4) that are unusually strong compared to
   natural image statistics are flagged.
"""

import numpy as np
from PIL import Image
from skimage import color
from scipy import ndimage

# ── Frequency Domain (FFT) Analysis ───────────────────────────────────

def fft_spectrum_analysis(
    image_path: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute the 2D power spectrum of the luminance channel and
    detect GAN/diffusion spectral artifacts.

    Returns
    -------
    spectrum_log  : log10(|FFT|²) centered, normalised to [0,1] (H×W float)
    radial_power  : 1D radially-averaged power vs. spatial frequency
    stats         : dict with peak locations and anomaly flags
    """
    img  = Image.open(image_path).convert("RGB")
    arr  = np.array(img, dtype=np.float32) / 255.0
    gray = color.rgb2gray(arr)          # [0,1] luminance

    # ── 2D FFT ───────────────────────────────────────────────
    fft       = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)    # zero-freq at center
    power     = np.abs(fft_shift) ** 2

    # Log-compress for visualisation (add 1 to avoid log(0))
    log_power = np.log10(power + 1.0)
    spectrum_log = (log_power - log_power.min()) / (np.ptp(log_power) + 1e-9)

    # ── Radially-averaged power spectrum ─────────────────────
    radial_power = _radial_profile(power)

    # ── Detect spectral peaks (GAN/diffusion artifacts) ──────
    flags = _detect_spectral_anomalies(spectrum_log, radial_power)

    stats = {
        "image_size":   img.size,
        "fft_shape":    gray.shape,
        "spectral_flags": flags,
        "high_freq_energy_ratio": _high_freq_ratio(radial_power),
        "peak_freq_bin":          int(np.argmax(radial_power[5:])) + 5,
    }

    return spectrum_log, radial_power, stats


def _radial_profile(power: np.ndarray) -> np.ndarray:
    """Compute the mean power at each integer radial distance from centre."""
    h, w  = power.shape
    cy, cx = h // 2, w // 2
    y, x  = np.ogrid[-cy:h - cy, -cx:w - cx]
    r     = np.sqrt(x * x + y * y).astype(int)

    max_r = min(cy, cx)
    profile = np.zeros(max_r)
    for i in range(max_r):
        mask = r == i
        if mask.any():
            profile[i] = power[mask].mean()
    return profile


def _high_freq_ratio(radial: np.ndarray) -> float:
    """
    Ratio of energy in the top 25% of spatial frequencies to total energy.
    Natural images are dominated by low frequencies; AI images often have
    elevated high-frequency energy.
    """
    cut = len(radial) * 3 // 4
    total   = radial.sum() + 1e-9
    hi_freq = radial[cut:].sum()
    return round(float(hi_freq / total), 4)


def _detect_spectral_anomalies(
    spectrum: np.ndarray,
    radial: np.ndarray,
) -> list[str]:
    """
    Heuristic checks for patterns typical of AI-generated or inpainted images.
    These are not definitive; they raise flags for further inspection.
    """
    flags = []
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    # Check for cruciform pattern (bright cross through DC component)
    # — common in GAN and diffusion outputs due to periodic upsampling
    horizontal_band = spectrum[cy - 2 : cy + 3, :].mean()
    vertical_band   = spectrum[:, cx - 2 : cx + 3].mean()
    global_mean     = spectrum.mean()

    if horizontal_band > global_mean * 2.0:
        flags.append(
            "⚠️  Horizontal spectral band detected. "
            "May indicate periodic row-level artifacts from upsampling."
        )
    if vertical_band > global_mean * 2.0:
        flags.append(
            "⚠️  Vertical spectral band detected. "
            "May indicate periodic column-level artifacts from upsampling."
        )

    # Elevated high-frequency energy
    hi_ratio = _high_freq_ratio(radial)
    if hi_ratio > 0.12:
        flags.append(
            f"⚠️  Elevated high-frequency energy ratio ({hi_ratio:.3f}). "
            f"AI-generated images often show this; natural images typically < 0.08."
        )

    # Flat spectrum tail (diffusion models produce unusually smooth spectra)
    tail = radial[len(radial) * 3 // 4 :]
    if tail.std() / (tail.mean() + 1e-9) < 0.15:
        flags.append(
            "ℹ️  Unusually smooth high-frequency spectrum tail. "
            "Consistent with diffusion model output."
        )

    return flags if flags else ["✅ No obvious spectral anomalies detected."]
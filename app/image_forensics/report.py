"""
report.py — Visual Forensic Report Generator
──────────────────────────────────────────────
Saves a multi-panel matplotlib figure that shows:
  Panel 1 : Original image
  Panel 2 : ELA map (amplified difference)
  Panel 3 : ELA heatmap overlay on original
  Panel 4 : Noise consistency heatmap
  Panel 5 : FFT power spectrum
  Panel 6 : Radially-averaged power spectrum (line chart)
  + Text summary sidebar
"""

import json
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image


# ── Color scheme ─────────────────────────────────────────────
DARK  = "#0A1628"
MID   = "#0F2040"
CARD  = "#132848"
CYAN  = "#00C9E0"
WHITE = "#FFFFFF"
OFF   = "#CBD5E1"
RED   = "#EF4444"
AMB   = "#F59E0B"
GRN   = "#10B981"

RISK_COLOR = {"low": GRN, "medium": AMB, "high": RED, "unknown": OFF}


def generate_report(
    image_path: str,
    ela_map: np.ndarray,
    original_rgb: np.ndarray,
    ela_stats: dict,
    exif_result: dict,
    noise_heatmap: np.ndarray,
    noise_stats: dict,
    spectrum_log: np.ndarray,
    radial_power: np.ndarray,
    fft_stats: dict,
    findings: list | None = None,
    output_path: str = "forensic_report.png",
) -> str:
    """
    Build and save the full forensic report figure.
    Returns the output path.
    """
    fig = plt.figure(figsize=(22, 14), facecolor=DARK)
    fig.suptitle(
        f"Image Forensics Report  ·  {Path(image_path).name}",
        color=WHITE, fontsize=16, fontweight="bold", y=0.97, x=0.42,
    )

    # ── Grid: 2 rows × 3 cols (image panels) + right sidebar ─
    outer = gridspec.GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[3.2, 1],
        wspace=0.04,
        left=0.03, right=0.97, top=0.93, bottom=0.04,
    )
    panel_grid = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0], hspace=0.32, wspace=0.12
    )

    # ── Panel helper ─────────────────────────────────────────
    def make_ax(row, col, title, subtitle=""):
        ax = fig.add_subplot(panel_grid[row, col])
        ax.set_facecolor(CARD)
        ax.set_title(
            f"{title}\n{subtitle}" if subtitle else title,
            color=WHITE, fontsize=10, fontweight="bold", pad=6,
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(CYAN); spine.set_linewidth(0.8)
        return ax

    h, w = original_rgb.shape[:2]

    # ── 1. Original Image ─────────────────────────────────────
    ax1 = make_ax(0, 0, "Original Image", f"{w}×{h}px")
    ax1.imshow(original_rgb)

    # ── 2. ELA Map ────────────────────────────────────────────
    ax2 = make_ax(
        0, 1, "ELA Map",
        f"quality={ela_stats['resave_quality']}  amplify×{ela_stats['amplify_factor']}"
    )
    im2 = ax2.imshow(ela_map, cmap="hot")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=OFF)

    # ── 3. ELA Overlay ────────────────────────────────────────
    ax3 = make_ax(
        0, 2, "ELA Overlay",
        f"suspicious pixels: {ela_stats['suspicious_pixel_pct']:.1f}%"
    )
    ela_gray = ela_map.mean(axis=2)
    ax3.imshow(original_rgb, alpha=0.55)
    ax3.imshow(ela_gray, cmap="hot", alpha=0.55,
               norm=Normalize(vmin=0, vmax=ela_gray.max()))

    # ── 4. Noise Heatmap ─────────────────────────────────────
    ax4 = make_ax(
        1, 0, "Noise Consistency Map",
        f"suspicious tiles: {noise_stats['suspicious_pct']:.1f}%"
    )
    im4 = ax4.imshow(noise_heatmap, cmap="plasma")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=OFF)

    # ── 5. FFT Power Spectrum ─────────────────────────────────
    ax5 = make_ax(1, 1, "FFT Power Spectrum", "log₁₀(|FFT|²)")
    ax5.imshow(spectrum_log, cmap="inferno")

    # ── 6. Radial Power Profile ───────────────────────────────
    ax6 = make_ax(1, 2, "Radial Power Profile", "spatial frequency →")
    ax6.set_facecolor(CARD)
    ax6.plot(radial_power, color=CYAN, linewidth=1.2)
    # Mark high-frequency cutoff
    cut = len(radial_power) * 3 // 4
    ax6.axvline(cut, color=AMB, linestyle="--", linewidth=0.8, label="HF cutoff (75%)")
    ax6.set_xlabel("Spatial frequency (bins)", color=OFF, fontsize=8)
    ax6.set_ylabel("Mean power", color=OFF, fontsize=8)
    ax6.tick_params(colors=OFF, labelsize=7)
    ax6.legend(fontsize=7, facecolor=MID, labelcolor=OFF, framealpha=0.7)
    ax6.set_xticks([]); ax6.set_yticks([])  # keep clean; override parent remove

    # ── Sidebar ───────────────────────────────────────────────
    ax_side = fig.add_subplot(outer[1])
    ax_side.set_facecolor(MID)
    ax_side.set_xticks([]); ax_side.set_yticks([])
    for spine in ax_side.spines.values():
        spine.set_edgecolor(CYAN); spine.set_linewidth(0.8)

    _draw_sidebar(fig, ax_side, ela_stats, exif_result, noise_stats, fft_stats,
                  findings or [])

    plt.savefig(output_path, dpi=130, facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _draw_sidebar(fig, ax, ela_stats, exif_result, noise_stats, fft_stats,
                  findings: list | None = None):
    """Render the text summary sidebar using figure-space annotations."""
    # We'll place text via ax.text with normalised axes coords
    risk     = exif_result.get("risk_level", "unknown")
    rc       = RISK_COLOR.get(risk, OFF)

    lines = []
    lines.append(("FORENSIC SUMMARY", WHITE, 11, True))
    lines.append(("─" * 30, CYAN, 8, False))

    # ── EXIF ─────────────────────────────────────────────────
    lines.append(("EXIF ANALYSIS", CYAN, 9, True))
    lines.append((f"Risk Level: {risk.upper()}", rc, 9, True))
    raw = exif_result.get("raw", {})
    make  = raw.get("Make",  "—")
    model = raw.get("Model", "—")
    sw    = raw.get("Software", "—")
    lines.append((f"Camera: {make} {model}", OFF, 8, False))
    lines.append((f"Software: {_trunc(str(sw), 28)}", OFF, 8, False))
    dt_orig = raw.get("DateTimeOriginal", "—")
    lines.append((f"Captured: {dt_orig}", OFF, 8, False))

    for flag in exif_result.get("flags", []):
        wrapped = textwrap.wrap(flag, 32)
        for wl in wrapped:
            fc = RED if "🚨" in flag else AMB if "⚠️" in flag else OFF
            lines.append((wl, fc, 7.5, False))

    lines.append(("", WHITE, 4, False))  # spacer
    lines.append(("─" * 30, CYAN, 8, False))

    # ── ELA ──────────────────────────────────────────────────
    lines.append(("ELA STATISTICS", CYAN, 9, True))
    lines.append((f"Mean ELA:   {ela_stats['mean_ela']:.2f}", OFF, 8, False))
    lines.append((f"Max ELA:    {ela_stats['max_ela']:.2f}", OFF, 8, False))
    lines.append((f"Std ELA:    {ela_stats['std_ela']:.2f}", OFF, 8, False))
    lines.append((f"Regional variance: {ela_stats['regional_variance']:.4f}", OFF, 8, False))
    sp_pct = ela_stats["suspicious_pixel_pct"]
    sp_col = RED if sp_pct > 10 else AMB if sp_pct > 5 else GRN
    lines.append((f"Suspicious pixels: {sp_pct:.1f}%", sp_col, 8, True))

    lines.append(("", WHITE, 4, False))
    lines.append(("─" * 30, CYAN, 8, False))

    # ── Noise ─────────────────────────────────────────────────
    lines.append(("NOISE CONSISTENCY", CYAN, 9, True))
    lines.append((f"Median σ: {noise_stats['median_noise_sigma']:.5f}", OFF, 8, False))
    lines.append((f"Std σ:    {noise_stats['std_noise_sigma']:.5f}", OFF, 8, False))
    ns_pct = noise_stats["suspicious_pct"]
    ns_col = RED if ns_pct > 15 else AMB if ns_pct > 8 else GRN
    lines.append((f"Suspicious tiles: {ns_pct:.1f}%", ns_col, 8, True))

    lines.append(("", WHITE, 4, False))
    lines.append(("─" * 30, CYAN, 8, False))

    # ── FFT ──────────────────────────────────────────────────
    lines.append(("FFT SPECTRUM", CYAN, 9, True))
    hfr = fft_stats.get("high_freq_energy_ratio", 0)
    hf_col = RED if hfr > 0.12 else AMB if hfr > 0.08 else GRN
    lines.append((f"HF energy ratio: {hfr:.4f}", hf_col, 8, True))
    for flag in fft_stats.get("spectral_flags", []):
        wrapped = textwrap.wrap(flag, 32)
        for wl in wrapped:
            fc = AMB if "⚠️" in flag else OFF
            lines.append((wl, fc, 7.5, False))

    # ── Plain-language findings ─────────────────────────────
    if findings:
        lines.append(("", WHITE, 4, False))
        lines.append(("─" * 30, CYAN, 8, False))
        lines.append(("PLAIN-LANGUAGE FINDINGS", CYAN, 9, True))

        _SEV_COLOR = {
            "CRITICAL": RED, "HIGH": RED, "MEDIUM": AMB,
            "LOW": GRN, "INFO": OFF,
        }

        import textwrap as _tw
        for f in findings:
            sev   = f.get("severity", "INFO")
            fc    = _SEV_COLOR.get(sev, OFF)
            badge = f"[{sev}]"
            msg   = f.get("message", "")
            # Badge line
            lines.append((badge, fc, 8, True))
            # Message wrapped to 32 chars
            for wl in _tw.wrap(msg, 32):
                lines.append((wl, OFF, 7.5, False))
            lines.append(("", WHITE, 3, False))  # micro-spacer

    # ── Render lines ─────────────────────────────────────────
    y = 0.97
    line_h = 0.027
    for text, color, size, bold in lines:
        if not text:
            y -= line_h * 0.6
            continue
        ax.text(
            0.05, y, text,
            transform=ax.transAxes,
            color=color, fontsize=size,
            fontweight="bold" if bold else "normal",
            verticalalignment="top",
            fontfamily="monospace",
        )
        y -= line_h * (size / 8.5)
        if y < 0.02:
            break


def save_json_report(
    image_path: str,
    ela_stats: dict,
    exif_result: dict,
    noise_stats: dict,
    fft_stats: dict,
    output_path: str = "forensic_report.json",
) -> str:
    """Save all numeric results as machine-readable JSON."""
    report = {
        "image": image_path,
        "ela":   ela_stats,
        "exif": {
            "risk_level": exif_result.get("risk_level"),
            "flags":      exif_result.get("flags"),
            "summary":    exif_result.get("summary"),
            "gps":        {k: v for k, v in exif_result.get("gps", {}).items()
                           if k != "raw"},
            "key_fields": {
                k: str(v) for k, v in exif_result.get("raw", {}).items()
                if k in {"Make", "Model", "Software", "DateTime",
                         "DateTimeOriginal", "DateTimeDigitized"}
            },
        },
        "noise": noise_stats,
        "fft":   fft_stats,
    }

    def _make_serializable(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, tuple): return list(obj)
        return str(obj)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=_make_serializable)
    return output_path


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"
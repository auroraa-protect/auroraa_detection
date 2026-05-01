"""
analyzer.py — Image Forensics Orchestrator
────────────────────────────────────────────
Single entry point that runs all forensic modules and produces:
  • A full visual report (PNG)
  • A machine-readable JSON report
  • A console summary with overall verdict

Usage
─────
  # From Python
  from analyzer import analyze
  result = analyze("photo.jpg")

  # From CLI
  python analyzer.py photo.jpg [--output reports/] [--ela-quality 75]

Pipeline
────────
  1. ELA       — JPEG re-save error level analysis
  2. EXIF      — Metadata extraction and forensic flagging
  3. Noise     — Tile-level noise consistency map
  4. FFT       — Frequency domain spectral analysis
  5. Verdict   — Weighted score combining all signals
  6. Report    — Visual PNG + JSON output
"""

import sys  
import argparse
import json
from pathlib import Path

import numpy as np

from app.image_forensics.core.ela import compute_ela
from app.image_forensics.core.exif import extract_exif
from app.image_forensics.core.noise import noise_consistency_map
from app.image_forensics.core.fft import fft_spectrum_analysis
from app.image_forensics.report import generate_report, save_json_report
from app.image_forensics.translate import translate


# ─────────────────────────────────────────────────────────────
# Verdict scoring weights
# ─────────────────────────────────────────────────────────────
# Each signal contributes a score in [0, 1].
# Weights must sum to 1.0.
WEIGHTS = {
    "ela_suspicious_pixels": 0.30,   # % of high-ELA pixels
    "ela_regional_variance":  0.15,   # spatial inconsistency of ELA
    "exif_risk":              0.30,   # EXIF risk level
    "noise_suspicious_tiles": 0.15,   # % of high-noise tiles
    "fft_high_freq_ratio":    0.10,   # elevated high-frequency energy
}


def compute_verdict(
    ela_stats: dict,
    exif_result: dict,
    noise_stats: dict,
    fft_stats: dict,
) -> dict:
    """
    Combine all forensic signals into a single [0,1] forgery probability
    and a traffic-light verdict.

    Score thresholds:
      0.00 – 0.30 : LIKELY AUTHENTIC
      0.30 – 0.55 : INCONCLUSIVE — further review recommended
      0.55 – 1.00 : LIKELY MANIPULATED
    """
    scores = {}

    # ── ELA suspicious pixels ─────────────────────────────────
    # Normalise: 0% → 0.0,  ≥20% → 1.0
    sp_pct = ela_stats.get("suspicious_pixel_pct", 0)
    scores["ela_suspicious_pixels"] = min(sp_pct / 20.0, 1.0)

    # ── ELA regional variance ─────────────────────────────────
    # Normalise: 0.0 → 0.0,  ≥0.5 (high spatial inconsistency) → 1.0
    rv = ela_stats.get("regional_variance", 0)
    scores["ela_regional_variance"] = min(rv / 0.5, 1.0)

    # ── EXIF risk ─────────────────────────────────────────────
    exif_map = {"low": 0.1, "medium": 0.55, "high": 0.9, "unknown": 0.5}
    scores["exif_risk"] = exif_map.get(exif_result.get("risk_level", "unknown"), 0.5)

    # ── Noise suspicious tiles ────────────────────────────────
    # Normalise: 0% → 0.0,  ≥25% → 1.0
    nt_pct = noise_stats.get("suspicious_pct", 0)
    scores["noise_suspicious_tiles"] = min(nt_pct / 25.0, 1.0)

    # ── FFT high-frequency energy ─────────────────────────────
    # Natural images: ~0.04–0.08. AI/manipulated: often >0.12
    hfr = fft_stats.get("high_freq_energy_ratio", 0)
    scores["fft_high_freq_ratio"] = min(max((hfr - 0.06) / 0.10, 0.0), 1.0)

    # ── Weighted total ────────────────────────────────────────
    total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    total = round(float(total), 4)

    if total < 0.30:
        verdict = "LIKELY AUTHENTIC"
        color   = "green"
    elif total < 0.55:
        verdict = "INCONCLUSIVE"
        color   = "amber"
    else:
        verdict = "LIKELY MANIPULATED"
        color   = "red"

    return {
        "forgery_score":    total,
        "verdict":          verdict,
        "color":            color,
        "component_scores": scores,
        "weights":          WEIGHTS,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def analyze(
    image_path: str,
    output_dir: str = "reports",
    ela_quality: int = 75,
    ela_amplify: float = 15.0,
    verbose: bool = True,
) -> dict:
    """
    Run the full forensic pipeline on a single image.

    Parameters
    ----------
    image_path  : path to image file
    output_dir  : directory for report outputs
    ela_quality : JPEG re-save quality for ELA (default 75)
    ela_amplify : ELA amplification factor (default 15)
    verbose     : print progress to stdout

    Returns
    -------
    Full results dict with all stats, verdict, and output file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    def log(msg):
        if verbose:
            print(msg)

    log(f"\n{'═'*60}")
    log(f"  IMAGE FORENSICS ANALYZER")
    log(f"  File: {image_path}")
    log(f"{'═'*60}")

    # ── Step 1: ELA ──────────────────────────────────────────
    log("  [1/4] Running Error Level Analysis...")
    ela_map, original_rgb, ela_stats = compute_ela(
        image_path, resave_quality=ela_quality, amplify=ela_amplify
    )
    log(f"        Mean ELA={ela_stats['mean_ela']:.2f}  "
        f"Suspicious pixels={ela_stats['suspicious_pixel_pct']:.1f}%  "
        f"Regional variance={ela_stats['regional_variance']:.4f}")

    # ── Step 2: EXIF ─────────────────────────────────────────
    log("  [2/4] Extracting EXIF metadata...")
    exif_result = extract_exif(image_path)
    log(f"        Risk={exif_result['risk_level'].upper()}  "
        f"Flags={len(exif_result['flags'])}")
    for flag in exif_result["flags"]:
        log(f"        {flag}")

    # ── Step 3: Noise ─────────────────────────────────────────
    log("  [3/4] Computing noise consistency map...")
    _, noise_heatmap, noise_stats = noise_consistency_map(image_path)
    log(f"        Median σ={noise_stats['median_noise_sigma']:.5f}  "
        f"Suspicious tiles={noise_stats['suspicious_pct']:.1f}%")

    # ── Step 4: FFT ───────────────────────────────────────────
    log("  [4/4] Frequency domain analysis...")
    spectrum_log, radial_power, fft_stats = fft_spectrum_analysis(image_path)
    log(f"        HF energy ratio={fft_stats['high_freq_energy_ratio']:.4f}")
    for flag in fft_stats["spectral_flags"]:
        log(f"        {flag}")

    # ── Verdict ──────────────────────────────────────────────
    verdict = compute_verdict(ela_stats, exif_result, noise_stats, fft_stats)
    log(f"\n  {'─'*56}")
    log(f"  FORGERY SCORE : {verdict['forgery_score']:.4f} / 1.0000")
    log(f"  VERDICT       : {verdict['verdict']}")
    log(f"  {'─'*56}")

    # ── Reports ──────────────────────────────────────────────
    png_path  = str(Path(output_dir) / f"{stem}_forensic_report.png")
    json_path = str(Path(output_dir) / f"{stem}_forensic_report.json")

    log(f"\n  Generating visual report → {png_path}")
    # findings are assembled after generate_report, so we pass a sentinel
    # and let report.py render them via a second call after translate() runs.
    # We build them now from a preliminary result dict so the PNG is complete.
    _pre_result = {
        "ela": ela_stats, "exif": exif_result,
        "noise": noise_stats, "fft": fft_stats,
        "verdict": verdict,
    }

    _findings_for_png = translate(_pre_result)

    generate_report(
        image_path=image_path,
        ela_map=ela_map,
        original_rgb=original_rgb,
        ela_stats=ela_stats,
        exif_result=exif_result,
        noise_heatmap=noise_heatmap,
        noise_stats=noise_stats,
        spectrum_log=spectrum_log,
        radial_power=radial_power,
        fft_stats=fft_stats,
        findings=_findings_for_png,
        output_path=png_path,
    )

    log(f"  Saving JSON report    → {json_path}")
    save_json_report(
        image_path=image_path,
        ela_stats=ela_stats,
        exif_result=exif_result,
        noise_stats=noise_stats,
        fft_stats=fft_stats,
        output_path=json_path,
    )

    log(f"\n  Done. Reports saved to {output_dir}/\n")

    # ── Plain-language findings ──────────────────────────────
    full_result = {
        "image":       image_path,
        "ela":         ela_stats,
        "exif":        exif_result,
        "noise":       noise_stats,
        "fft":         fft_stats,
        "verdict":     verdict,
        "report_png":  png_path,
        "report_json": json_path,
    }
    full_result["findings"] = translate(full_result)
    return full_result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Image Forensics Analyzer — ELA + EXIF + Noise + FFT"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--output", default="reports",
        help="Output directory for reports (default: reports/)"
    )
    parser.add_argument(
        "--ela-quality", type=int, default=75,
        help="JPEG quality for ELA resave step (default: 75)"
    )
    parser.add_argument(
        "--ela-amplify", type=float, default=15.0,
        help="Amplification factor for ELA visualisation (default: 15)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress console output"
    )
    args = parser.parse_args()

    result = analyze(
        image_path=args.image,
        output_dir=args.output,
        ela_quality=args.ela_quality,
        ela_amplify=args.ela_amplify,
        verbose=not args.quiet,
    )

    # Exit code: 0 = authentic/inconclusive, 1 = manipulated
    sys.exit(1 if result["verdict"]["color"] == "red" else 0)


if __name__ == "__main__":
    _cli()
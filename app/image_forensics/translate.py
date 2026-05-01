"""
translate.py — Forensic Score → Plain-Language Findings
─────────────────────────────────────────────────────────
Converts the raw numeric output from analyze() into a list of human-readable
findings that a fraud manager at an NBFC can act on immediately — no
data-science knowledge required.

Each finding is a dict:
  {
    "field":    str  — the document region or data point being flagged
    "signal":   str  — internal signal identifier (for programmatic use)
    "severity": str  — "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFO"
    "message":  str  — one-line plain-English description
    "evidence": str  — numeric detail supporting the finding
  }

Usage
─────
  from image_forensics.core.translate import translate
  findings = translate(analyze_result)
"""

from __future__ import annotations
from typing import Any


# ── Severity ordering (for sorting) ──────────────────────────────────────
_SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}


def translate(result: dict[str, Any]) -> list[dict[str, str]]:
    """
    Translate an analyze() result dict into a list of plain-language findings.

    Parameters
    ----------
    result : the dict returned by image_forensics.analyzer.analyze()

    Returns
    -------
    List of finding dicts, sorted by severity (most critical first).
    """
    findings: list[dict[str, str]] = []

    ela_stats   = result.get("ela",    {})
    exif_result = result.get("exif",   {})
    noise_stats = result.get("noise",  {})
    fft_stats   = result.get("fft",    {})
    verdict     = result.get("verdict", {})

    # ── 1. Overall verdict ────────────────────────────────────
    findings += _verdict_findings(verdict)

    # ── 2. ELA (pixel manipulation) ──────────────────────────
    findings += _ela_findings(ela_stats)

    # ── 3. EXIF (metadata) ───────────────────────────────────
    findings += _exif_findings(exif_result)

    # ── 4. Noise consistency ─────────────────────────────────
    findings += _noise_findings(noise_stats)

    # ── 5. Frequency domain ───────────────────────────────────
    findings += _fft_findings(fft_stats)

    # Sort by severity rank
    findings.sort(key=lambda f: _SEVERITY_RANK.get(f["severity"], 99))
    return findings


# ─────────────────────────────────────────────────────────────
# Sub-translators
# ─────────────────────────────────────────────────────────────

def _verdict_findings(verdict: dict) -> list[dict]:
    score   = verdict.get("forgery_score", 0)
    label   = verdict.get("verdict", "UNKNOWN")
    pct     = round(score * 100, 1)

    if label == "LIKELY MANIPULATED":
        return [_finding(
            field    = "Document Overall",
            signal   = "overall_verdict",
            severity = "CRITICAL",
            message  = "ALERT: Document shows strong indicators of manipulation",
            evidence = f"Forgery probability score: {pct}% (threshold: 55%)",
        )]
    elif label == "INCONCLUSIVE":
        return [_finding(
            field    = "Document Overall",
            signal   = "overall_verdict",
            severity = "MEDIUM",
            message  = "INCONCLUSIVE: Further manual review recommended",
            evidence = f"Forgery probability score: {pct}% — borderline result",
        )]
    else:
        return [_finding(
            field    = "Document Overall",
            signal   = "overall_verdict",
            severity = "INFO",
            message  = "Document appears authentic — no significant anomalies detected",
            evidence = f"Forgery probability score: {pct}%",
        )]


def _ela_findings(ela: dict) -> list[dict]:
    findings = []
    sp_pct   = ela.get("suspicious_pixel_pct", 0)
    rv       = ela.get("regional_variance", 0)

    # Suspicious pixel percentage
    if sp_pct > 10:
        findings.append(_finding(
            field    = "Document Image Content",
            signal   = "ela_suspicious_pixels_high",
            severity = "HIGH",
            message  = (
                "Pixel inconsistency detected — portions of this document "
                "appear to have been digitally altered"
            ),
            evidence = (
                f"{sp_pct:.1f}% of pixels show abnormal compression error "
                f"(expected < 5% for an unedited document)"
            ),
        ))
    elif sp_pct > 5:
        findings.append(_finding(
            field    = "Document Image Content",
            signal   = "ela_suspicious_pixels_medium",
            severity = "MEDIUM",
            message  = (
                "Minor pixel inconsistency detected — possible light editing "
                "or re-saving of the image"
            ),
            evidence = (
                f"{sp_pct:.1f}% of pixels show elevated compression error "
                f"(moderate range: 5–10%)"
            ),
        ))

    # Regional variance — spatial inconsistency
    if rv > 0.40:
        findings.append(_finding(
            field    = "Document Regions",
            signal   = "ela_regional_variance_high",
            severity = "HIGH",
            message  = (
                "Uneven compression detected across document regions — "
                "one or more areas may have been replaced or pasted in"
            ),
            evidence = (
                f"Regional ELA variance coefficient: {rv:.3f} "
                f"(values above 0.4 indicate splicing)"
            ),
        ))
    elif rv > 0.20:
        findings.append(_finding(
            field    = "Document Regions",
            signal   = "ela_regional_variance_medium",
            severity = "LOW",
            message  = (
                "Slightly uneven compression across document — may indicate "
                "partial re-editing or different source regions"
            ),
            evidence = f"Regional ELA variance coefficient: {rv:.3f}",
        ))

    return findings


def _exif_findings(exif: dict) -> list[dict]:
    findings = []
    raw      = exif.get("raw", {})
    flags    = exif.get("flags", [])

    software   = str(raw.get("Software", "")).strip()
    dt_orig    = raw.get("DateTimeOriginal")
    dt_mod     = raw.get("DateTime")
    make       = raw.get("Make")
    model      = raw.get("Model")

    # Software field reveals editor
    _EDITING_KEYWORDS = [
        "photoshop", "gimp", "lightroom", "affinity", "pixelmator",
        "canva", "capture one", "darktable", "rawtherapee", "snapseed",
        "vsco", "stable diffusion", "midjourney", "dall-e", "firefly",
        "adobe", "paint.net", "preview",
    ]
    if software:
        lower = software.lower()
        for kw in _EDITING_KEYWORDS:
            if kw in lower:
                findings.append(_finding(
                    field    = "Document Metadata — Software",
                    signal   = "exif_editing_software",
                    severity = "HIGH",
                    message  = (
                        f"Document was processed by editing software "
                        f"after original creation"
                    ),
                    evidence = f"EXIF Software field: '{software}'",
                ))
                break

    # DateTime mismatch
    if dt_orig and dt_mod:
        try:
            from datetime import datetime
            fmt = "%Y:%m:%d %H:%M:%S"
            t_mod = datetime.strptime(str(dt_mod),  fmt)
            t_ori = datetime.strptime(str(dt_orig), fmt)
            delta = abs((t_mod - t_ori).total_seconds())
            if delta > 60:
                h = int(delta // 3600)
                m = int((delta % 3600) // 60)
                findings.append(_finding(
                    field    = "Document Metadata — Date",
                    signal   = "exif_datetime_mismatch",
                    severity = "HIGH",
                    message  = (
                        "Metadata creation date does not match document date — "
                        "file was modified after original capture"
                    ),
                    evidence = (
                        f"Original: {dt_orig} | Last saved: {dt_mod} "
                        f"| Difference: {h}h {m}m"
                    ),
                ))
        except ValueError:
            pass

    # No camera make/model
    if not make and not model:
        findings.append(_finding(
            field    = "Document Metadata — Camera",
            signal   = "exif_no_camera",
            severity = "MEDIUM",
            message  = (
                "Document does not appear to originate from a camera — "
                "may be a screenshot, scanned re-print, or digital forgery"
            ),
            evidence = "EXIF Make and Model fields are both absent",
        ))

    # Parse individual raw flags for GPS and thumbnail
    for flag in flags:
        if "GPS" in flag and "present" in flag:
            findings.append(_finding(
                field    = "Document Metadata — Location",
                signal   = "exif_gps_present",
                severity = "INFO",
                message  = (
                    "GPS location data embedded in document — "
                    "verify coordinates match claimed scan location"
                ),
                evidence = flag,
            ))
        if "thumbnail" in flag.lower() or "Thumbnail" in flag:
            findings.append(_finding(
                field    = "Document Metadata — Thumbnail",
                signal   = "exif_thumbnail",
                severity = "LOW",
                message  = (
                    "Embedded thumbnail detected — if thumbnail differs "
                    "from main image, content was altered after original capture"
                ),
                evidence = flag,
            ))

    return findings


def _noise_findings(noise: dict) -> list[dict]:
    findings  = []
    susp_pct  = noise.get("suspicious_pct", 0)
    susp_n    = noise.get("suspicious_tiles", 0)
    total_n   = noise.get("total_tiles", 1)
    med_sigma = noise.get("median_noise_sigma", 0)

    if susp_pct > 15:
        findings.append(_finding(
            field    = "Document Texture / Scan Quality",
            signal   = "noise_inconsistency_high",
            severity = "HIGH",
            message  = (
                "Noise pattern inconsistency detected — multiple image regions "
                "have abnormal texture, suggesting content was spliced from "
                "different sources"
            ),
            evidence = (
                f"{susp_n} of {total_n} image tiles show irregular noise "
                f"({susp_pct:.1f}% — threshold: 15%)"
            ),
        ))
    elif susp_pct > 8:
        findings.append(_finding(
            field    = "Document Texture / Scan Quality",
            signal   = "noise_inconsistency_medium",
            severity = "MEDIUM",
            message  = (
                "Moderate noise inconsistency — some image regions show "
                "different texture characteristics"
            ),
            evidence = (
                f"{susp_n} of {total_n} image tiles show elevated noise sigma "
                f"({susp_pct:.1f}%)"
            ),
        ))

    return findings


def _fft_findings(fft: dict) -> list[dict]:
    findings = []
    hfr      = fft.get("high_freq_energy_ratio", 0)
    flags    = fft.get("spectral_flags", [])

    if hfr > 0.12:
        findings.append(_finding(
            field    = "Document Frequency Signature",
            signal   = "fft_high_freq_energy",
            severity = "MEDIUM",
            message  = (
                "Compression artifacts detected — the document's frequency "
                "signature is inconsistent with a genuine scanned document"
            ),
            evidence = (
                f"High-frequency energy ratio: {hfr:.3f} "
                f"(natural documents typically < 0.08)"
            ),
        ))
    elif hfr > 0.08:
        findings.append(_finding(
            field    = "Document Frequency Signature",
            signal   = "fft_elevated_freq",
            severity = "LOW",
            message  = (
                "Slightly elevated high-frequency energy — "
                "may indicate image processing or upscaling"
            ),
            evidence = f"High-frequency energy ratio: {hfr:.3f}",
        ))

    for flag in flags:
        if "Horizontal spectral band" in flag:
            findings.append(_finding(
                field    = "Document Frequency Signature",
                signal   = "fft_horizontal_band",
                severity = "LOW",
                message  = (
                    "Periodic row-level artifacts detected in frequency domain — "
                    "may indicate AI generation or systematic image processing"
                ),
                evidence = "Horizontal spectral band exceeds 2× global mean",
            ))
        elif "Vertical spectral band" in flag:
            findings.append(_finding(
                field    = "Document Frequency Signature",
                signal   = "fft_vertical_band",
                severity = "LOW",
                message  = (
                    "Periodic column-level artifacts detected in frequency domain"
                ),
                evidence = "Vertical spectral band exceeds 2× global mean",
            ))

    return findings


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _finding(
    field: str,
    signal: str,
    severity: str,
    message: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "field":    field,
        "signal":   signal,
        "severity": severity,
        "message":  message,
        "evidence": evidence,
    }

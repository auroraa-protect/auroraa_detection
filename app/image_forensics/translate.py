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
    "action":   str  — recommended next step for the fraud manager
  }

Public API
──────────
  translate(result)         → List[finding dict]   (sorted by severity)
  translate_summary(result) → summary dict          (counts, verdict, recommendation)

Usage
─────
  from image_forensics.translate import translate, translate_summary
  findings = translate(analyze_result)
  summary  = translate_summary(analyze_result)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ── Severity ordering (for sorting) ──────────────────────────────────────────
_SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}

# ── AI-generation / editing software keywords ────────────────────────────────
_AI_GEN_KEYWORDS = {
    "stable diffusion", "midjourney", "dall-e", "firefly",
    "imagen", "bing image creator", "adobe ai", "generative fill",
    "runway", "leonardo ai", "playground ai", "ideogram",
}
_EDITING_KEYWORDS = {
    "photoshop", "gimp", "lightroom", "affinity", "pixelmator",
    "canva", "capture one", "darktable", "rawtherapee", "snapseed",
    "vsco", "adobe", "paint.net", "preview",
}

# ── Recommended actions (keyed by signal) ────────────────────────────────────
_ACTIONS: dict[str, str] = {
    # Verdict
    "overall_verdict_critical":     "Reject this document and request the customer present the original physical document for in-person verification.",
    "overall_verdict_inconclusive": "Escalate to a senior fraud analyst and request a physical document plus one additional government-issued ID.",
    "overall_verdict_authentic":    "Proceed with the standard KYC workflow and retain this report in the audit file.",
    # ELA
    "ela_suspicious_pixels_high":   "Request the original uncompressed scan or physical document from the customer before proceeding.",
    "ela_suspicious_pixels_medium": "Flag the application for secondary review and ask the customer to resubmit a cleaner scan.",
    "ela_mean_ela_high":            "Reject or hold the document and request the customer provide the original unedited file.",
    "ela_regional_variance_high":   "Escalate to a senior analyst immediately and highlight the spatially inconsistent regions.",
    "ela_regional_variance_medium": "Note this finding in the case file and consider requesting an alternate proof document.",
    # EXIF
    "exif_ai_generation":           "Reject this document immediately — do not accept AI-generated images as valid identity proof.",
    "exif_editing_software":        "Request the original unedited file directly from the issuing authority and verify authenticity.",
    "exif_datetime_mismatch":       "Ask the customer to explain in writing why the document was modified after its original creation date.",
    "exif_no_camera":               "Request an alternate government-issued document as this image does not appear to be a direct camera scan.",
    "exif_many_fields_missing":     "Treat as high-risk and request an alternate document from a recognised issuing authority.",
    "exif_stripped":                "Reject and request the original file — images stripped of all metadata are treated as unverifiable.",
    "exif_risk_high":               "Cross-reference the document details with the issuing authority's database before approving.",
    "exif_gps_present":             "Verify the embedded GPS coordinates against the customer's declared address in the application.",
    "exif_thumbnail":               "Ask a senior analyst to visually compare the embedded thumbnail with the main document image.",
    # Noise
    "noise_inconsistency_high":     "Request the physical document immediately — texture inconsistency strongly suggests content splicing.",
    "noise_inconsistency_medium":   "Flag for secondary review and note the texture variance in the case file.",
    "noise_very_low_sigma":         "Escalate to a senior analyst — abnormally smooth texture is a strong indicator of a synthetic document.",
    # FFT
    "fft_high_freq_energy":         "Request the original scan file from the customer and verify it against the issuing authority.",
    "fft_elevated_freq":            "Note this finding in the case file; no immediate action is required unless other flags are present.",
    "fft_horizontal_band":          "Note this finding and consider submitting the document to an AI-detection tool for confirmation.",
    "fft_vertical_band":            "Note this finding in the case file and monitor for corroborating anomalies.",
    "fft_cruciform_pattern":        "Reject this document immediately — the detected pattern is a known signature of AI-generated images.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def translate(result: dict[str, Any]) -> list[dict[str, str]]:
    """
    Translate an analyze() result dict into a list of plain-language findings.

    Parameters
    ----------
    result : the dict returned by image_forensics.analyzer.analyze()

    Returns
    -------
    List of finding dicts, sorted by severity (most critical first).
    Each dict has keys: field, signal, severity, message, evidence, action.
    """
    findings: list[dict[str, str]] = []
    seen_signals: set[str] = set()

    ela_stats   = result.get("ela",    {}) or {}
    exif_result = result.get("exif",   {}) or {}
    noise_stats = result.get("noise",  {}) or {}
    fft_stats   = result.get("fft",    {}) or {}
    verdict     = result.get("verdict", {}) or {}

    # ── 1. Overall verdict ────────────────────────────────────────────────────
    verdict_findings = _verdict_findings(verdict)

    # ── 2. ELA (pixel manipulation) ───────────────────────────────────────────
    findings += _ela_findings(ela_stats)

    # ── 3. EXIF (metadata) ────────────────────────────────────────────────────
    findings += _exif_findings(exif_result)

    # ── 4. Noise consistency ──────────────────────────────────────────────────
    findings += _noise_findings(noise_stats)

    # ── 5. Frequency domain ───────────────────────────────────────────────────
    findings += _fft_findings(fft_stats)

    # Deduplicate by signal, preserving first (highest-priority) occurrence
    deduped: list[dict[str, str]] = []
    for f in findings:
        if f["signal"] not in seen_signals:
            seen_signals.add(f["signal"])
            deduped.append(f)

    # Sort the remaining findings by severity
    deduped.sort(key=lambda f: _SEVERITY_RANK.get(f["severity"], 99))

    # Construct final list: verdict first, then up to 7 other findings
    final_findings = verdict_findings + deduped
    return final_findings[:8]


def translate_summary(result: dict[str, Any]) -> dict[str, Any]:
    """
    Return a concise top-level summary for dashboard / API consumers.

    Keys
    ----
    verdict          : str  — "LIKELY AUTHENTIC" | "INCONCLUSIVE" | "LIKELY MANIPULATED"
    forgery_score    : float — [0.0, 1.0]
    color            : str  — "green" | "amber" | "red"
    finding_counts   : dict — count per severity level
    critical_signals : list — signal IDs of CRITICAL findings
    high_signals     : list — signal IDs of HIGH findings
    overall_recommendation : str — one sentence action for fraud manager
    generated_at     : str  — ISO-8601 timestamp
    """
    findings = translate(result)
    verdict  = result.get("verdict", {}) or {}

    counts: dict[str, int] = {s: 0 for s in _SEVERITY_RANK}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1

    critical_signals = [f["signal"] for f in findings if f["severity"] == "CRITICAL"]
    high_signals     = [f["signal"] for f in findings if f["severity"] == "HIGH"]

    label = verdict.get("verdict", "UNKNOWN")
    if label == "LIKELY MANIPULATED":
        rec = "Document shows strong manipulation indicators — reject and request physical verification."
    elif label == "INCONCLUSIVE":
        rec = "Result is borderline — escalate to senior analyst and request alternate proof."
    else:
        rec = "Document appears authentic — proceed with standard KYC workflow."

    return {
        "verdict":                verdict.get("verdict", "UNKNOWN"),
        "forgery_score":          verdict.get("forgery_score", 0.0),
        "color":                  verdict.get("color", "green"),
        "finding_counts":         counts,
        "critical_signals":       critical_signals,
        "high_signals":           high_signals,
        "overall_recommendation": rec,
        "generated_at":           datetime.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sub-translators
# ─────────────────────────────────────────────────────────────────────────────

def _verdict_findings(verdict: dict) -> list[dict]:
    score = verdict.get("forgery_score", 0)
    label = verdict.get("verdict", "UNKNOWN")
    pct   = round(score * 100, 1)

    if label == "LIKELY MANIPULATED":
        return [_finding(
            field    = "Document Overall",
            signal   = "overall_verdict_critical",
            severity = "CRITICAL",
            message  = "This document shows strong signs of having been digitally altered and should not be accepted.",
            evidence = f"Overall forgery score: {pct}% (rejection threshold: ≥55%)",
        )]
    elif label == "INCONCLUSIVE":
        return [_finding(
            field    = "Document Overall",
            signal   = "overall_verdict_inconclusive",
            severity = "MEDIUM",
            message  = "The result is borderline — the document could not be confirmed as authentic or manipulated without further review.",
            evidence = f"Overall forgery score: {pct}% (inconclusive range: 30–55%)",
        )]
    # LIKELY AUTHENTIC — no anomaly, emit nothing
    return []


def _ela_findings(ela: dict) -> list[dict]:
    findings = []
    sp_pct   = ela.get("suspicious_pixel_pct", 0) or 0
    rv       = ela.get("regional_variance",    0) or 0
    mean_ela = ela.get("mean_ela",             0) or 0

    # ── Overall image processing level ───────────────────────────────────────
    if mean_ela > 18:
        findings.append(_finding(
            field    = "Document Image Content",
            signal   = "ela_mean_ela_high",
            severity = "HIGH",
            message  = (
                "This document has been saved or processed multiple times — "
                "a genuine scan submitted directly from a device does not show this pattern"
            ),
            evidence = (
                f"Re-compression error level: {mean_ela:.2f} "
                f"(expected for unedited document: < 8)"
            ),
        ))

    # ── Suspicious pixel percentage ───────────────────────────────────────────
    if sp_pct > 10:
        findings.append(_finding(
            field    = "Document Image Content",
            signal   = "ela_suspicious_pixels_high",
            severity = "HIGH",
            message  = (
                "A large portion of this document's pixels look different from the rest — "
                "this is a strong sign that parts of the image have been digitally replaced or altered"
            ),
            evidence = (
                f"{sp_pct:.1f}% of image area shows abnormal re-save inconsistency "
                f"(expected < 5% for an untouched document)"
            ),
        ))
    elif sp_pct > 5:
        findings.append(_finding(
            field    = "Document Image Content",
            signal   = "ela_suspicious_pixels_medium",
            severity = "MEDIUM",
            message  = (
                "Some parts of this document look inconsistent with the rest — "
                "possible light editing or repeated re-saving before submission"
            ),
            evidence = (
                f"{sp_pct:.1f}% of image area shows elevated re-save inconsistency "
                f"(moderate range: 5–10%)"
            ),
        ))

    # ── Spatial inconsistency (splicing) ─────────────────────────────────────
    if rv > 0.40:
        findings.append(_finding(
            field    = "Document Regions",
            signal   = "ela_regional_variance_high",
            severity = "HIGH",
            message  = (
                "Different sections of this document appear to have come from different sources — "
                "text, photo, or stamp areas may have been cut and pasted from another document"
            ),
            evidence = (
                f"Inter-region inconsistency score: {rv:.3f} "
                f"(values above 0.40 strongly indicate splicing)"
            ),
        ))
    elif rv > 0.20:
        findings.append(_finding(
            field    = "Document Regions",
            signal   = "ela_regional_variance_medium",
            severity = "LOW",
            message  = (
                "Minor unevenness detected between document sections — "
                "may indicate partial editing or the document was assembled from multiple scans"
            ),
            evidence = (
                f"Inter-region inconsistency score: {rv:.3f} "
                f"(mild range: 0.20–0.40)"
            ),
        ))

    return findings


def _exif_findings(exif: dict) -> list[dict]:
    findings = []
    raw   = exif.get("raw",        {}) or {}
    flags = exif.get("flags",      []) or []
    risk  = exif.get("risk_level", "unknown")

    software   = str(raw.get("Software", "")).strip()
    dt_orig    = raw.get("DateTimeOriginal")
    dt_mod     = raw.get("DateTime")
    make       = raw.get("Make")
    model      = raw.get("Model")

    # ── No EXIF at all (stripped / synthetic) ────────────────────────────────
    if not raw:
        findings.append(_finding(
            field    = "Document Metadata",
            signal   = "exif_stripped",
            severity = "HIGH",
            message  = (
                "No EXIF metadata found — the image may have been deliberately "
                "stripped to hide its digital provenance or origin"
            ),
            evidence = "EXIF data block is entirely absent",
        ))
        return findings  # Nothing more to check without raw data

    # ── AI-generation software (highest priority within EXIF) ────────────────
    if software:
        lower = software.lower()
        for kw in _AI_GEN_KEYWORDS:
            if kw in lower:
                findings.append(_finding(
                    field    = "Document Metadata — Software",
                    signal   = "exif_ai_generation",
                    severity = "CRITICAL",
                    message  = (
                        f"AI image-generation tool detected in metadata — "
                        f"document is likely entirely synthetic and not a real scan"
                    ),
                    evidence = f"EXIF Software field: '{software}'",
                ))
                break  # Don't double-flag as editing software too
        else:
            # Check for conventional editing software only if not AI-generated
            for kw in _EDITING_KEYWORDS:
                if kw in lower:
                    findings.append(_finding(
                        field    = "Document Metadata — Software",
                        signal   = "exif_editing_software",
                        severity = "HIGH",
                        message  = (
                            "Document was processed by editing software "
                            "after original creation"
                        ),
                        evidence = f"EXIF Software field: '{software}'",
                    ))
                    break

    # ── DateTime mismatch ────────────────────────────────────────────────────
    if dt_orig and dt_mod:
        try:
            fmt   = "%Y:%m:%d %H:%M:%S"
            t_mod = datetime.strptime(str(dt_mod),  fmt)
            t_ori = datetime.strptime(str(dt_orig), fmt)
            delta = abs((t_mod - t_ori).total_seconds())
            if delta > 60:
                h = int(delta // 3600)
                m = int((delta % 3600) // 60)
                severity = "HIGH" if delta > 3600 else "MEDIUM"
                findings.append(_finding(
                    field    = "Document Metadata — Date",
                    signal   = "exif_datetime_mismatch",
                    severity = severity,
                    message  = (
                        "Metadata creation date does not match document date — "
                        "file was modified after original capture"
                    ),
                    evidence = (
                        f"Original: {dt_orig}  |  Last saved: {dt_mod} "
                        f"|  Difference: {h}h {m}m"
                    ),
                ))
        except ValueError:
            pass

    # ── No camera Make/Model ──────────────────────────────────────────────────
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

    # ── Many camera fields missing ────────────────────────────────────────────
    _EXPECTED_CAMERA_FIELDS = {
        "Make", "Model", "ExposureTime", "FNumber",
        "ISOSpeedRatings", "DateTimeOriginal", "FocalLength",
    }
    present = set(raw.keys())
    missing = _EXPECTED_CAMERA_FIELDS - present
    if len(missing) >= 4:
        findings.append(_finding(
            field    = "Document Metadata — Camera Fields",
            signal   = "exif_many_fields_missing",
            severity = "MEDIUM",
            message  = (
                f"{len(missing)} expected camera metadata fields are absent — "
                f"image is likely synthetic, a screenshot, or heavily stripped"
            ),
            evidence = f"Missing fields: {', '.join(sorted(missing))}",
        ))

    # ── EXIF risk_level (aggregate high-risk signal from exif.py) ────────────
    if risk == "high":
        findings.append(_finding(
            field    = "Document Metadata — Risk",
            signal   = "exif_risk_high",
            severity = "HIGH",
            message  = (
                "Multiple metadata red flags detected — EXIF analysis "
                "classifies this document as high forensic risk"
            ),
            evidence = (
                f"EXIF risk_level: HIGH "
                f"(≥2 warning signals found in metadata)"
            ),
        ))

    # ── GPS and Thumbnail info from flags ─────────────────────────────────────
    for flag in flags:
        flag_lower = flag.lower()
        if "gps" in flag_lower and "present" in flag_lower:
            findings.append(_finding(
                field    = "Document Metadata — Location",
                signal   = "exif_gps_present",
                severity = "INFO",
                message  = (
                    "GPS location data embedded in document — "
                    "verify coordinates match the customer's claimed scan location"
                ),
                evidence = _clean_flag(flag),
            ))
        if "thumbnail" in flag_lower:
            findings.append(_finding(
                field    = "Document Metadata — Thumbnail",
                signal   = "exif_thumbnail",
                severity = "LOW",
                message  = (
                    "Embedded thumbnail detected — if the thumbnail differs "
                    "from the main image, content was altered after original capture"
                ),
                evidence = _clean_flag(flag),
            ))

    return findings


def _noise_findings(noise: dict) -> list[dict]:
    findings  = []
    susp_pct  = noise.get("suspicious_pct",      0) or 0
    susp_n    = noise.get("suspicious_tiles",     0) or 0
    total_n   = noise.get("total_tiles",          1) or 1
    med_sigma = noise.get("median_noise_sigma",   0) or 0

    # ── Very low grain — potential AI/synthetic origin ────────────────────────
    if 0 < med_sigma < 0.002:
        findings.append(_finding(
            field    = "Document Texture",
            signal   = "noise_very_low_sigma",
            severity = "HIGH",
            message  = (
                "This document is unnaturally smooth — genuine camera scans always contain "
                "a small amount of grain; the absence of any grain suggests a computer-generated image"
            ),
            evidence = (
                f"Measured grain level: {med_sigma:.6f} "
                f"(genuine scans are typically ≥ 0.003)"
            ),
        ))

    # ── High texture inconsistency (splicing) ─────────────────────────────────
    if susp_pct > 15:
        findings.append(_finding(
            field    = "Document Texture / Scan Quality",
            signal   = "noise_inconsistency_high",
            severity = "HIGH",
            message  = (
                "The surface texture of this document is not uniform — "
                "several sections look like they were taken from a different image and pasted in"
            ),
            evidence = (
                f"{susp_n} of {total_n} image sections show inconsistent texture "
                f"({susp_pct:.1f}% — threshold: 15%)"
            ),
        ))
    elif susp_pct > 8:
        findings.append(_finding(
            field    = "Document Texture / Scan Quality",
            signal   = "noise_inconsistency_medium",
            severity = "MEDIUM",
            message  = (
                "Some sections of this document have a noticeably different surface texture — "
                "this warrants a closer manual inspection"
            ),
            evidence = (
                f"{susp_n} of {total_n} image sections show elevated texture variation "
                f"({susp_pct:.1f}%)"
            ),
        ))

    return findings


def _fft_findings(fft: dict) -> list[dict]:
    findings = []
    hfr      = fft.get("high_freq_energy_ratio", 0) or 0
    flags    = fft.get("spectral_flags",          []) or []

    # ── Structural artifact level ─────────────────────────────────────────────
    if hfr > 0.12:
        findings.append(_finding(
            field    = "Document Structural Integrity",
            signal   = "fft_high_freq_energy",
            severity = "MEDIUM",
            message  = (
                "This document contains an unusual level of fine-detail noise — "
                "genuine scanned documents do not show this pattern, suggesting heavy processing or manipulation"
            ),
            evidence = (
                f"Structural artifact ratio: {hfr:.4f} "
                f"(genuine scans: < 0.08; manipulated/AI documents: often > 0.12)"
            ),
        ))
    elif hfr > 0.08:
        findings.append(_finding(
            field    = "Document Structural Integrity",
            signal   = "fft_elevated_freq",
            severity = "LOW",
            message  = (
                "A mildly elevated level of fine-detail noise was detected — "
                "this may result from image upscaling or heavy compression"
            ),
            evidence = (
                f"Structural artifact ratio: {hfr:.4f} "
                f"(mild elevation range: 0.08–0.12)"
            ),
        ))

    # ── Repeating pattern artifacts ───────────────────────────────────────────
    has_horiz = any("Horizontal spectral band" in f for f in flags)
    has_vert  = any("Vertical spectral band"   in f for f in flags)

    if has_horiz and has_vert:
        findings.append(_finding(
            field    = "Document Structural Integrity",
            signal   = "fft_cruciform_pattern",
            severity = "HIGH",
            message  = (
                "A repeating cross-shaped pattern was detected inside this document's image data — "
                "this is a known fingerprint left by AI image-generation software"
            ),
            evidence = (
                "Repeating row and column artifacts both exceed twice the image average — "
                "consistent with GAN or diffusion model generation"
            ),
        ))
    else:
        if has_horiz:
            findings.append(_finding(
                field    = "Document Structural Integrity",
                signal   = "fft_horizontal_band",
                severity = "LOW",
                message  = (
                    "A repeating horizontal stripe pattern was detected in the image data — "
                    "this can be a sign of AI-generated content or systematic image processing"
                ),
                evidence = "Horizontal repeating artifact exceeds twice the image average",
            ))
        if has_vert:
            findings.append(_finding(
                field    = "Document Structural Integrity",
                signal   = "fft_vertical_band",
                severity = "LOW",
                message  = (
                    "A repeating vertical stripe pattern was detected in the image data — "
                    "may indicate systematic upscaling or AI generation"
                ),
                evidence = "Vertical repeating artifact exceeds twice the image average",
            ))

    return findings


def _component_score_findings(verdict: dict) -> list[dict]:
    """Removed — component breakdown is developer context, not an officer finding."""
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _finding(
    field: str,
    signal: str,
    severity: str,
    message: str,
    evidence: str,
) -> dict[str, str]:
    """Build a standardised finding dict with an auto-resolved action."""
    action = _ACTIONS.get(signal, "Document this finding in the case notes for audit.")
    return {
        "field":    field,
        "signal":   signal,
        "severity": severity,
        "message":  message,
        "evidence": evidence,
        "action":   action,
    }


def _clean_flag(flag: str) -> str:
    """Strip leading emoji/whitespace from an exif.py flag string."""
    return flag.lstrip("⚠️ ℹ️ 🚨 ").strip()

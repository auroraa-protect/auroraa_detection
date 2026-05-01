"""
exif.py — EXIF Metadata Extraction & Forensic Analysis
────────────────────────────────────────────────────────
EXIF (Exchangeable Image File Format) data is embedded by cameras and
editing software. It records camera model, lens, GPS, date/time, and the
software used to edit or save the file.

Forensic signals to look for:
  - Software field contains editing tool names (Photoshop, GIMP, etc.)
  - DateTime vs DateTimeOriginal mismatch (file was re-saved after editing)
  - Missing fields that should always be present for a real camera shot
  - GPS coordinates inconsistent with claimed location or image content
  - Thumbnail embedded inside EXIF doesn't match visible image
    (a classic tell: someone pasted new pixels but kept the old thumbnail)
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from typing import Any


# ── Software names that suggest post-processing ──────────────────────────
_EDITING_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "affinity", "pixelmator",
    "canva", "capture one", "darktable", "rawtherapee",
    "snapseed", "vsco", "stable diffusion", "midjourney",
    "dall-e", "firefly", "adobe", "paint.net", "preview",
]

# Fields that a genuine camera always writes
_EXPECTED_CAMERA_FIELDS = {
    "Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "DateTimeOriginal", "FocalLength",
}


def extract_exif(image_path: str) -> dict:
    """
    Extract and forensically annotate EXIF data from an image.

    Returns a dict with:
      raw        : all decoded EXIF tag → value pairs
      gps        : parsed GPS dict (if present)
      flags      : list of forensic warning strings
      risk_level : "low" | "medium" | "high"
      summary    : human-readable paragraph
    """
    try:
        img = Image.open(image_path)
        raw_exif = img._getexif()  # returns {tag_id: value} or None
    except Exception as e:
        return _empty_result(f"Could not open image: {e}")

    if raw_exif is None:
        return _empty_result("No EXIF data found — image may be stripped or synthetic.")

    # ── Decode tag IDs → human names ─────────────────────────
    decoded: dict[str, Any] = {}
    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, str(tag_id))
        # Skip raw binary blobs (MakerNote, UserComment bytes, thumbnail bytes)
        if isinstance(value, bytes) and len(value) > 64:
            decoded[tag_name] = f"<binary {len(value)} bytes>"
        else:
            decoded[tag_name] = value

    # ── Extract GPS ───────────────────────────────────────────
    gps_info = _parse_gps(raw_exif)

    # ── Forensic flag checks ──────────────────────────────────
    flags = []

    # 1. Editing software
    software = str(decoded.get("Software", "")).strip()
    if software:
        lower_sw = software.lower()
        for kw in _EDITING_KEYWORDS:
            if kw in lower_sw:
                flags.append(
                    f"⚠️  Software field reveals editing tool: '{software}'. "
                    f"Image was processed after capture."
                )
                break

    # 2. DateTime vs DateTimeOriginal mismatch
    dt_modified  = decoded.get("DateTime")
    dt_original  = decoded.get("DateTimeOriginal")
    dt_digitized = decoded.get("DateTimeDigitized")

    if dt_modified and dt_original:
        try:
            fmt = "%Y:%m:%d %H:%M:%S"
            t_mod = datetime.strptime(str(dt_modified),  fmt)
            t_ori = datetime.strptime(str(dt_original),  fmt)
            delta_seconds = abs((t_mod - t_ori).total_seconds())
            if delta_seconds > 60:
                flags.append(
                    f"⚠️  Timestamp mismatch: original={dt_original}, "
                    f"last-saved={dt_modified} "
                    f"(Δ {int(delta_seconds // 3600)}h "
                    f"{int((delta_seconds % 3600) // 60)}m). "
                    f"File was re-saved after original capture."
                )
        except ValueError:
            pass

    # 3. Missing expected camera fields
    present_fields = set(decoded.keys())
    missing = _EXPECTED_CAMERA_FIELDS - present_fields
    if len(missing) >= 4:
        flags.append(
            f"ℹ️  {len(missing)} expected camera fields missing "
            f"({', '.join(sorted(missing))}). "
            f"Could indicate a synthetic, screenshot, or stripped image."
        )
    elif missing:
        flags.append(
            f"ℹ️  Some camera fields absent: {', '.join(sorted(missing))}."
        )

    # 4. No Make/Model at all (strong synthetic signal)
    if not decoded.get("Make") and not decoded.get("Model"):
        flags.append(
            "🚨 No camera Make or Model field. "
            "Real camera photos almost always include these."
        )

    # 5. GPS present — note it (useful for cross-checking)
    if gps_info.get("latitude") is not None:
        flags.append(
            f"ℹ️  GPS coordinates present: "
            f"{gps_info['latitude']:.6f}°, {gps_info['longitude']:.6f}°. "
            f"Verify against image content and claimed location."
        )

    # 6. Thumbnail in EXIF
    if "JPEGThumbnail" in decoded or any("Thumbnail" in k for k in decoded):
        flags.append(
            "ℹ️  Embedded thumbnail detected. "
            "If thumbnail differs from main image, content was altered after shooting."
        )

    # ── Risk scoring ──────────────────────────────────────────
    high_signals = sum(1 for f in flags if "🚨" in f or "⚠️" in f)
    if high_signals >= 2:
        risk = "high"
    elif high_signals == 1:
        risk = "medium"
    else:
        risk = "low"

    # ── Summary ───────────────────────────────────────────────
    make   = decoded.get("Make",  "Unknown make")
    model  = decoded.get("Model", "Unknown model")
    summary = (
        f"Image claims to be from {make} {model}. "
        f"Software: {software or 'not recorded'}. "
        f"Original capture: {dt_original or 'not recorded'}. "
        f"Last saved: {dt_modified or 'not recorded'}. "
        f"GPS: {'present' if gps_info.get('latitude') else 'absent'}. "
        f"Forensic risk: {risk.upper()}."
    )

    return {
        "raw":        decoded,
        "gps":        gps_info,
        "flags":      flags,
        "risk_level": risk,
        "summary":    summary,
    }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _parse_gps(raw_exif: dict) -> dict:
    """Extract and decode the GPS sub-IFD (tag 34853)."""
    GPS_TAG_ID = 34853
    gps_raw = raw_exif.get(GPS_TAG_ID)
    if not gps_raw:
        return {}

    gps = {}
    for tag_id, value in gps_raw.items():
        tag_name = GPSTAGS.get(tag_id, str(tag_id))
        gps[tag_name] = value

    result = {}
    try:
        lat  = _dms_to_decimal(gps.get("GPSLatitude"),  gps.get("GPSLatitudeRef",  "N"))
        lon  = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"))
        alt_raw = gps.get("GPSAltitude")
        alt  = float(alt_raw) if alt_raw is not None else None
        result = {"latitude": lat, "longitude": lon, "altitude_m": alt, "raw": gps}
    except Exception:
        result = {"raw": gps}
    return result


def _dms_to_decimal(dms, ref: str) -> float | None:
    """Convert degrees/minutes/seconds tuple to decimal degrees."""
    if dms is None:
        return None
    d, m, s = [float(x) for x in dms]
    decimal = d + m / 60 + s / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return round(decimal, 8)


def _empty_result(reason: str) -> dict:
    return {
        "raw":        {},
        "gps":        {},
        "flags":      [f"ℹ️  {reason}"],
        "risk_level": "unknown",
        "summary":    reason,
    }
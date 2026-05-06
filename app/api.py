import base64
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import File, HTTPException, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from app.forensics.analyzer import analyze

detection_router = APIRouter(tags=["Vault"])

# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@detection_router.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    """
    Analyze an uploaded document image for signs of manipulation.

    Accepts: JPEG, PNG, WebP, BMP, TIFF
    Returns: JSON forensic report
    """
    # ── Validate file type ────────────────────────────────────
    ALLOWED_TYPES = {
        "image/jpeg", "image/png", "image/webp",
        "image/bmp", "image/tiff",
    }
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Accepted: JPEG, PNG, WebP, BMP, TIFF.",
        )

    # ── Save upload to temporary file ─────────────────────────
    ext = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        await file.close()

        # ── Run forensic pipeline ─────────────────────────────────
        try:
            result = analyze(
                image_path=tmp_path,
                verbose=False,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {exc}",
            )
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)

    # ── Build response ────────────────────────────────────────
    verdict       = result["verdict"]

    return JSONResponse({
        # ── Plain-language layer ──────────────────────────────
        "findings": result.get("findings", []),

        # ── Verdict ──────────────────────────────────────────
        "verdict":       verdict["verdict"],
        "forgery_score": verdict["forgery_score"],
        "verdict_color": verdict["color"],

        # ── Raw stats (for advanced / developer users) ────────
        "raw": {
            "ela":   result["ela"],
            "exif":  {
                "risk_level": result["exif"]["risk_level"],
                "flags":      result["exif"]["flags"],
                "summary":    result["exif"]["summary"],
                "key_fields": {
                    k: str(v)
                    for k, v in result["exif"].get("raw", {}).items()
                    if k in {"Make", "Model", "Software",
                             "DateTime", "DateTimeOriginal"}
                },
            },
            "noise": result["noise"],
            "fft":   {
                k: v for k, v in result["fft"].items()
                if k != "fft_shape"  # not serialisable
            },
        },
    })

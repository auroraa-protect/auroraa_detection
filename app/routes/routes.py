import base64
import os
import shutil
import uuid
from pathlib import Path

from fastapi import File, HTTPException, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from app.image_forensics.analyzer import analyze

# ── Directories ──────────────────────────────────────────────
UPLOAD_DIR  = Path("uploads")
REPORT_DIR  = Path("reports")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

detection_router = APIRouter()

# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@detection_router.get("/health")
def health():
    return {"status": "ok"}


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

    # ── Save upload with unique name ─────────────────────────
    ext       = Path(file.filename).suffix or ".jpg"
    unique_id = uuid.uuid4().hex[:12]
    save_name = f"{unique_id}{ext}"
    save_path = UPLOAD_DIR / save_name

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    # ── Run forensic pipeline ─────────────────────────────────
    try:
        result = analyze(
            image_path=str(save_path),
            output_dir=str(REPORT_DIR),
            verbose=False,
        )
    except Exception as exc:
        # Clean up upload on error
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        )

    # ── Base64-encode the PNG report ──────────────────────────
    # png_path = Path(result["report_png"])
    # try:
    #     with open(png_path, "rb") as img_f:
    #         report_b64 = base64.b64encode(img_f.read()).decode("utf-8")
    # except FileNotFoundError:
    #     report_b64 = None

    # ── Build response ────────────────────────────────────────
    json_filename = Path(result["report_json"]).name
    verdict       = result["verdict"]

    return JSONResponse({
        # ── Plain-language layer ──────────────────────────────
        "findings": result.get("findings", []),

        # ── Verdict ──────────────────────────────────────────
        "verdict":       verdict["verdict"],
        "forgery_score": verdict["forgery_score"],
        "verdict_color": verdict["color"],

        # ── Report assets ─────────────────────────────────────
        # "report_png_b64":  report_b64,
        "report_json_url": f"/reports/{json_filename}",

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

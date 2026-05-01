"""
app.py — Auroraa Vault FastAPI Server
──────────────────────────────────────
Exposes the image forensics engine as a REST API for the existing
Auroraa Vault frontend.

Endpoints
─────────
  POST /api/analyze
        Accepts a multipart image upload.
        Returns JSON with:
          • findings        — plain-language list (from translate.py)
          • verdict         — LIKELY AUTHENTIC / INCONCLUSIVE / LIKELY MANIPULATED
          • forgery_score   — float 0.0–1.0
          • report_png_b64  — base64-encoded PNG report (full forensic chart)
          • report_json_url — relative URL to download the JSON report
          • ela             — raw ELA stats
          • exif            — raw EXIF result
          • noise           — raw noise stats
          • fft             — raw FFT stats

  GET  /health
        Health check — returns { "status": "ok" }.

  GET  /reports/{filename}
        Serves a generated report file (PNG or JSON).

Running
───────
  # Local dev
  python app.py

  # Production (Render.com / Railway)
  uvicorn app.main:app --host 0.0.0.0 --port $PORT

CORS
────
  All origins are allowed so the Vault frontend (any host) can call this API.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes.routes import detection_router

# ── Directories ──────────────────────────────────────────────
UPLOAD_DIR  = Path("uploads")
REPORT_DIR  = Path("reports")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Auroraa Vault — Document Forensics API",
    description="Image manipulation detection via ELA, EXIF, Noise & FFT analysis.",
    version="1.0.0",
)

# ── CORS (allow any origin so the Vault frontend can call freely) ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(detection_router, tags=["detection"])

# ── Serve generated reports as static files ───────────────────
app.mount("/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")

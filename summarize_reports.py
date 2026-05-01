"""
summarize_reports.py — Master report aggregator
────────────────────────────────────────────────
Reads every per-image forensic JSON report in  app/reports/
and writes a single  app/reports/master_summary.json  that contains:

  • One concise row per image (file name, verdict, score, risk, key flags)
  • Aggregate statistics across all scanned images
  • Verdict distribution breakdown
  • Top recurring EXIF flags

Run from the project root:
    python summarize_reports.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPORT_DIR  = Path("app") / "reports"
SKIP_FILES  = {"audit_report.json", "master_summary.json"}
OUT_PATH    = REPORT_DIR / "master_summary.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_reports() -> list[dict]:
    """Load every per-image JSON report from the reports directory."""
    reports = []
    for path in sorted(REPORT_DIR.glob("*.json")):
        if path.name in SKIP_FILES:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data["_source_file"] = path.name
            reports.append(data)
        except Exception as exc:
            print(f"  ⚠  Skipping {path.name}: {exc}")
    return reports


def _slim_row(r: dict) -> dict:
    """Extract the key fields for one image into a flat summary row."""
    verdict_block = r.get("verdict", {}) or {}
    exif_block    = r.get("exif",    {}) or {}
    ela_block     = r.get("ela",     {}) or {}
    noise_block   = r.get("noise",   {}) or {}
    fft_block     = r.get("fft",     {}) or {}
    comp_scores   = verdict_block.get("component_scores", {}) or {}

    return {
        "file":               r.get("image_path") or r.get("_source_file", "unknown"),
        "report_file":        r.get("_source_file"),
        "analysed_at":        r.get("analysed_at"),
        # ── Verdict ──────────────────────────────────────────────────────────
        "verdict":            verdict_block.get("verdict"),
        "forgery_score":      verdict_block.get("forgery_score"),
        "color":              verdict_block.get("color"),
        # ── EXIF ─────────────────────────────────────────────────────────────
        "exif_risk":          exif_block.get("risk_level"),
        "exif_flags":         exif_block.get("flags", []),
        "exif_software":      (exif_block.get("raw") or {}).get("Software"),
        # ── ELA ──────────────────────────────────────────────────────────────
        "ela_suspicious_pct": ela_block.get("suspicious_pixel_pct"),
        "ela_regional_var":   ela_block.get("regional_variance"),
        # ── Noise ────────────────────────────────────────────────────────────
        "noise_suspicious_pct": noise_block.get("suspicious_pct"),
        # ── FFT ──────────────────────────────────────────────────────────────
        "fft_high_freq_ratio":  fft_block.get("high_freq_energy_ratio"),
        # ── Component scores ─────────────────────────────────────────────────
        "component_scores":   comp_scores,
    }


def _aggregate(rows: list[dict]) -> dict:
    """Compute aggregate statistics across all image rows."""
    total = len(rows)
    if total == 0:
        return {}

    verdict_counts: dict[str, int] = {}
    risk_counts:    dict[str, int] = {}
    flag_counts:    dict[str, int] = {}
    scores = []

    for row in rows:
        v = row.get("verdict") or "UNKNOWN"
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

        risk = row.get("exif_risk") or "unknown"
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

        for flag in (row.get("exif_flags") or []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

        s = row.get("forgery_score")
        if s is not None:
            scores.append(float(s))

    avg_score = round(sum(scores) / len(scores), 4) if scores else None
    max_score = round(max(scores), 4) if scores else None
    min_score = round(min(scores), 4) if scores else None

    # Sort flags by frequency
    top_flags = sorted(flag_counts.items(), key=lambda x: -x[1])

    # Images with high risk
    high_risk = [
        r["file"] for r in rows
        if (r.get("forgery_score") or 0) >= 0.55
    ]

    return {
        "total_images":       total,
        "verdict_distribution": verdict_counts,
        "exif_risk_distribution": risk_counts,
        "forgery_score": {
            "average": avg_score,
            "min":     min_score,
            "max":     max_score,
        },
        "top_exif_flags": [
            {"flag": flag, "count": count}
            for flag, count in top_flags[:10]
        ],
        "high_risk_images": high_risk,
        "high_risk_count":  len(high_risk),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n  Reading reports from {REPORT_DIR} ...\n")
all_reports = _load_reports()

if not all_reports:
    print("  No per-image report JSON files found.")
    print(f"  Run  python run_audit_test.py  first to generate them.\n")
    raise SystemExit(0)

rows = [_slim_row(r) for r in all_reports]
stats = _aggregate(rows)

master = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "aggregate":    stats,
    "images":       rows,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(master, f, indent=2, default=str)

print(f"  Loaded {len(all_reports)} report(s).\n")
print(f"  Verdict distribution:")
for verdict, count in stats.get("verdict_distribution", {}).items():
    bar = "█" * count
    print(f"    {verdict:<22} {bar} ({count})")

print(f"\n  Average forgery score : {stats['forgery_score']['average']}")
print(f"  Min / Max             : {stats['forgery_score']['min']} / {stats['forgery_score']['max']}")
print(f"  High-risk images      : {stats['high_risk_count']}")

if stats.get("top_exif_flags"):
    print(f"\n  Top EXIF flags:")
    for entry in stats["top_exif_flags"][:5]:
        print(f"    {entry['flag']:<35} × {entry['count']}")

print(f"\n  Master summary saved → {OUT_PATH}\n")

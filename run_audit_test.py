"""
run_audit_test.py — End-to-end test runner for audit.py
────────────────────────────────────────────────────────
Run from the project root:

    python run_audit_test.py

What it does
────────────
1. Runs every image in  app/samples/original/   labelled  AUTHENTIC
2. Runs every image in  app/samples/tampered/   labelled  MANIPULATED
   (if that folder is empty, it auto-generates tampered copies by
    double-JPEG-compressing each original at quality=50 — a quick
    approximation for testing purposes only)
3. Calls audit_batch() with all results
4. Prints the full structured audit report to stdout as JSON

The server does NOT need to be running — this script calls the Python
API directly (no HTTP), so it works offline.
"""

import argparse
import json
import sys
import shutil
import tempfile
from pathlib import Path

# ── Make sure the project root is on sys.path ─────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from app.image_forensics.analyzer import analyze
from audit    import audit_batch


ORIGINAL_DIR = ROOT / "samples" / "original"
TAMPERED_DIR = ROOT / "samples" / "tampered"
REPORT_DIR   = ROOT / "reports"
AUDIT_OUT    = REPORT_DIR / "audit_report.json"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI args — manually control how many images to use
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Run forensic audit batch test against sample images."
)
parser.add_argument(
    "--original", "-o",
    type=int,
    default=5,
    metavar="N",
    help="Number of images to use from app/samples/original/  (default: all)",
)
parser.add_argument(
    "--tampered", "-t",
    type=int,
    default=5,
    metavar="N",
    help="Number of images to use from app/samples/tampered/  (default: all)",
)
args = parser.parse_args()


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

all_original = sorted(
    p for p in ORIGINAL_DIR.iterdir()
    if p.suffix.lower() in SUPPORTED
)
all_tampered = sorted(
    p for p in TAMPERED_DIR.iterdir()
    if p.suffix.lower() in SUPPORTED
)

# Apply manual limits from CLI flags
original_images = all_original[:args.original] if args.original else all_original
tampered_images = all_tampered[:args.tampered] if args.tampered else all_tampered

print(f"\n{'═'*60}")
print(f"  AUDIT TEST RUNNER")
print(f"{'═'*60}")
print(f"  Original (AUTHENTIC)   : {len(original_images)} / {len(all_original)} image(s)")
print(f"  Tampered (MANIPULATED) : {len(tampered_images)} / {len(all_tampered)} image(s)")
if args.original or args.tampered:
    print(f"  ↳ custom selection via --original / --tampered flags")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Auto-generate tampered copies if tampered/ is empty
# ─────────────────────────────────────────────────────────────────────────────

_tmp_dir = None

if not tampered_images:
    print("\n  ⚠️  tampered/ folder is empty.")
    print("  Auto-generating tampered proxies by double-JPEG-compressing originals...")

    try:
        from PIL import Image as _PILImage
        _has_pil = True
    except ImportError:
        _has_pil = False

    if _has_pil:
        _tmp_dir = Path(tempfile.mkdtemp(prefix="audit_tampered_"))
        for src in original_images:
            img = _PILImage.open(src).convert("RGB")
            # Pass 1: heavily compress
            tmp1 = _tmp_dir / f"pass1_{src.name}"
            img.save(tmp1, "JPEG", quality=45)
            # Pass 2: re-open and re-save — simulates re-saving after editing
            img2 = _PILImage.open(tmp1).convert("RGB")
            dst  = _tmp_dir / f"tampered_{src.stem}.jpg"
            img2.save(dst, "JPEG", quality=55)
            tampered_images.append(dst)
            print(f"    ↳ {dst.name}")
        print(f"  Generated {len(tampered_images)} tampered proxy image(s).\n")
    else:
        print("  ⚠️  Pillow not available — skipping tampered generation.")
        print("  Place real tampered images in app/samples/tampered/ to test properly.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Run analyze() on every image
# ─────────────────────────────────────────────────────────────────────────────

evaluations = []
errors       = []


def _run(path: Path, ground_truth: str) -> None:
    label = "✓" if ground_truth == "AUTHENTIC" else "✗"
    print(f"  [{label}] Analysing {path.name} ({ground_truth}) ...", end=" ", flush=True)
    try:
        result = analyze(str(path), verbose=False)
        result["ground_truth"] = ground_truth
        evaluations.append(result)
        verdict = result["verdict"]["verdict"]
        score   = result["verdict"]["forgery_score"]
        correct = (
            (ground_truth == "AUTHENTIC"   and verdict != "LIKELY MANIPULATED") or
            (ground_truth == "MANIPULATED" and verdict == "LIKELY MANIPULATED")
        )
        tag = "✅" if correct else "❌"
        print(f"{tag}  {verdict} ({score:.3f})")
    except Exception as exc:
        errors.append({"file": path.name, "error": str(exc)})
        print(f"💥  ERROR: {exc}")


print("\n  Running analysis...\n")
for img in original_images:
    _run(img, "AUTHENTIC")
for img in tampered_images:
    _run(img, "MANIPULATED")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Run audit_batch()
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print("  Running audit_batch()...")
print(f"{'─'*60}\n")

report = audit_batch(evaluations)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Print results
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


_section("SUMMARY")
s = report["summary"]
print(f"  Images tested : {s['total_images']}  "
      f"(authentic={s['authentic']}, manipulated={s['manipulated']})")
print(f"  Accuracy      : {s['accuracy']:.1%}")
print(f"  Precision     : {s['precision']}")
print(f"  Recall        : {s['recall']}")
print(f"  True  Positives : {s['true_positives']}")
print(f"  False Positives : {s['false_positives']}  ← clean docs flagged as forged")
print(f"  True  Negatives : {s['true_negatives']}")
print(f"  False Negatives : {s['false_negatives']}  ← forged docs passed as clean")


_section("SIGNAL DIAGNOSIS")
for sig, info in report["signal_diagnosis"].items():
    status = info["assessment"].upper()
    icon   = {"RELIABLE": "✅", "OVER-PERFORMING": "🔴", "UNDER-PERFORMING": "🟡", "UNRELIABLE": "⚫"}.get(status, "?")
    print(f"\n  {icon}  {sig}  (weight={info['current_weight']:.2f})")
    print(f"     Separation   : {info['separation']:+.4f}  "
          f"(auth={info['mean_score_authentic']:.3f}, manip={info['mean_score_manipulated']:.3f})")
    if info["problem"]:
        print(f"     ⚠  {info['problem']}")


_section("WEIGHT RECOMMENDATIONS")
recs = report["weight_recommendations"]
print(f"\n  {'Signal':<28} {'Current':>8} {'Proposed':>9}")
print(f"  {'-'*28} {'-'*8} {'-'*9}")
current_w = {s: d["current_weight"] for s, d in report["signal_diagnosis"].items()}
for sig, new_w in recs["proposed_weights"].items():
    cur = current_w.get(sig, "?")
    delta = new_w - cur if isinstance(cur, float) else 0
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    print(f"  {sig:<28} {cur:>8.3f} {new_w:>8.4f}  {arrow}")
print(f"\n  Note: {recs['note']}")


_section("PRIORITY FIXES")
for fix in report["priority_fixes"]:
    print(f"\n  #{fix['rank']}  [{fix['type']}]  {fix['impact']}")
    print(f"     {fix['action']}")


_section("SYSTEMIC PATTERNS")
print()
for sentence in report["systemic_patterns"].split(". "):
    if sentence.strip():
        print(f"  • {sentence.strip().rstrip('.')}.")


# ── Build consolidated report ─────────────────────────────────────────────────
# Strip non-serialisable types (numpy arrays, Path objects, etc.)
def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


# Build slim per-image records (drop heavy raw arrays from analyze() output)
def _slim_image_record(e: dict) -> dict:
    verdict_raw = e.get("verdict", {}) or {}
    exif_raw    = e.get("exif",    {}) or {}
    return {
        "file":            e.get("image_path", "unknown"),
        "ground_truth":    e.get("ground_truth", "UNKNOWN"),
        "correct":         (
            (e["ground_truth"] == "AUTHENTIC"   and verdict_raw.get("verdict") != "LIKELY MANIPULATED") or
            (e["ground_truth"] == "MANIPULATED" and verdict_raw.get("verdict") == "LIKELY MANIPULATED")
        ),
        "findings":        e.get("findings", []),
        "verdict":         verdict_raw.get("verdict"),
        "forgery_score":   verdict_raw.get("forgery_score"),
        "verdict_color":   verdict_raw.get("color"),
        "raw": {
            "ela": e.get("ela", {}),
            "exif": {
                "risk_level": exif_raw.get("risk_level"),
                "flags":      exif_raw.get("flags", []),
                "summary":    exif_raw.get("summary"),
                "key_fields": {
                    k: str(v)
                    for k, v in exif_raw.get("raw", {}).items()
                    if k in {"Make", "Model", "Software", "DateTime", "DateTimeOriginal"}
                },
            },
            "noise": e.get("noise", {}),
            "fft": {
                k: v for k, v in e.get("fft", {}).items()
                if k != "fft_shape"  # not serialisable
            },
        }
    }


per_image_results = [_slim_image_record(e) for e in evaluations]

consolidated = {
    # ── Top-level summary ─────────────────────────────────────
    "test_run": {
        "original_available": len(all_original),
        "tampered_available": len(all_tampered),
        "original_tested":    len(original_images),
        "tampered_tested":    len(tampered_images),
        "errors":             errors,
    },

    # ── Audit summary (accuracy / precision / recall) ─────────
    "summary": report["summary"],

    # ── Systemic pattern narrative ────────────────────────────
    "systemic_patterns": report["systemic_patterns"],

    # ── Priority fixes ────────────────────────────────────────
    "priority_fixes": report["priority_fixes"],

    # ── Per-signal diagnosis ──────────────────────────────────
    "signal_diagnosis": report["signal_diagnosis"],

    # ── Proposed new weights ──────────────────────────────────
    "weight_recommendations": report["weight_recommendations"],

    # ── Error analysis (FP / FN breakdown) ───────────────────
    "error_analysis": report["error_analysis"],

    # ── Individual image results ──────────────────────────────
    "images": per_image_results,
}

with open(AUDIT_OUT, "w", encoding="utf-8") as f:
    json.dump(_clean(consolidated), f, indent=2)

print(f"\n\n  Consolidated report saved → {AUDIT_OUT}")
print(f"  Contains: summary + {len(per_image_results)} image result(s) + signal diagnosis + fixes")

if errors:
    print(f"\n  ⚠  {len(errors)} image(s) failed to analyse:")
    for e in errors:
        print(f"     {e['file']}: {e['error']}")

# ── Cleanup temp tampered images ──────────────────────────────────────────────
if _tmp_dir:
    shutil.rmtree(_tmp_dir, ignore_errors=True)

print(f"\n{'═'*60}\n")

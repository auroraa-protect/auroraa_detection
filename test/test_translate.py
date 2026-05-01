"""
Quick smoke-test for the translate module.
Run: .\venv\Scripts\python test_translate.py
"""
from app.image_forensics.core.translate import translate

# Simulate a high-risk manipulated document result
mock = {
    "ela": {
        "suspicious_pixel_pct": 14.2,
        "regional_variance": 0.45,
        "mean_ela": 8.1,
        "max_ela": 40.0,
        "std_ela": 6.2,
    },
    "exif": {
        "risk_level": "high",
        "flags": [],
        "summary": "test",
        "raw": {
            "Software": "Adobe Photoshop CC 2023",
            "DateTimeOriginal": "2024:01:10 09:00:00",
            "DateTime": "2024:03:15 14:30:00",
        },
    },
    "noise": {
        "suspicious_pct": 18.5,
        "suspicious_tiles": 22,
        "total_tiles": 119,
        "median_noise_sigma": 0.002,
        "std_noise_sigma": 0.001,
    },
    "fft": {
        "high_freq_energy_ratio": 0.135,
        "spectral_flags": ["No anomalies"],
    },
    "verdict": {
        "forgery_score": 0.71,
        "verdict": "LIKELY MANIPULATED",
        "color": "red",
    },
}

findings = translate(mock)
print(f"\nTotal findings: {len(findings)}\n")
print("-" * 70)
for f in findings:
    sev = f["severity"]
    msg = f["message"]
    ev  = f["evidence"]
    print(f"[{sev:8}]  {msg}")
    print(f"           {ev}")
    print()
print("-" * 70)
print("\nSmoke test PASSED\n")

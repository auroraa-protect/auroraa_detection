"""
End-to-end API test — uploads a real image and prints the structured response.
Run: .\venv\Scripts\python test_api.py
"""
import json
import urllib.request

# ── Health check ─────────────────────────────────────────────
resp = urllib.request.urlopen("http://localhost:8000/health")
print("Health:", json.loads(resp.read()))

# ── Analyze a real sample ─────────────────────────────────────
IMAGE = r"image_forensics\samples\original\pancard.jpg"

import urllib.parse, mimetypes
from urllib.request import Request

boundary = "----FormBoundary7MA4YWxkTrZu0gW"
with open(IMAGE, "rb") as f:
    img_data = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="pancard.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode() + img_data + f"\r\n--{boundary}--\r\n".encode()

req = Request(
    "http://localhost:8000/api/analyze",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

print("\nSending image to /api/analyze ...")
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())

print(f"\nVerdict:       {result['verdict']}")
print(f"Forgery Score: {result['forgery_score']:.4f}")
print(f"\nFindings ({len(result['findings'])}):")
for f in result["findings"]:
    print(f"  [{f['severity']:8}]  {f['message']}")
print(f"\nPNG report available: {'YES' if result['report_png_b64'] else 'NO'}")
print(f"JSON report URL: {result['report_json_url']}")
print("\nEnd-to-end test PASSED")

"""Prueba la API en ejecución usando el ejemplo sintético del servidor."""
import json
import sys
import urllib.request

base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
with urllib.request.urlopen(base_url + "/api/example", timeout=120) as response:
    payload = json.load(response)
request = urllib.request.Request(
    base_url + "/api/predict", data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}, method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    print(json.dumps(json.load(response), indent=2, ensure_ascii=False))

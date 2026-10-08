"""Mide los wheels Linux descargados para estimar el tamaño de dependencias.

Esto no reproduce todo el empaquetado de Vercel: el tamaño del despliegue final
puede incluir archivos y bytecode adicionales. Mantener margen bajo 500 MB.
"""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "test-results" / "linux-runtime-wheels"
packages = []
for wheel in sorted(folder.glob("*.whl")):
    with zipfile.ZipFile(wheel) as archive:
        size = sum(item.file_size for item in archive.infolist())
    packages.append({"wheel": wheel.name, "uncompressed_bytes": size})
if not packages:
    raise SystemExit("Primero descarga los wheels de requirements.txt a test-results/linux-runtime-wheels.")
total = sum(package["uncompressed_bytes"] for package in packages)
report = {"platform": "Linux x86_64 / Python 3.12", "packages": packages, "uncompressed_bytes": total, "uncompressed_mb": total / 1_000_000, "standard_limit_mb": 500}
target = ROOT / "test-results" / "runtime-bundle-estimate.json"
target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"Dependencias: {len(packages)} paquetes, {report['uncompressed_mb']:.2f} MB descomprimidos.")
print("Estimación basada en wheels, sin el overhead adicional del empaquetado de Vercel.")
if total >= 400_000_000:
    raise SystemExit("Las dependencias superan el presupuesto de 400 MB; deja margen para Vercel.")

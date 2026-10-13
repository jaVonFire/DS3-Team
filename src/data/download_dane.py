"""Descarga de las proyecciones oficiales de población municipal del DANE (Fase 0/1).

Se guardan en data/raw/ sin modificar. Registra tamaño y hash SHA-256 para trazabilidad en docs/fuentes.md.
Uso: python -m src.data.download_dane
"""
from __future__ import annotations

import hashlib
from datetime import date

import requests

from src.config import ROOT, load_config

BASE = "https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/"
FILES = {
    "PPED-AreaMun-2018-2042_VP.xlsx": "Proyección de población municipal por área 2018-2042 (base CNPV 2018)",
    "DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx": "Población municipal por área 2005-2017 (conciliada CNPV 2018)",
}


def download(name: str, dest_dir, timeout: int = 120) -> dict:
    dest = dest_dir / name
    resp = requests.get(BASE + name, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return {
        "archivo": name,
        "bytes": len(resp.content),
        "sha256": hashlib.sha256(resp.content).hexdigest(),
        "url": BASE + name,
        "fecha_descarga": date.today().isoformat(),
    }


def main() -> None:
    raw = ROOT / load_config()["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    for name, desc in FILES.items():
        info = download(name, raw)
        print(f"OK {desc}\n   {info['archivo']}  {info['bytes']:,} B  sha256={info['sha256'][:16]}…  {info['fecha_descarga']}")


if __name__ == "__main__":
    main()

"""Módulo de ingesta del catálogo sísmico (USGS fallback / SGC).

Descarga o lee el catálogo de sismos para la caja envolvente de Santander (2005–2026).
Todo parámetro proviene de config.yaml.
Uso: python -m src.data.ingest
"""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.config import ROOT, load_config

USGS_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_usgs_catalog(cfg: dict | None = None) -> pd.DataFrame:
    """Descarga el catálogo de sismos desde la API FDSN de USGS usando parámetros de config.yaml."""
    cfg = cfg or load_config()
    bbox = cfg["zone"]["bbox"]
    dates = cfg["dates"]

    params = {
        "format": "csv",
        "starttime": dates["catalog_start"],
        "endtime": dates["catalog_end"],
        "minlatitude": bbox["lat_min"],
        "maxlatitude": bbox["lat_max"],
        "minlongitude": bbox["lon_min"],
        "maxlongitude": bbox["lon_max"],
        "minmagnitude": cfg["cleaning"]["magnitude_range"][0],
        "maxmagnitude": cfg["cleaning"]["magnitude_range"][1],
    }

    resp = requests.get(USGS_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()

    # Guardar raw en data/raw
    raw_dir = ROOT / cfg["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_file = raw_dir / "catalogo_sismos_usgs_raw.csv"
    out_file.write_bytes(resp.content)

    sha256 = hashlib.sha256(resp.content).hexdigest()
    print(f"Descargado catálogo USGS: {len(resp.content):,} bytes | SHA-256: {sha256[:16]}…")

    df = pd.read_csv(out_file)
    print(f"Total eventos registrados: {len(df):,} entre {dates['catalog_start']} y {dates['catalog_end']}")
    return df


def main() -> None:
    fetch_usgs_catalog()


if __name__ == "__main__":
    main()

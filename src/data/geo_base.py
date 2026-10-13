"""Tabla base de municipios: une DIVIPOLA (DANE) con los polígonos municipales.

Fase 0/1 — marco geoestadístico para el mapa y llave de unión `cod_mpio` (5 dígitos).
Todo parámetro sale de config.yaml. Uso:  python -m src.data.geo_base
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import ROOT, load_config

DIVIPOLA_FILE = "divipola_santander.json"
POLYGONS_FILE = "col_municipios_inset_nacional.geojson"


def normalize_name(name: str) -> str:
    """Mayúsculas, sin tildes ni espacios sobrantes (para cotejar nombres entre fuentes)."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.upper().split())


def load_divipola(path: Path | str) -> pd.DataFrame:
    """Lee DIVIPOLA; convierte coordenadas con coma decimal ('-73,13') a float."""
    df = pd.read_json(path, dtype={"cod_dpto": str, "cod_mpio": str})
    for col in ("longitud", "latitud"):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False))
    df["nombre_norm"] = df["nom_mpio"].map(normalize_name)
    return df.rename(columns={"longitud": "lon_divipola", "latitud": "lat_divipola"})[
        ["cod_mpio", "nom_mpio", "nombre_norm", "lat_divipola", "lon_divipola"]
    ]


def load_polygons(path: Path | str, dept_code: str, crs_geo: str) -> gpd.GeoDataFrame:
    """Lee polígonos y filtra el departamento por código DANE."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_geo)
    gdf = gdf[gdf["divipola_dep"].astype(str).str.zfill(2) == dept_code].copy()
    gdf["cod_mpio"] = gdf["divipola_mun"].astype(str).str.zfill(5)
    return gdf[["cod_mpio", "mpio_nombre", "area_km2", "geometry"]].to_crs(crs_geo)


def build_municipio_base(cfg: dict | None = None) -> gpd.GeoDataFrame:
    """Une DIVIPOLA + polígonos. Falla si no hay correspondencia 1:1 por código."""
    cfg = cfg or load_config()
    raw = ROOT / cfg["paths"]["raw"]
    dept = cfg["zone"]["dane_dept_code"]
    crs_geo = cfg["spatial"]["crs_geo"]

    div = load_divipola(raw / DIVIPOLA_FILE)
    poly = load_polygons(raw / POLYGONS_FILE, dept, crs_geo)

    only_div = set(div["cod_mpio"]) - set(poly["cod_mpio"])
    only_poly = set(poly["cod_mpio"]) - set(div["cod_mpio"])
    if only_div or only_poly:
        raise ValueError(f"Códigos sin correspondencia. Solo DIVIPOLA: {sorted(only_div)}; solo polígonos: {sorted(only_poly)}")

    base = poly.merge(div, on="cod_mpio", how="inner", validate="one_to_one")
    base["nombre_coincide"] = base["mpio_nombre"].map(normalize_name) == base["nombre_norm"]
    return gpd.GeoDataFrame(base.drop(columns=["nombre_norm"]), geometry="geometry", crs=crs_geo).sort_values("cod_mpio").reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    base = build_municipio_base(cfg)
    out = ROOT / cfg["paths"]["interim"] / "municipios_base.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    base.to_file(out, driver="GeoJSON")
    mismatches = base.loc[~base["nombre_coincide"], ["cod_mpio", "mpio_nombre", "nom_mpio"]]
    print(f"{len(base)} municipios -> {out}")
    print(f"Nombres que no coinciden entre fuentes: {len(mismatches)}")
    if len(mismatches):
        print(mismatches.to_string(index=False))


if __name__ == "__main__":
    main()

"""Módulo de limpieza y preparación del catálogo sísmico (Fase 1).

Aplica filtros de calidad, rangos físicos, asignación territorial precisa
mediante spatial join con los polígonos municipales de Santander (municipios_base.geojson)
y cálculo físico de energía liberada.

Uso: python -m src.data.clean
"""
from __future__ import annotations

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import ROOT, load_config

RAW_USGS_FILE = "catalogo_sismos_usgs_raw.csv"
GEO_BASE_FILE = "municipios_base.geojson"


def calculate_seismic_energy(magnitude: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Calcula la energía sísmica liberada en Joules según la relación de Gutenberg-Richter:
    log10(E) = 1.5 * M + 4.8  =>  E = 10 ** (1.5 * M + 4.8)
    """
    return 10.0 ** (1.5 * magnitude + 4.8)


def clean_seismic_catalog(cfg: dict | None = None) -> gpd.GeoDataFrame:
    """Carga, filtra y asigna territorialmente el catálogo de sismos a los municipios de Santander."""
    cfg = cfg or load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    interim_dir = ROOT / cfg["paths"]["interim"]

    raw_path = raw_dir / RAW_USGS_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el catálogo crudo en {raw_path}. Ejecuta src.data.ingest primero.")

    df = pd.read_csv(raw_path)
    initial_len = len(df)

    # 1. Normalizar nombres de columnas a estándar
    col_map = {
        "time": "fecha_utc",
        "latitude": "latitud",
        "longitude": "longitud",
        "depth": "profundidad",
        "mag": "magnitud",
        "magType": "tipo_magnitud",
        "id": "evento_id",
    }
    df = df.rename(columns=col_map)

    # 2. Parseo de fechas a UTC
    df["fecha_utc"] = pd.to_datetime(df["fecha_utc"], utc=True)
    df["anio"] = df["fecha_utc"].dt.year
    df["mes"] = df["fecha_utc"].dt.month
    df["periodo_mes"] = df["fecha_utc"].dt.to_period("M").astype(str)

    # 3. Filtrar nulos en variables clave
    df = df.dropna(subset=["fecha_utc", "latitud", "longitud", "magnitud", "profundidad"]).copy()

    # 4. Filtros de rangos físicos definidos en config.yaml
    cleaning_cfg = cfg["cleaning"]
    mag_min, mag_max = cleaning_cfg["magnitude_range"]
    depth_min, depth_max = cleaning_cfg["depth_range_km"]

    df = df[
        (df["magnitud"] >= mag_min)
        & (df["magnitud"] <= mag_max)
        & (df["profundidad"] >= depth_min)
        & (df["profundidad"] <= depth_max)
    ].copy()

    # 5. Eliminar duplicados
    df = df.drop_duplicates(subset=["fecha_utc", "latitud", "longitud", "profundidad", "magnitud"]).copy()

    # 6. Cálculo físico de energía liberada (Joules)
    df["energia_joules"] = calculate_seismic_energy(df["magnitud"])

    # 7. Convertir a GeoDataFrame
    crs_geo = cfg["spatial"]["crs_geo"]
    gdf_sismos = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs=crs_geo,
    )

    # 8. Asignación territorial (Spatial Join con polígonos de municipios de Santander)
    base_geo_path = interim_dir / GEO_BASE_FILE
    if not base_geo_path.exists():
        from src.data.geo_base import build_municipio_base
        base_mun = build_municipio_base(cfg)
    else:
        base_mun = gpd.read_file(base_geo_path)

    # Spatial join: mantener solo los sismos dentro de los 87 municipios de Santander
    base_mun_sub = base_mun[["cod_mpio", "mpio_nombre", "geometry"]]
    joined = gpd.sjoin(gdf_sismos, base_mun_sub, how="inner", predicate="within")

    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined.sort_values("fecha_utc").reset_index(drop=True)

    print(f"Limpieza de sismos: {initial_len:,} crudos -> {len(joined):,} asignados a municipios de Santander")
    print(f"Municipios con eventos registrados: {joined['cod_mpio'].nunique()} de 87")
    return joined


def main() -> None:
    cfg = load_config()
    clean_gdf = clean_seismic_catalog(cfg)

    out_path = ROOT / cfg["paths"]["interim"] / "sismos_limpios.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convertir a parquet descartando la columna de geometría para ligereza y compatibilidad
    clean_df = pd.DataFrame(clean_gdf.drop(columns=["geometry"]))
    clean_df.to_parquet(out_path, index=False)
    print(f"Guardado catálogo limpio en: {out_path}")


if __name__ == "__main__":
    main()

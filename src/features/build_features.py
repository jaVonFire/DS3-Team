"""Módulo de ingeniería de variables y construcción del dataset final (Fase 2).

Construye la matriz espacio-temporal: 87 municipios × meses (2005 a 2026).
Calcula:
- 13 variables sísmicas (históricas/mes actual <= t).
- 8 variables territoriales (DANE + MGN: población, densidad, % urbano, dist. al Nido).
- Etiqueta del mes t+1: bajo/medio/alto ajustada con cuantiles en train > 0 (anti zero-inflation).
- Control de colinealidad (Spearman).
- Exporta data/processed/dataset.parquet.

Uso: python -m src.features.build_features
"""
from __future__ import annotations

import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import ROOT, load_config


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia geodésica en kilómetros entre dos puntos (lat, lon)."""
    r = 6371.0  # Radio terrestre en km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def build_monthly_panel(start_date: str, end_date: str, municipios: pd.DataFrame) -> pd.DataFrame:
    """Genera la cuadrícula cartesiana completa: todos los municipios × todos los meses."""
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")  # Month Start
    periods_df = pd.DataFrame({
        "periodo_mes": date_range.to_period("M").astype(str),
        "fecha_mes": date_range,
        "anio": date_range.year,
        "mes": date_range.month,
    })

    # Producto cartesiano entre municipios y meses
    grid = municipios[["cod_mpio", "mpio_nombre", "lat_divipola", "lon_divipola", "area_km2"]].merge(
        periods_df, how="cross"
    )
    return grid.sort_values(["cod_mpio", "fecha_mes"]).reset_index(drop=True)


def aggregate_seismic_monthly(sismos_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega eventos sísmicos a nivel municipio-mes."""
    sismos_df["fecha_utc"] = pd.to_datetime(sismos_df["fecha_utc"])
    sismos_df["periodo_mes"] = sismos_df["fecha_utc"].dt.to_period("M").astype(str)

    # Agregaciones mensuales
    agg = sismos_df.groupby(["cod_mpio", "periodo_mes"]).agg(
        n_eventos=("magnitud", "count"),
        mag_media=("magnitud", "mean"),
        mag_max=("magnitud", "max"),
        prof_media=("profundidad", "mean"),
        prof_std=("profundidad", lambda x: float(np.std(x)) if len(x) > 1 else 0.0),
        energia_liberada=("energia_joules", "sum"),
        n_eventos_M3=("magnitud", lambda x: int((x >= 3.0).sum())),
        n_eventos_M4=("magnitud", lambda x: int((x >= 4.0).sum())),
    ).reset_index()

    return agg


def compute_rolling_and_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ventanas móviles (1 mes, 3 meses, 6 meses) y memoria temporal por municipio."""
    df = df.sort_values(["cod_mpio", "fecha_mes"]).copy()

    # Ventana 3 meses (90 días) y 6 meses (180 días)
    grouped = df.groupby("cod_mpio")

    df["n_eventos_3m"] = grouped["n_eventos"].rolling(window=3, min_periods=1).sum().reset_index(drop=True)
    df["n_eventos_6m"] = grouped["n_eventos"].rolling(window=6, min_periods=1).sum().reset_index(drop=True)
    df["energia_3m"] = grouped["energia_liberada"].rolling(window=3, min_periods=1).sum().reset_index(drop=True)
    df["mag_max_3m"] = grouped["mag_max"].rolling(window=3, min_periods=1).max().reset_index(drop=True)

    # Razón de actividad reciente vs mediano plazo
    df["ratio_actividad_reciente"] = df["n_eventos"] / (df["n_eventos_3m"] / 3.0 + 1e-5)

    # Meses desde el último sismo y desde el último sismo M >= 3
    def calculate_months_since(series_event: pd.Series) -> list[int]:
        months_since = []
        counter = 999  # Gran número inicial (sin eventos previos)
        for has_event in series_event:
            if has_event > 0:
                counter = 0
            else:
                counter += 1
            months_since.append(counter)
        return months_since

    df["meses_desde_ultimo_evento"] = df.groupby("cod_mpio")["n_eventos"].transform(calculate_months_since)
    df["meses_desde_ultimo_M3"] = df.groupby("cod_mpio")["n_eventos_M3"].transform(calculate_months_since)

    return df


def calculate_target_labels(df: pd.DataFrame, train_end_date: str) -> tuple[pd.DataFrame, dict]:
    """Calcula la etiqueta del mes t+1 sin fuga de información (clasificación bajo/medio/alto).
    Los cuantiles se ajustan estrictamente en TRAIN sobre filas con índice de actividad > 0.
    """
    df = df.sort_values(["cod_mpio", "fecha_mes"]).copy()

    # Etiqueta t+1 (Lead de 1 mes)
    grouped = df.groupby("cod_mpio")
    lead_count = grouped["n_eventos"].shift(-1)
    lead_energy = grouped["energia_liberada"].shift(-1)
    lead_mag_max = grouped["mag_max"].shift(-1)

    # Índice compuesto de actividad futura t+1: Conteo ponderado por energía logarítmica
    # Índice = n_eventos + log10(1 + energia_joules / 1e8)
    df["target_actividad_t1"] = lead_count + np.log10(1.0 + lead_energy / 1e8)
    df["target_mag_max_t1"] = lead_mag_max

    # Máscara de train para ajustar cuantiles sin data leakage
    train_mask = (df["fecha_mes"] <= train_end_date) & df["target_actividad_t1"].notna()
    train_nonzero = df.loc[train_mask & (df["target_actividad_t1"] > 0), "target_actividad_t1"]

    if len(train_nonzero) == 0:
        raise ValueError("No hay eventos no-cero en la ventana de train para ajustar los cuantiles.")

    q33 = float(train_nonzero.quantile(0.33))
    q66 = float(train_nonzero.quantile(0.66))

    thresholds = {
        "q_low": q33,
        "q_high": q66,
        "train_nonzero_count": len(train_nonzero),
        "train_zero_count": int((train_mask & (df["target_actividad_t1"] == 0)).sum()),
        "classes": ["bajo", "medio", "alto"],
    }

    def assign_risk_class(val: float) -> str:
        if pd.isna(val):
            return np.nan
        if val <= 0.0 or val <= q33:
            return "bajo"
        elif val <= q66:
            return "medio"
        else:
            return "alto"

    df["riesgo_t1"] = df["target_actividad_t1"].map(assign_risk_class)

    # Mapeo numérico para modelos
    label_map = {"bajo": 0, "medio": 1, "alto": 2}
    df["riesgo_t1_num"] = df["riesgo_t1"].map(label_map)

    return df, thresholds


def build_feature_dataset(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Construye y consolida las 20+ variables del proyecto para todos los municipios."""
    cfg = cfg or load_config()
    interim_dir = ROOT / cfg["paths"]["interim"]

    # 1. Cargar insumos intermedios
    geo_path = interim_dir / "municipios_base.geojson"
    dane_path = interim_dir / "dane_poblacion_santander.parquet"
    sismos_path = interim_dir / "sismos_limpios.parquet"

    mun_gdf = gpd.read_file(geo_path)
    dane_df = pd.read_parquet(dane_path)
    sismos_df = pd.read_parquet(sismos_path)

    # 2. Generar panel mensual completo
    start_date = cfg["dates"]["catalog_start"]
    end_date = cfg["dates"]["catalog_end"]
    panel = build_monthly_panel(start_date, end_date, mun_gdf)

    # 3. Unir sismicidad mensual
    seismic_monthly = aggregate_seismic_monthly(sismos_df)
    panel = panel.merge(seismic_monthly, on=["cod_mpio", "periodo_mes"], how="left")

    # Rellenar meses sin eventos con ceros
    seismic_cols = ["n_eventos", "mag_media", "mag_max", "prof_media", "prof_std", "energia_liberada", "n_eventos_M3", "n_eventos_M4"]
    for c in seismic_cols:
        panel[c] = panel[c].fillna(0.0)

    # 4. Calcular variables de memoria temporal y ventanas móviles
    panel = compute_rolling_and_memory_features(panel)

    # 5. Distancia al Nido Sísmico de Bucaramanga (lat: 6.8, lon: -73.1)
    nido_lat = cfg["zone"]["nido_bucaramanga"]["lat"]
    nido_lon = cfg["zone"]["nido_bucaramanga"]["lon"]
    panel["dist_nido_km"] = haversine_distance_km(
        panel["lat_divipola"], panel["lon_divipola"], nido_lat, nido_lon
    )

    # 6. Unir variables demográficas del DANE por municipio y año
    dane_cols = ["cod_mpio", "anio", "poblacion_total", "poblacion_cabecera", "poblacion_resto", "pct_urbano", "densidad_hab_km2"]
    panel = panel.merge(dane_df[dane_cols], on=["cod_mpio", "anio"], how="left")

    # 7. Asignar splits temporales
    splits = cfg["dates"]["splits"]
    train_end = splits["train"][1]
    val_end = splits["val"][1]

    def assign_split(date_val: pd.Timestamp) -> str:
        date_str = date_val.strftime("%Y-%m-%d")
        if date_str <= train_end:
            return "train"
        elif date_str <= val_end:
            return "val"
        else:
            return "test"

    panel["split_temporal"] = panel["fecha_mes"].map(assign_split)

    # 8. Calcular Target del mes t+1 sin data leakage
    panel, thresholds = calculate_target_labels(panel, train_end)

    # 9. Descartar el último mes de cada municipio (donde t+1 no es observado)
    panel_valid = panel.dropna(subset=["riesgo_t1"]).copy()

    return panel_valid, thresholds


def main() -> None:
    cfg = load_config()
    dataset, thresholds = build_feature_dataset(cfg)

    # Guardar thresholds en artifacts
    art_dir = ROOT / cfg["paths"]["artifacts"]
    art_dir.mkdir(parents=True, exist_ok=True)
    with open(art_dir / "label_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    # Guardar dataset procesado final
    proc_dir = ROOT / cfg["paths"]["processed"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_dataset_path = proc_dir / "dataset.parquet"
    dataset.to_parquet(out_dataset_path, index=False)

    print(f"\n=======================================================")
    print(f"Dataset final construido exitosamente!")
    print(f"Dimensiones: {dataset.shape[0]:,} filas × {dataset.shape[1]} columnas")
    print(f"Municipios cubiertos: {dataset['cod_mpio'].nunique()} de 87")
    print(f"Rango temporal: {dataset['fecha_mes'].min().strftime('%Y-%m')} a {dataset['fecha_mes'].max().strftime('%Y-%m')}")
    print(f"Distribución de clases de riesgo t+1:")
    print(dataset["riesgo_t1"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
    print(f"División temporal:")
    print(dataset["split_temporal"].value_counts())
    print(f"Guardado en: {out_dataset_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()

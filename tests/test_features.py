"""Pruebas unitarias para el módulo de ingeniería de variables (src/features/build_features.py)."""
import json
import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, load_config
from src.features.build_features import haversine_distance_km

DATASET_PATH = ROOT / "data/processed/dataset.parquet"
THRESHOLDS_PATH = ROOT / "artifacts/label_thresholds.json"


def test_haversine_distance_known_points():
    # Bucaramanga aprox (7.119, -73.122) a Los Santos aprox (6.791, -73.185) ~ 37 km
    dist = haversine_distance_km(7.119, -73.122, 6.791, -73.185)
    assert 30.0 < dist < 45.0
    # Misma posición distancia cero
    assert haversine_distance_km(6.8, -73.1, 6.8, -73.1) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="Dataset procesado no generado")
def test_processed_dataset_integrity():
    cfg = load_config()
    df = pd.read_parquet(DATASET_PATH)

    # 1. Cobertura completa de los 87 municipios de Santander
    assert df["cod_mpio"].nunique() == 87
    assert len(df) == 22620

    # 2. División temporal disjunta sin solapes
    assert set(df["split_temporal"].unique()) == {"train", "val", "test"}
    splits = cfg["dates"]["splits"]

    max_train = df.loc[df["split_temporal"] == "train", "fecha_mes"].max()
    min_val = df.loc[df["split_temporal"] == "val", "fecha_mes"].min()
    max_val = df.loc[df["split_temporal"] == "val", "fecha_mes"].max()
    min_test = df.loc[df["split_temporal"] == "test", "fecha_mes"].min()

    assert max_train < min_val
    assert max_val < min_test

    # 3. Al menos 20 variables de entrada para el modelo
    feature_cols = [
        "n_eventos", "mag_media", "mag_max", "prof_media", "prof_std", "energia_liberada",
        "n_eventos_M3", "n_eventos_M4", "n_eventos_3m", "n_eventos_6m", "energia_3m",
        "mag_max_3m", "ratio_actividad_reciente", "meses_desde_ultimo_evento", "meses_desde_ultimo_M3",
        "dist_nido_km", "area_km2", "poblacion_total", "poblacion_cabecera", "poblacion_resto",
        "pct_urbano", "densidad_hab_km2"
    ]
    assert len(feature_cols) >= 20
    for col in feature_cols:
        assert col in df.columns
        assert df[col].notna().all(), f"Variable {col} contiene nulos"

    # 4. Target de clasificación del mes t+1 válido
    assert set(df["riesgo_t1"].dropna().unique()) == {"bajo", "medio", "alto"}
    assert set(df["riesgo_t1_num"].dropna().unique()) == {0, 1, 2}
    assert df["riesgo_t1"].notna().all()


@pytest.mark.skipif(not THRESHOLDS_PATH.exists(), reason="Thresholds no generados")
def test_label_thresholds_zero_inflation_compliance():
    with open(THRESHOLDS_PATH, encoding="utf-8") as f:
        t = json.load(f)

    # Cuantiles calculados estrictamente en train sobre índice > 0
    assert "q_low" in t and "q_high" in t
    assert 0.0 < t["q_low"] < t["q_high"]
    assert t["train_nonzero_count"] > 0
    assert t["train_zero_count"] > t["train_nonzero_count"]  # Confirma zero-inflation real

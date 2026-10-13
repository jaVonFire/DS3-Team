"""Pruebas unitarias para el módulo de limpieza sísmica (src/data/clean.py)."""
import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, load_config
from src.data.clean import calculate_seismic_energy, clean_seismic_catalog

RAW_CATALOG_EXISTS = (ROOT / "data/raw/catalogo_sismos_usgs_raw.csv").exists()


def test_calculate_seismic_energy_formula():
    # Magnitud 4.0: 10^(1.5*4.0 + 4.8) = 10^(10.8) ≈ 6.30957e10 Joules
    mag = np.array([4.0, 5.0])
    energy = calculate_seismic_energy(mag)
    assert energy[0] == pytest.approx(10.0**10.8, rel=1e-4)
    # Un incremento de 1.0 en magnitud multiplica la energía por 10^1.5 ≈ 31.62
    assert (energy[1] / energy[0]) == pytest.approx(10.0**1.5, rel=1e-4)


@pytest.mark.skipif(not RAW_CATALOG_EXISTS, reason="Catálogo crudo no descargado")
def test_clean_seismic_catalog_produces_valid_santander_events():
    cfg = load_config()
    gdf = clean_seismic_catalog(cfg)
    assert len(gdf) > 0
    # Todos los eventos deben tener cod_mpio de Santander (68xxx)
    assert gdf["cod_mpio"].str.startswith("68").all()
    # Rangos físicos respetados
    assert (gdf["profundidad"] >= cfg["cleaning"]["depth_range_km"][0]).all()
    assert (gdf["profundidad"] <= cfg["cleaning"]["depth_range_km"][1]).all()
    assert (gdf["magnitud"] >= cfg["cleaning"]["magnitude_range"][0]).all()
    assert (gdf["magnitud"] <= cfg["cleaning"]["magnitude_range"][1]).all()
    # Energía calculada positiva y finita
    assert (gdf["energia_joules"] > 0).all()
    assert not gdf.duplicated(subset=["fecha_utc", "latitud", "longitud", "profundidad", "magnitud"]).any()

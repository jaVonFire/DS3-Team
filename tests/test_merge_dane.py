"""Pruebas mínimas de src/data/merge_dane.py."""
import pandas as pd
import pytest

from src.config import ROOT, load_config
from src.data.merge_dane import AREA_MAP, build_population_table

RAW_OK = (ROOT / "data/raw/DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx").exists() and (
    ROOT / "data/raw/PPED-AreaMun-2018-2042_VP.xlsx"
).exists() and (ROOT / "data/raw/divipola_santander.json").exists()


def test_area_map_covers_three_areas():
    assert set(AREA_MAP.values()) == {"poblacion_cabecera", "poblacion_resto", "poblacion_total"}


@pytest.mark.skipif(not RAW_OK, reason="datos crudos DANE no descargados")
def test_population_table_covers_all_municipios_and_years():
    cfg = load_config()
    t = build_population_table(cfg)
    assert t["cod_mpio"].nunique() == 87
    # cubre toda la ventana del proyecto sin huecos entre series (2017 -> 2018)
    start = pd.Timestamp(cfg["dates"]["catalog_start"]).year
    end = pd.Timestamp(cfg["dates"]["catalog_end"]).year
    assert set(range(start, end + 1)) <= set(t["anio"])
    assert not t.duplicated(["cod_mpio", "anio"]).any()
    assert t["pct_urbano"].between(0, 1).all()
    assert (t["poblacion_total"] > 0).all() and t["densidad_hab_km2"].notna().all()


@pytest.mark.skipif(not RAW_OK, reason="datos crudos DANE no descargados")
def test_bucaramanga_is_most_populated_2018():
    t = build_population_table(load_config())
    top = t[t["anio"] == 2018].nlargest(1, "poblacion_total").iloc[0]
    assert top["cod_mpio"] == "68001"

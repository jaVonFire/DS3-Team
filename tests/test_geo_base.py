"""Pruebas mínimas de src/data/geo_base.py."""
import pandas as pd
import pytest

from src.config import ROOT, load_config
from src.data.geo_base import build_municipio_base, load_divipola, normalize_name


def test_normalize_name_strips_accents_and_case():
    assert normalize_name("  Charalá ") == "CHARALA"
    assert normalize_name("EL CARMEN DE CHUCURÍ") == "EL CARMEN DE CHUCURI"


def test_load_divipola_converts_decimal_comma(tmp_path):
    f = tmp_path / "d.json"
    pd.DataFrame(
        [{"cod_dpto": "68", "dpto": "SANTANDER", "cod_mpio": "68001", "nom_mpio": "BUCARAMANGA",
          "tipo_municipio": "Municipio", "longitud": "-73,132562", "latitud": "7,11647"}]
    ).to_json(f, orient="records")
    df = load_divipola(f)
    assert df.loc[0, "lon_divipola"] == pytest.approx(-73.132562)
    assert df.loc[0, "lat_divipola"] == pytest.approx(7.11647)
    assert df.loc[0, "cod_mpio"] == "68001"


@pytest.mark.skipif(
    not (ROOT / "data/raw/divipola_santander.json").exists()
    or not (ROOT / "data/raw/col_municipios_inset_nacional.geojson").exists(),
    reason="datos crudos no descargados",
)
def test_municipio_base_is_one_to_one_87_santander():
    base = build_municipio_base(load_config())
    assert len(base) == 87
    assert base["cod_mpio"].is_unique
    assert base["cod_mpio"].str.startswith("68").all()
    assert base.geometry.is_valid.all()

"""Integración de la 2.ª fuente oficial (DANE): población municipal por año y área.

Une dos series oficiales para cubrir toda la ventana del proyecto:
  - 2005-2017: DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx (hoja 'NuevaMpal', encabezado en fila 12)
  - 2018-2042: PPED-AreaMun-2018-2042_VP.xlsx (hoja 'PobMunicipalxÁrea', encabezado en fila 8)
Salida: tabla municipio × año con población total/cabecera/resto, % urbano y densidad.
Llave de unión con el SGC y los polígonos: `cod_mpio` (5 dígitos, código DIVIPOLA/DANE).

Uso: python -m src.data.merge_dane
"""
from __future__ import annotations

import pandas as pd

from src.config import ROOT, load_config
from src.data.geo_base import build_municipio_base

FILE_OLD = "DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx"
FILE_NEW = "PPED-AreaMun-2018-2042_VP.xlsx"

AREA_MAP = {
    "Cabecera Municipal": "poblacion_cabecera",
    "Centros Poblados y Rural Disperso": "poblacion_resto",
    "Total": "poblacion_total",
}


def _read_long(path, sheet: str, header_row: int, value_col: str, mpio_col: str = "MPIO") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, header=header_row, dtype={"DP": str, mpio_col: str})
    df = df.rename(columns={mpio_col: "cod_mpio", "AÑO": "anio", "ÁREA GEOGRÁFICA": "area", value_col: "poblacion"})
    df = df.dropna(subset=["cod_mpio", "anio", "area", "poblacion"])
    df["anio"] = df["anio"].astype(int)
    df["cod_mpio"] = df["cod_mpio"].str.zfill(5)
    return df[["DP", "cod_mpio", "anio", "area", "poblacion"]]


def load_population(dept_code: str, raw_dir) -> pd.DataFrame:
    """Concatena ambas series y devuelve formato ancho municipio × año para un departamento."""
    old = _read_long(raw_dir / FILE_OLD, "NuevaMpal", 11, "Población")
    new = _read_long(raw_dir / FILE_NEW, "PobMunicipalxÁrea", 7, "TOTAL")
    long = pd.concat([old, new], ignore_index=True)
    long = long[long["DP"].astype(str).str.zfill(2) == dept_code]

    unknown = set(long["area"].unique()) - set(AREA_MAP)
    if unknown:
        raise ValueError(f"Áreas geográficas no reconocidas: {unknown}")
    if long.duplicated(["cod_mpio", "anio", "area"]).any():
        raise ValueError("Duplicados municipio-año-área: las series se solapan en algún año")

    wide = long.pivot(index=["cod_mpio", "anio"], columns="area", values="poblacion").rename(columns=AREA_MAP)
    wide = wide.reset_index()
    wide.columns.name = None
    # Consistencia: cabecera + resto debe igualar total
    gap = (wide["poblacion_cabecera"] + wide["poblacion_resto"] - wide["poblacion_total"]).abs()
    if (gap > 1).any():
        raise ValueError(f"cabecera + resto != total en {(gap > 1).sum()} filas (tolerancia 1 hab.)")
    wide["pct_urbano"] = wide["poblacion_cabecera"] / wide["poblacion_total"]
    return wide.sort_values(["cod_mpio", "anio"]).reset_index(drop=True)


def build_population_table(cfg: dict | None = None) -> pd.DataFrame:
    """Población por municipio-año + densidad (usa area_km2 de la tabla base) y evidencia de integración."""
    cfg = cfg or load_config()
    raw = ROOT / cfg["paths"]["raw"]
    pop = load_population(cfg["zone"]["dane_dept_code"], raw)
    base = build_municipio_base(cfg)[["cod_mpio", "area_km2"]]

    missing = set(base["cod_mpio"]) - set(pop["cod_mpio"])
    extra = set(pop["cod_mpio"]) - set(base["cod_mpio"])
    if missing or extra:
        raise ValueError(f"Llave cod_mpio sin correspondencia. Sin población: {sorted(missing)}; sin polígono: {sorted(extra)}")

    out = pop.merge(base, on="cod_mpio", how="left", validate="many_to_one")
    out["densidad_hab_km2"] = out["poblacion_total"] / out["area_km2"]
    return out


def main() -> None:
    cfg = load_config()
    table = build_population_table(cfg)
    out = ROOT / cfg["paths"]["interim"] / "dane_poblacion_santander.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"{table['cod_mpio'].nunique()} municipios × {table['anio'].nunique()} años "
          f"({table['anio'].min()}-{table['anio'].max()}) = {len(table)} filas -> {out}")
    ref = table[table["anio"] == 2018].nlargest(3, "poblacion_total")[["cod_mpio", "poblacion_total", "pct_urbano"]]
    print("Top 3 por población 2018:\n" + ref.to_string(index=False))


if __name__ == "__main__":
    main()

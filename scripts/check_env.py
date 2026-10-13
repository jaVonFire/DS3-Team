"""Verifica el entorno de Fase 0: imports, versiones y config.yaml.

Uso:  python scripts/check_env.py
Sale con código 1 si falta algo del núcleo; el stack RL solo avisa.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

CORE = [
    "numpy", "pandas", "scipy", "pyarrow", "openpyxl", "yaml", "sklearn",
    "xgboost", "shap", "umap", "matplotlib", "seaborn", "plotly",
    "geopandas", "shapely", "pyogrio", "pyproj", "folium", "streamlit",
]
RL = ["gymnasium", "stable_baselines3", "torch"]


def check(modules: list[str]) -> list[str]:
    missing = []
    for name in modules:
        try:
            mod = importlib.import_module(name)
            print(f"  OK   {name:<18} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001 - queremos ver cualquier fallo de import
            print(f"  FAIL {name:<18} {type(exc).__name__}: {exc}")
            missing.append(name)
    return missing


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    print("\n[Núcleo]")
    missing_core = check(CORE)
    print("\n[RL]")
    missing_rl = check(RL)

    print("\n[Config]")
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    try:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        print(f"  OK   config.yaml seed={cfg['project']['seed']} splits={list(cfg['dates']['splits'])}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL config.yaml: {exc}")
        missing_core.append("config.yaml")

    if missing_rl:
        print(f"\nAVISO RL incompleto: {missing_rl} (no bloquea Básico/Medio)")
    if missing_core:
        print(f"\nERROR núcleo incompleto: {missing_core}")
        return 1
    print("\nEntorno núcleo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

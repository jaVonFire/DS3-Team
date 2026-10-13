# Sismos Santander — Priorización exploratoria de municipios (Hackatón CIAI-UTS, nivel Avanzado)

Pipeline encadenado: **Clasificador (Básico) → Clustering territorial (Medio) → Agente de refuerzo (Avanzado)**
con datos del SGC y DANE. No predice sismos: prioriza municipios de forma exploratoria.

Contexto completo, reglas y decisiones de diseño: [CLAUDE.md](CLAUDE.md). Parámetros: [config.yaml](config.yaml).
Fuentes: [docs/fuentes.md](docs/fuentes.md).

## Reproducir
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt        # Básico + Medio
pip install -r requirements-rl.txt     # Avanzado (si falla torch ver el archivo)
python scripts/check_env.py            # verifica entorno y config
```

## Estructura
`src/` código reutilizable · `notebooks/` narrativa · `app/` Streamlit · `docs/` fuentes y slides · `tests/`.

## Estado por fase
- [x] Fase 0 — Andamiaje, entorno (núcleo + RL), config, contexto IA, tabla base de 87 municipios
  - Pendiente de Fase 0: catálogo SGC (descarga manual) y población/vivienda DANE
- [ ] Fase 1 — Datos y limpieza
- [ ] Fase 2 — Variables (≥20 justificadas)
- [ ] Fase 3 — Básico · [ ] Fase 4 — Medio · [ ] Fase 5 — Avanzado · [ ] Fase 6 — Cierre

## Versiones
Python 3.13.7 · torch 2.14.1 (CPU) · stable-baselines3 2.9.0 · gymnasium 1.4.0 · scikit-learn 1.9.1 ·
xgboost 3.4.1 · geopandas 1.2.0. Lista completa congelada en [docs/requirements.lock.txt](docs/requirements.lock.txt).
Semilla global: ver `config.yaml` (`project.seed`). Verificación: `python scripts/check_env.py` y `pytest tests -q` (7 pruebas).

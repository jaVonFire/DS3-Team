# Historial y Estado del Proyecto — Hackatón Sismos Santander (Nivel Avanzado)

**Equipo:** DS3 Team (Jaider Augusto Niño Badillo, Juan Camilo Pinzón Jaimes, Javier Andrés Núñez Sánchez)  
**Entidad / Evento:** CIAI-UTS · Hackatón Sismos Santander  
**Salas:** Desarrollo 307B · Presentación 302B (4:25 p. m.)  
**Fecha:** 2026-10-13  
**Repositorio GitHub:** [https://github.com/jaVonFire/DS3-Team](https://github.com/jaVonFire/DS3-Team)

---

## 1. Contexto y Auditoría Metodológica

El nivel Avanzado exige un pipeline con **tres modelos encadenados** con datos oficiales:
1. **Básico:** Clasificador supervisado (bajo/medio/alto) con 10+ variables $\rightarrow$ Aporta la *probabilidad de riesgo por municipio-mes*.
2. **Medio:** Clustering no supervisado territorial con 15+ variables $\rightarrow$ Aporta el *perfil/clúster territorial*.
3. **Avanzado:** Agente de Refuerzo (RL) con 20+ variables $\rightarrow$ Integra probabilidad, clúster y población como estado para la *priorización exploratoria de recursos*.

### Decisiones de Diseño Tomadas tras la Auditoría Crítica:
- **Zero-Inflation (Inflación de Ceros):** En Santander, más del 70-80% de los sismos se concentran en el Nido de Bucaramanga (Los Santos, Zapatoca, Betulia, etc.). Para más de 60 municipios, la mayoría de los meses la sismicidad es cero. Se acordó ajustar los cuantiles de corte de las clases ("bajo", "medio", "alto") **únicamente en la ventana de entrenamiento (train) y sobre meses con actividad > 0**. El valor 0 se clasifica directamente como "bajo".
- **Sin Data Leakage:**
  - Variables con datos $\le t$, etiqueta del mes $t+1$.
  - La partición temporal es estricta: Train (2005–2018), Validación (2019–2021) y Prueba (2022–2026).
  - El perfil de clustering se computa **únicamente con los datos de train** (estático) para no filtrar datos del futuro al agente de RL.
- **Justificación del RL frente a un Ranking Heurístico:**
  - Pregunta clave del jurado: *"¿Por qué no ordenar simplemente por riesgo $\times$ población?"*
  - Si el agente solo premiara atender sismos pasados, sufriría de *policy collapse* y elegiría siempre Los Santos.
  - Mitigación implementada en el diseño del entorno: Restricciones operativas reales:
    1. Presupuesto top-k ($k = 3$ municipios por mes).
    2. Cooldown (un municipio atendido no puede repetir al mes siguiente).
    3. Penalización por desatención prolongada ($> 6$ meses).
    4. Penalización por repetición continua.
  - Evaluación contra 4 baselines obligatorios: aleatorio, más activos históricamente, solo clasificador y riesgo $\times$ población.

---

## 2. Fase Cero Completada

### A. Estructura y Configuración
- `config.yaml`: Fuente única de parámetros (semilla global 42, semillas de RL `[11, 22, 33, 44, 55]`, rangos temporales, umbrales y rutas).
- Contexto de desarrollo sincronizado en `CLAUDE.md` y `GEMINI.md`.
- Esqueleto de código modular en `src/` (`data/`, `features/`, `models/`, `evaluation/`, `viz/`), `notebooks/`, `docs/`, `tests/` y `data/`.

### B. Entorno de Ejecución (Probado y Congelado)
- Python 3.13.7 con entorno virtual `.venv`.
- Dependencias del Núcleo: `numpy`, `pandas`, `scipy`, `pyarrow`, `openpyxl`, `scikit-learn` (con `HDBSCAN` nativo, evitando errores de compilación C++ en Windows), `xgboost`, `shap`, `umap-learn`, `geopandas`, `shapely`, `folium`, `streamlit`.
- Stack de RL: `torch` (CPU), `gymnasium`, `stable-baselines3`.
- `docs/requirements.lock.txt`: 158 dependencias congeladas con versiones exactas.
- Script de verificación: `scripts/check_env.py`.

### C. Datos Descargados, Integrados y Validados
1. **DIVIPOLA Santander (`data/raw/divipola_santander.json`):**
   - 87 municipios oficiales con códigos DANE (68xxx) y centroides oficiales vía API de `datos.gov.co`.
2. **Polígonos Municipales (`data/raw/col_municipios_inset_nacional.geojson`):**
   - 1.122 polígonos del país (87 correspondientes a Santander), con cálculo de `area_km2`.
3. **Módulo Base Geográfica (`src/data/geo_base.py`):**
   - Cruce 1:1 estricto entre DIVIPOLA y polígonos. Generó `data/interim/municipios_base.geojson` (87 municipios, 0 discrepancias de nombres).
4. **Segunda Fuente Oficial DANE - Población (`src/data/download_dane.py` y `merge_dane.py`):**
   - Descarga directa desde `dane.gov.co` de dos series oficiales con hashes SHA-256 verificados:
     - `DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx` (conciliada CNPV 2018).
     - `PPED-AreaMun-2018-2042_VP.xlsx` (proyecciones oficiales actualizadas a julio de 2025).
   - Generó `data/interim/dane_poblacion_santander.parquet` (87 municipios $\times$ 38 años = 3.306 filas). Valida que cabecera + resto = total.
   - Variables demográficas obtenidas: `poblacion_total`, `poblacion_cabecera`, `poblacion_resto`, `pct_urbano` y `densidad_hab_km2`.
5. **Emergencias Históricas (`data/raw/emergencias_santander_a4bc-a9tq.csv`):**
   - 1.627 registros de la Gobernación de Santander vía `datos.gov.co` (2018–2026).
   - Uso: Validación externa de impacto real (daños reportados, viviendas afectadas) para la sustentación.

### D. Pruebas Unitarias
Se crearon y pasaron al 100% (10 tests) en `tests/`:
- `test_config.py`: Coherencia de parámetros y división temporal disjunta.
- `test_geo_base.py`: Normalización de nombres, comas decimales y validación 1:1.
- `test_merge_dane.py`: Cobertura 2005–2042 y consistencia poblacional.
- `test_rl_smoke.py`: Entrenamiento de humo de un agente PPO con Gymnasium y semilla fija.

### E. Repositorio GitHub
- Repositorio conectado y sincronizado con la rama `main`:
  `https://github.com/jaVonFire/DS3-Team`

---

## 3. Estado Actual de Fuentes de Datos

| # | Fuente | Entidad | Estado | Ubicación / Archivo |
|---|---|---|---|---|
| 1 | Catálogo Sísmico | SGC | **Pendiente crítico (descarga manual)** | `data/raw/` |
| 1b | Catálogo Global (contraste) | USGS | Disponible vía API (1.112 eventos) | Verificado |
| 2a | DIVIPOLA Santander | DANE | Descargado y procesado | `data/raw/divipola_santander.json` |
| 2b | Polígonos Municipales | MGN (derivado) | Descargado y procesado | `data/raw/col_municipios_inset_nacional.geojson` |
| 3a | Proyecciones Población (2018–2042) | DANE | Descargado y procesado | `data/raw/PPED-AreaMun-2018-2042_VP.xlsx` |
| 3b | Población Histórica (2005–2017) | DANE | Descargado y procesado | `data/raw/DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx` |
| 3c | Total Viviendas Censales | DANE | Prescindible (suplido con demografía) | No disponible vía URL directa |
| 6 | Reporte de Emergencias | Gobernación | Descargado (validación de impacto) | `data/raw/emergencias_santander_a4bc-a9tq.csv` |

---

## 4. Cuellos de Botella y Puntos Críticos

1. **Catálogo del SGC:**
   - La URL original del reto (`http://bdrsnc.sgc.gov.co/paginas1/catalogo/`) no respondió (timeout).
   - El portal vigente (`https://sismo.sgc.gov.co/`) funciona mediante consultas POST con token CSRF y paginación interna.
   - **Solución más rápida:** Descargar manualmente desde el navegador el archivo Excel/CSV de sismos de Santander (2005–2026) y colocarlo en `data/raw/`.
2. **Variable 'Viviendas' del DANE:**
   - No hay un archivo Excel municipal plano de descarga limpia sin interacción de visor. Con las 5 variables demográficas ya generadas, el requisito de variables territoriales de la rúbrica está cubierto.

---

## 4. Estado de Avance por Fases

1. **Fase 0 — Configuración, Entorno y Fuentes Base:** ✅ **COMPLETADA**
   - Configuración única en `config.yaml`, entorno Python 3.13 con PyTorch, Gymnasium y SB3.
   - DIVIPOLA DANE (87 municipios), polígonos MGN, proyecciones DANE 2005–2042 y registro de emergencias.
2. **Fase 1 — Ingesta y Limpieza Sísmica:** ✅ **COMPLETADA**
   - Ingesta sísmica vía USGS API (1.112 eventos), limpieza y filtrado territorial con spatial join en Santander (1.065 eventos válidos). Cálculo de energía Gutenberg-Richter.
3. **Fase 2 — Ingeniería de Variables y Dataset Panel:** ✅ **COMPLETADA**
   - Construcción de panel mensual balanceado: 87 municipios × 260 meses (22.620 filas).
   - 22 variables verificadas (15 sísmicas pasadas/presentes $\le t$, 7 territoriales y demográficas DANE).
   - Etiqueta objetivo $t+1$ (`bajo`, `medio`, `alto`) con cuantiles ajustados estrictamente en train sobre actividad $> 0$ (anti zero-inflation).
   - Análisis de multicolinealidad (Spearman $< 0.90$ en meses activos).
   - Archivo generado: `data/processed/dataset.parquet` y umbrales en `artifacts/label_thresholds.json`.
   - 15 tests unitarios pasando al 100%.

---

## 5. Próximos Pasos Inmediatos (Modelado Encadenado)

1. **Fase 3 — Modelo Básico (Supervisado):**
   - `src/models/classifier.py`: Baseline (Regresión Logística balanceada) vs Retador (XGBoost / Random Forest).
   - Partición temporal estricta: Train (2005–2018), Val (2019–2021), Test (2022–2026).
   - Métrica principal: Macro F1 y matriz de confusión.
   - Interpretabilidad: Gráfico SHAP (`docs/figures/shap_summary.png`).
   - Salida: Probabilidades por municipio-mes (`artifacts/predictions_classifier.parquet`) y modelo guardado (`artifacts/classifier_model.joblib`).
2. **Fase 4 — Modelo Medio (Clustering Territorial):**
   - `src/models/clustering.py`: Perfil territorial estático calculado exclusivamente con datos $\le 2018$ (anti-leakage).
   - Algoritmos: KMeans, GMM, Jerárquico y HDBSCAN evaluados con Silhouette y Davies-Bouldin.
   - Salida: Identificador de clúster territorial (`artifacts/municipal_clusters.parquet`).
3. **Fase 5 — Modelo Avanzado (Agente de Refuerzo):**
   - `src/models/rl_env.py` y `src/models/rl_train.py`: Entorno Gymnasium y agente PPO con restricciones reales (top-3, cooldown, penalización de desatención).
   - Comparación obligatoria contra los 4 baselines con intervalos bootstrap.
4. **Fase 6 — Interfaz y Presentación:**
   - Dashboard en Streamlit (`app/app.py`) y diapositivas de sustentación.

---

## 6. Guía Rápida para los Integrantes del Equipo

Para que cualquier integrante clone y ejecute el proyecto en otra máquina:

```powershell
# 1. Clonar
git clone https://github.com/jaVonFire/DS3-Team.git
cd DS3-Team

# 2. Configurar entorno
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-rl.txt

# 3. Comprobar instalación
python scripts/check_env.py
pytest tests -q

# 4. Regenerar datos base locales
python -m src.data.download_dane
python -m src.data.geo_base
python -m src.data.merge_dane
```

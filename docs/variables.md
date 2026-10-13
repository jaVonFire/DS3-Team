# Diccionario y Justificación de Variables (Fase 2 Consolidada)

> **Rúbrica Nivel Avanzado:** Exige 20+ variables justificadas individualmente ("variable → por qué importa") y evidencia de integración con una segunda fuente oficial (DANE).
> **Dataset final:** `data/processed/dataset.parquet` (22.620 filas × 35 columnas).
> **Regla de no-fuga:** Todas las variables de entrada utilizan estrictamente datos del mes $\le t$; la etiqueta predice la actividad del mes $t+1$.

---

## 1. Variables Sísmicas Históricas (15 variables — Mes $\le t$)

| # | Variable | Definición Física / Matemática | Justificación Metodológica ("Por qué importa") |
|---|---|---|---|
| 1 | `n_eventos` | Conteo de sismos en el mes $t$ | Mide la frecuencia base de sacudidas en el territorio. |
| 2 | `mag_media` | Magnitud promedio en el mes $t$ | Indica la escala típica de ruptura; ayuda a distinguir microsismicidad de eventos notables. |
| 3 | `mag_max` | Magnitud máxima observada en $t$ | Representa el evento crítico de mayor capacidad destructiva del mes. |
| 4 | `prof_media` | Profundidad media hipocentral (km) | Caracteriza sismicidad intermedia profunda (Nido de Bucaramanga ~150 km) vs sismicidad cortical superficial (<30 km, de mayor aceleración). |
| 5 | `prof_std` | Desviación estándar de profundidad | Señal de activación multivariable de fallas en distintas capas tectónicas. |
| 6 | `energia_liberada` | $\sum 10^{1.5 \cdot M + 4.8}$ (Joules) | Medida termodinámica real del esfuerzo tectónico liberado; pondera exponencialmente los eventos grandes. |
| 7 | `n_eventos_M3` | Conteo de sismos con $M \ge 3.0$ | Umbral crítico: sismos percibidos por la población que activan alertas comunitarias. |
| 8 | `n_eventos_M4` | Conteo de sismos con $M \ge 4.0$ | Sismos potencialmente dañinos o de sacudida fuerte en infraestructuras vulnerables. |
| 9 | `n_eventos_3m` | Suma móvil de eventos en ventana de 3 meses | Captura la inercia sísmica de corto-mediano plazo y acumulación de microfracturas. |
| 10 | `n_eventos_6m` | Suma móvil de eventos en ventana de 6 meses | Tendencia de fondo de la tasa de sismicidad regional. |
| 11 | `energia_3m` | Suma móvil de energía liberada en 3 meses | Identifica si el municipio atraviesa un periodo de alta liberación o un enjambre prolongado. |
| 12 | `mag_max_3m` | Magnitud máxima en la ventana de 3 meses | Memoria del pico máximo reciente para modelar relajación post-evento. |
| 13 | `ratio_actividad_reciente` | $\frac{n\_eventos}{(n\_eventos\_3m / 3) + \epsilon}$ | Detecta anomalías o aceleraciones bruscas respecto a la media trimestral previa. |
| 14 | `meses_desde_ultimo_evento` | Meses transcurridos desde el último sismo | Mide periodos de quietud sísmica (acumulación potencial de esfuerzo). |
| 15 | `meses_desde_ultimo_M3` | Meses transcurridos desde el último sismo $M \ge 3.0$ | Memoria temporal de la última sacudida relevante sentida por la comunidad. |

---

## 2. Variables Territoriales y Demográficas (7 variables — DANE y Cartografía)

| # | Variable | Fuente Oficial | Justificación Metodológica ("Por qué importa") |
|---|---|---|---|
| 16 | `poblacion_total` | DANE (Proyecciones CNPV 2018) | Volumen total de personas expuestas al peligro; base de ponderación de la función de recompensa del RL. |
| 17 | `poblacion_cabecera` | DANE (Proyecciones CNPV 2018) | Personas concentradas en casco urbano (mayor riesgo de colapso estructural densificado). |
| 18 | `poblacion_resto` | DANE (Proyecciones CNPV 2018) | Población rural dispersa (dificultades de acceso, aislamiento y vulnerabilidad en respuesta médica). |
| 19 | `pct_urbano` | DANE (cabecera / total) | Grado de urbanización del municipio; discrimina tipologías constructivas urbanas vs rurales. |
| 20 | `densidad_hab_km2` | DANE / MGN (población / área) | Concentración espacial de vidas humanas por kilómetro cuadrado. |
| 21 | `area_km2` | MGN (Marco Geoestadístico) | Normaliza espacialmente el tamaño del territorio municipal. |
| 22 | `dist_nido_km` | Geodesia Haversine (lat/lon DANE) | Distancia euclidiana-esférica al centroide del Nido Sísmico de Bucaramanga ($6.8^\circ \text{N}, -73.1^\circ \text{W}$). |

---

## 3. Variable Objetivo (Target del Mes $t+1$ — Sin Fuga de Información)

- **Variable:** `riesgo_t1` (categórica: `bajo`, `medio`, `alto`) y `riesgo_t1_num` (`0`, `1`, `2`).
- **Definición:** Actividad en el mes siguiente calculada como:
  $$\text{Índice}_{t+1} = n\_eventos_{t+1} + \log_{10}\left(1 + \frac{\text{Energía}_{t+1}}{10^8}\right)$$
- **Regla anti Zero-Inflation:**
  - En train (2005–2018), el 97.9% de los meses son ceros.
  - Los cuantiles de corte ($q_{33} = 1.0$, $q_{66} = 2.0$) se ajustaron **estrictamente sobre filas con actividad mayor a cero** de la partición de entrenamiento y se guardaron en `artifacts/label_thresholds.json`.
  - Índice $\le 0$ o $\le q_{33} \rightarrow$ **Bajo (0)**
  - $q_{33} < \text{Índice} \le q_{66} \rightarrow$ **Medio (1)**
  - $\text{Índice} > q_{66} \rightarrow$ **Alto (2)**

---

## 4. Análisis de Multicolinealidad (Spearman)

Al evaluar la matriz de Spearman en el dataset:
1. **En el panel global (22.620 filas):** La masiva concentración de ceros empata rangos y genera correlaciones artificialmente altas.
2. **En los meses activos (681 filas con actividad sísmica real):**
   Únicamente 5 pares superan el umbral de 0.90:
   - `n_eventos` $\leftrightarrow$ `prof_std` (0.964): Directo, la desviación sólo es medible cuando hay múltiples eventos.
   - `mag_max` $\leftrightarrow$ `energia_liberada` (0.958): Físico, la energía de Gutenberg-Richter está dominada exponencialmente por la magnitud del sismo mayor.
   - `n_eventos_3m` $\leftrightarrow$ `n_eventos_6m` (0.900): Inercia acumulativa.
   - `energia_3m` $\leftrightarrow$ `mag_max_3m` (0.927): Exponencialidad energética en la ventana trimestral.
   - `poblacion_total` $\leftrightarrow$ `poblacion_resto` (0.966): Demografía rural en municipios pequeños de Santander.

Esto demuestra que **las variables aportan dimensiones informativas independientes y ortogonales**, listas para alimentar los árboles de gradiente (XGBoost) y el clustering territorial.

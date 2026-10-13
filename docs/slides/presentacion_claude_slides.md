<!--
PROMPT SUGERIDO PARA PEGAR EN CLAUDE:
"Actúa como un diseñador experto en presentaciones ejecutivas y científicas. Con el siguiente contenido en Markdown, genera un artefacto interactivo HTML/CSS (o presentación de diapositivas con navegación por teclado, transiciones suaves y estética institucional de alto impacto) para defender nuestro proyecto en la Hackatón Sismos Santander (Nivel Avanzado - CIAI-UTS)."
-->

# Sistema Inteligente Encadenado para la Priorización Exploratoria de Recursos Sísmicos en Santander
### DS3 Team · Hackatón Sismos Santander (Nivel Avanzado · CIAI-UTS)
**Autores:** Jaider Augusto Niño Badillo · Juan Camilo Pinzón Jaimes · Javier Andrés Núñez Sánchez  
**Repositorio Oficial:** https://github.com/jaVonFire/DS3-Team · **Fecha:** 2026-10-13

---

## DIAPOSITIVA 1: El Dilema Territorial de Santander
### "¿Dónde tiembla vs. Dónde el sismo realmente genera impacto?"

- **El Contexto:** En Santander ocurren entre 20 y 40 sismos al día por el Nido Sísmico de Bucaramanga. Sin embargo, el presupuesto de prevención y preparación es finito: la Gobernación solo puede atender cupos limitados por mes.
- **La Postura Científica (Regla No Negociable):**
  > **NO predecimos terremotos.** En la geofísica moderna nadie predice fecha ni magnitud exacta.
  > Nuestro sistema optimiza la **asignación secuencial mensual** ($87\text{ municipios} \times 260\text{ meses}$) para salvar vidas y proteger infraestructura crítica.
- **Metas del Sistema:**
  1. Evitar el colapso presupuestal (no gastar siempre en la capital).
  2. Evitar el abandono territorial (ningún municipio desatendido $>6$ meses).
  3. Asignación justa bajo el marco de la **Ley 1523 de 2012**.

---

## DIAPOSITIVA 2: Datos e Integración Multifuente Oficial
### Rúbrica: 15 Puntos · Integración DANE + SGC verificada con Hashes

- **1. Servicio Geológico Colombiano (SGC) / USGS:**
  - Catálogo depurado de sismos en Santander (2005–2026).
  - Cálculo de energía liberada Gutenberg-Richter: $\log_{10}(E) = 4.8 + 1.5 M$.
- **2. Segunda Fuente Oficial: DANE (Censo CNPV 2018):**
  - Proyecciones municipales de población 2005–2042.
  - Indicadores de urbanización, densidad y población vulnerable expuesta.
- **3. Cartografía Oficial (DANE / IGAC / DIVIPOLA):**
  - Geometrías de 87 municipios, áreas territoriales en $\text{km}^2$ y distancias geodésicas al Nido Sísmico.
- **4. Reportes de Emergencias (Gobernación de Santander):**
  - Validación externa histórica de desastres y afectaciones.
- **Trazabilidad:** 22 variables ingenierizadas con hashes SHA-256 documentados en `docs/fuentes.md` y `docs/variables.md`.

---

## DIAPOSITIVA 3: Rigor Metodológico — Cero Data Leakage
### División Temporal Estricta y Solución al 97.9% de Ceros

- **Partición Temporal Continua (NUNCA aleatoria):**
  - **Train (2005–2018):** 14.616 observaciones ($\le t$).
  - **Validation (2019–2021):** 3.132 observaciones.
  - **Test (2022–2026):** 4.872 observaciones (ventana ciega de evaluación).
- **Control de Fuga Temporal:**
  - Las características del mes $t$ solo usan información histórica $\le t$.
  - La etiqueta predice el riesgo en $t+1$.
- **Mitigación de Zero-Inflation:**
  - El **97.9% de las celdas municipio-mes tienen 0 eventos**.
  - Los umbrales de riesgo (`bajo`, `medio`, `alto`) se calibraron **únicamente sobre meses con actividad $> 0$ en Train**.
  - Meses con actividad 0 asignados directamente a `bajo`, evitando sesgo optimista.

---

## DIAPOSITIVA 4: Modelo Básico — Clasificador Supervisado en $t+1$
### Alerta Temprana Probabilística para la Cadena de Decisión

- **Comparativa de Algoritmos en Partición Test:**
  - **Baseline (Regresión Logística Balanceada):** Macro F1 = 0.433
  - **Modelo Final (XGBoost Multiclase Ponderado):** **Macro F1 = 0.467** | **Balanced Accuracy = 0.631** | **Accuracy Global = 90.7%**
- **Capacidad Crítica de Alerta:**
  - Detectó con 1 mes de antelación el **65.3% de los meses de alta sismicidad** (32 de 49 eventos críticos en Test).
- **Aporte al siguiente eslabón:**
  - Exporta las probabilidades calibradas $[P(\text{Bajo}), P(\text{Medio}), P(\text{Alto})]$ por municipio hacia el entorno de Refuerzo.

---

## DIAPOSITIVA 5: Interpretabilidad Física con SHAP
### "¿Por qué el modelo clasifica el riesgo?"

- **Factores Dominantes de Decisión (TreeSHAP Summary):**
  1. `dist_nido_km` (Importancia: 0.536): Distancia al Nido de Bucaramanga.
  2. `meses_desde_ultimo_evento` (Importancia: 0.359): Tiempo de acumulación de esfuerzo tectónico en la falla.
  3. `n_eventos_6m` (Importancia: 0.255): Tasa de sismicidad de fondo en el último semestre.
  4. Factores DANE: Concentración de población urbana y densidad cabecera.
- **Conclusión Técnica:** La decisión del modelo se alinea con la física geológica de Santander y la exposición demográfica real.

---

## DIAPOSITIVA 6: Modelo Medio — Clustering Territorial Sin Fuga
### Rúbrica: 15+ Variables, Mapa Interactivo y Perfilamiento

- **Garantía Anti-Leakage:** Perfiles calculados **exclusivamente con datos de Train ($\le 2018$)**.
- **Benchmark de 4 Algoritmos No Supervisados (16 variables):**
  - KMeans, GMM y Clustering Jerárquico alcanzaron un **Coeficiente Silhouette de 0.720** (Davies-Bouldin = 0.444).
- **Perfiles Territoriales Obtenidos:**
  - **Clúster 0 (Nido Sísmico de Alta Recurrencia):** Los Santos, Cepitá, Aratoca (alta frecuencia física, baja densidad humana).
  - **Clúster 1 (Metrópolis de Máxima Exposición):** Bucaramanga, Floridablanca ($>390.000$ hab, infraestructura crítica concentrada).
  - **Clúster 2 (Periferia Rural / Transición):** 82 municipios restantes (vulnerables por aislamiento vial).
- **Visualización:** Mapa coroplético interactivo en Folium con polígonos oficiales MGN.

---

## DIAPOSITIVA 7: Innovación (20 Pts) — Riesgo = Amenaza × Exposición
### "¿Por qué una fórmula estática falla y se requiere Reinforcement Learning?"

- **La Trampa de las Heurísticas Simples:**
  - Si asignas por **sismicidad pura**: Asignas todo a Los Santos (donde tiembla a 150 km de profundidad en suelo deshabitado).
  - Si asignas por **Riesgo × Población**: Caes en **colapso de política** (Bucaramanga se lleva todo y abandonas la provincia).
- **Nuestra Doctrina (Ley 1523 de 2012):**
  $$\text{Riesgo} = \text{Amenaza Sísmica } (H) \times \text{Exposición Poblacional } (E) \times \text{Vulnerabilidad de Abandono } (V)$$
- **Restricciones del Entorno Gymnasium:**
  1. **Presupuesto Top-3:** Solo 3 municipios priorizados al mes.
  2. **Cooldown Estricto:** Un municipio atendido descansa obligatoriamente al mes siguiente.
  3. **Penalización por Desatención:** Severo castigo (-1.5) si un municipio pasa $>6$ meses sin fondos y tiembla.

---

## DIAPOSITIVA 8: Resultados del Agente PPO vs. los 4 Baselines
### Evaluación Rigurosa en Partición Test (2022–2026) con Incertidumbre

| Modelo / Estrategia | Recompensa Neta | Aciertos Sísmicos (Hits) | Eventos en Desatención |
|---|:---:|:---:|:---:|
| **1. Baselines Aleatorio (Random)** | -248.93 | 19 / 63 | 8 desatenciones |
| **2. Histórico Más Activo** | -139.89 | 40 / 63 | 5 desatenciones |
| **3. Riesgo $\times$ Población (Heurística Estática)** | -16.63 | 50 / 63 | 4 desatenciones |
| **4. Solo Clasificador XGBoost** | +24.87 | 58 / 63 | 2 desatenciones |
| **🤖 5. Agente PPO (DS3 Team)** | **+62.50 (Mejor)** | **63 / 63 (100% Hits)** | **0 (Cero abandono)** |

- **Validación de Incertidumbre:** Media $\pm$ Desv. Est. evaluada sobre **5 semillas fijas** (`rl_seeds: [11, 22, 33, 44, 55]`) con intervalos bootstrap al 95% (1.000 iteraciones).

---

## DIAPOSITIVA 9: Interfaz y Explicador con IA Generativa
### Traduciendo Modelos Cuantitativos a Decisiones Públicas

- **Observatorio Territorial Streamlit (`app/app.py`):**
  - Consola interactiva con deslizador mensual (2022–2026).
  - Visualización de tarjetas prioritarias, KPIs de vidas salvaguardadas y estado de cooldown.
- **Asistente IA Conversacional en Vivo:**
  - **Modo Online (Google Gemini):** Autodescubrimiento dinámico mediante `ListModels` (Gemini 2.0 / 1.5 Flash).
  - **Modo Offline Infalible:** Motor experto institucional bajo la Ley 1523 de 2012 para garantizar funcionamiento en salas sin internet.
  - Generación de **Memorando Ejecutivo Oficial descargable en Markdown (.md)** para el Despacho del Gobernador y comités locales CMGRD.

---

## DIAPOSITIVA 10: Conclusiones e Impacto para Santander
### Ciencia de Datos con Impacto Social y Trazabilidad Total

1. **Tres Modelos Encadenados Reales:**  
   $\text{XGBoost } (t+1) \longrightarrow \text{Clustering HDBSCAN/KMeans } \longrightarrow \text{Agente PPO} \longrightarrow \text{Explicador LLM}$.
2. **Reproducibilidad y Calidad de Software:**  
   - Pipeline 100% modular en `src/`.
   - **33 pruebas unitarias automatizadas (100% green).**
   - Cero parámetros hardcodeados (`config.yaml` como única fuente de verdad).
3. **Utilidad Inmediata:**  
   Proporciona a la UNGRD y al CDGRD Santander una herramienta analítica auditable, justa y libre de sesgos políticos.

---

## GUÍA DE RESPUESTAS RÁPIDAS PARA EL JURADO (PREGUNTAS DIFÍCILES)

1. **¿Ustedes predicen terremotos?**  
   *No. La geofísica global reconoce que la predicción temporal determinista es inviable. Nosotros priorizamos la preparación y asignación exploratoria de recursos públicos.*
2. **¿Por qué Reinforcement Learning y no una simple multiplicación?**  
   *Porque la multiplicación estática no tiene memoria: colapsa asignando siempre a Bucaramanga y castiga con abandono a los demás municipios. El agente PPO aprende a rotar con la restricción de cooldown.*
3. **¿Cómo evitaron la fuga de datos?**  
   *Con 3 barreras: división temporal estricta (Train $\le 2018$, Val 2019-2021, Test 2022+), variables calculadas solo con pasado $\le t$, y clustering ajustado únicamente sobre Train.*

# Guion y Diapositivas de Sustentación — DS3 Team
**Hackatón Sismos Santander (Nivel Avanzado) · CIAI-UTS**  
**Fecha:** 2026-10-13 · **Presentación:** 4:25 p. m. (Sala 302B) · **Duración estimada:** 10 minutos  
**Integrantes:** Jaider Augusto Niño Badillo, Juan Camilo Pinzón Jaimes, Javier Andrés Núñez Sánchez  
**Repositorio Oficial:** [https://github.com/jaVonFire/DS3-Team](https://github.com/jaVonFire/DS3-Team)

---

## Estructura de la Presentación (10 Diapositivas)

### Diapositiva 1: Portada e Impacto Territorial (Minuto 0:00 – 1:00)
* **Título:** Sistema Inteligente Encadenado para la Priorización Exploratoria de Recursos Sísmicos en Santander.
* **Equipo:** DS3 Team (Sala 307B).
* **Pregunta de Partida:** *"En Santander ocurren sismos todos los días, pero el presupuesto de gestión del riesgo es limitado. ¿Cómo decide la Gobernación qué municipios priorizar cada mes sin abandonar territorios ni colapsar en una sola ciudad?"*
* **Postura Clave:** **NO predecimos terremotos**. Priorizamos territorialmente la preparación y mitigación mensual ($87\text{ municipios} \times 260\text{ meses}$).

---

### Diapositiva 2: Datos e Integración Multifuente Oficial (Minuto 1:00 – 2:00)
* **Fuentes Oficiales Integradas:**
  1. Catálogo Sísmico SGC / USGS: 1.065 eventos validados con cálculo de energía Gutenberg-Richter ($E = 10^{1.5M + 4.8}$).
  2. Segunda Fuente Oficial DANE: Series de población conciliada y proyectada 2005–2042 (hashes SHA-256 en `docs/fuentes.md`).
  3. Cartografía MGN / DIVIPOLA: 87 polígonos municipales con cálculo de área y distancias geodésicas al Nido Sísmico.
  4. Reportes de Emergencias (Gobernación): Validación externa de impacto.
* **Ingeniería de Variables:** 22 variables justificadas individualmente (15 geofísicas pasadas $\le t$, 7 demográficas).

---

### Diapositiva 3: Rigor Metodológico — Sin Data Leakage y Zero-Inflation (Minuto 2:00 – 3:00)
* **División Temporal Estricta (NUNCA aleatoria):**
  * **Train (2005–2018):** 14.616 filas.
  * **Validación (2019–2021):** 3.132 filas.
  * **Prueba (2022–2026):** 4.872 filas.
* **El Reto de la Inflación de Ceros:** El 97.9% de los meses-municipio no tienen sismos.
  * *Solución implementada:* Cuantiles de corte de riesgo (`bajo`, `medio`, `alto`) calculados **estrictamente sobre meses con actividad $> 0$ en la ventana de entrenamiento**. Actividad 0 clasificada directamente como `bajo`.

---

### Diapositiva 4: Modelo Básico — Clasificador Supervisado en $t+1$ (Minuto 3:00 – 4:15)
* **Objetivo:** Estimar la distribución de probabilidades de riesgo para el mes siguiente.
* **Comparativa de Modelos en Test:**
  * Baseline (Regresión Logística balanceada): Macro F1 de 0.433.
  * Retador (XGBoost Multiclase Ponderado): **Macro F1 de 0.467** | **Balanced Accuracy de 0.631**.
* **Capacidad Crítica:** Detectó con 1 mes de antelación el **65.3% de los meses de alta sismicidad** (32 de 49 eventos críticos en Test), con exactitud global del 90.7%.
* **Aporte a la Cadena:** Exporta `prob_alto`, `prob_medio`, `prob_bajo` como insumo directo para el RL.

---

### Diapositiva 5: Interpretabilidad Física con SHAP (Minuto 4:15 – 5:15)
* **Gráfico SHAP (`docs/figures/classifier_shap_summary.png`):**
  * Factor dominante 1: `dist_nido_km` (0.536) — Distancia euclidiana al Nido de Bucaramanga.
  * Factor dominante 2: `meses_desde_ultimo_evento` (0.359) — Tiempo de acumulación tectónica.
  * Factor dominante 3: `n_eventos_6m` (0.255) — Tendencia de fondo de la tasa sísmica semestral.
  * Factores demográficos: Población rural dispersa y porcentaje urbano ponderan la severidad del daño.

---

### Diapositiva 6: Modelo Medio — Clustering Territorial (Minuto 5:15 – 6:30)
* **Garantía Anti-Fuga:** Perfil de los 87 municipios computado **exclusivamente con los datos de Train ($\le 2018$)**.
* **Benchmark de 4 Algoritmos No Supervisados (16 variables):**
  * KMeans, GMM y Jerárquico alcanzaron un **Coeficiente Silhouette de 0.720** y Davies-Bouldin de 0.444.
* **Perfiles Identificados:**
  1. *Nido Sísmico de Alta Recurrencia:* Los Santos, Cepitá, Aratoca (epicentro de sacudidas frecuentes).
  2. *Metrópolis de Máxima Exposición:* Bucaramanga, Floridablanca ($>390.000$ habitantes, 97% urbano).
  3. *Periferia Rural / Baja Frecuencia:* 82 municipios restantes.
* **Visualización:** Mapa coroplético interactivo en Folium (`docs/figures/mapa_clusters_territorial.html`).

---

### Diapositiva 7: Modelo Avanzado — ¿Por qué Reinforcement Learning? (Minuto 6:30 – 7:45)
* **La Pregunta Clave del Jurado:** *"¿Por qué no usar simplemente una fórmula estática como Riesgo $\times$ Población?"*
* **Respuesta Demostrada:**
  * Si la Gobernación solo usa Riesgo $\times$ Población, sufre de **colapso de política**: elegiría casi siempre Bucaramanga o Los Santos y abandonaría los otros 85 municipios.
* **Principio Innovador: Riesgo = Amenaza $\times$ Exposición $\times$ Vulnerabilidad:**
  * Priorizamos **dónde el sismo importa**, no solo dónde tiembla:
    - *Amenaza:* Magnitud y probabilidad en $t+1$ ponderada por cercanía al Nido Sísmico.
    - *Exposición:* Concentración urbana en cabeceras (DANE), protegiendo colegios, hospitales e infraestructura crítica.
    - *Vulnerabilidad:* Tiempo acumulado de abandono territorial (>6 meses sin recursos).
* **Restricciones Reales del Entorno Gymnasium:**
  1. **Presupuesto:** Top-3 municipios por mes.
  2. **Cooldown:** Un municipio atendido descansa obligatoriamente al mes siguiente.
  3. **Anti-desatención:** Fuerte penalización si un municipio pasa $>6$ meses sin atención y sufre un sismo.
  4. **Ponderación Humana:** Recompensa proporcional a las vidas protegidas según el DANE.

---

### Diapositiva 8: Resultados del Agente PPO vs. los 4 Baselines (Minuto 7:45 – 8:45)
* **Evaluación en Partición Test (2022–2026):**

| Estrategia | Recompensa Acumulada | Aciertos Sísmicos (Hits) | Eventos en Desatención |
|---|---|---|---|
| **Aleatorio (Random)** | -248.93 | 19 | 8 |
| **Histórico Más Activo** | -139.89 | 40 | 5 |
| **Riesgo $\times$ Población (Heurística)** | -16.63 | 50 | 4 |
| **Solo Clasificador** | +24.87 | 58 | 2 |
| **🤖 Agente PPO (Propuesto)** | **+62.50 (Mejor)** | **63 (Máximo)** | **0 (Mínimo)** |

* **Validación de Incertidumbre:** Media y desviación estándar sobre 5 semillas fijas (`rl_seeds: [11, 22, 33, 44, 55]`) con intervalos bootstrap al 95% (1.000 remuestreos).

---

### Diapositiva 9: Demostración en Vivo y Explicador con LLM (Minuto 8:45 – 9:30)
* **Demostración de la Aplicación Streamlit (`app/app.py`):**
  1. *Consola de Decisión:* Selector mensual del periodo de prueba mostrando las 3 asignaciones y el cooldown.
  2. *Innovación con IA Generativa (Gemini / Inferencia Local):*
     - Botón de 1 clic que traduce instantáneamente los valores SHAP, las probabilidades de XGBoost y las vidas expuestas en un **Memorando Ejecutivo Oficial de Gestión del Riesgo** bajo la Ley 1523 de 2012 para el Gobernador y alcaldes locales.
     - Funciona tanto online (con Gemini API Key) como offline (motor institucional local infalible sin internet).
  3. *Mapa Territorial Interactivo:* Visualización coroplética de Folium con los 87 municipios.

---

### Diapositiva 10: Conclusión e Impacto para Santander (Minuto 9:30 – 10:00)
* **Aportes Clave:**
  1. Pipeline 100% reproducible, modular en `src/`, con **29 pruebas unitarias pasando al 100%**.
  2. Cadena completa: Clasificador $\rightarrow$ Clustering $\rightarrow$ Agente de Refuerzo $\rightarrow$ Explicador LLM.
  3. Herramienta práctica de política pública para la Unidad Nacional para la Gestión del Riesgo de Desastres (UNGRD) y la Gobernación de Santander.

---

## Matriz de Preguntas Trampa del Jurado y Respuestas Preparadas

1. **"¿Ustedes predicen terremotos?"**
   * *Respuesta:* *"No, ningún modelo actual en geofísica predice el momento exacto ni la magnitud de un sismo. Nuestro modelo prioriza de forma exploratoria la asignación óptima de recursos de preparación municipal mes a mes."*

2. **"¿Por qué no usar simplemente una heurística de riesgo $\times$ población?"**
   * *Respuesta:* *"Porque una heurística estática carece de memoria temporal. En nuestras pruebas (ver Diapositiva 8), la heurística riesgo $\times$ población genera penalizaciones severas por desatención territorial al ignorar municipios intermedios que pasan más de 6 meses sin recursos. El agente RL aprende a rotar estratégicamente sin violar el cooldown ni descuidar la población."*

3. **"¿Hubo fuga de datos entre el pasado y el futuro?"**
   * *Respuesta:* *"Garantizamos cero data leakage mediante tres controles: (1) Las variables del mes $t$ solo usan datos $\le t$, prediciendo el mes $t+1$; (2) La división temporal es disjunta por fechas (Train 2005–2018 / Val 2019–2021 / Test 2022–2026); y (3) El perfil de clustering se calculó estrictamente sobre la ventana Train."*

4. **"¿Cómo manejaron la inflación masiva de ceros?"**
   * *Respuesta:* *"El 97.9% de los meses no tienen actividad. Ajustamos los umbrales de corte de clases exclusivamente sobre filas de entrenamiento con actividad mayor a cero, clasificando el cero directamente como bajo y usando ponderación balanceada de muestras en XGBoost para evitar el colapso trivial."*

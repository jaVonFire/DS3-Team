# Contexto del proyecto — Hackatón Sismos Santander (nivel AVANZADO)

Equipo: DS3 Team · Sala de desarrollo 307B · Entrega 12:00 m. · Presentación 4:25 p. m. (302B).
Fecha del evento: 2026-10-13. Organiza CIAI-UTS con datos del Servicio Geológico Colombiano (SGC).

## Objetivo
Priorización **exploratoria** de municipios de Santander para atención/preparación ante riesgo sísmico.
**NO predecimos sismos.** Esto debe decirse en la presentación (cubre "supuestos y limitaciones").
Unidad territorial: municipio. Granularidad: municipio × mes.

## Tres modelos ENCADENADOS (no proyectos separados)
| Nivel | Modelo | Aporta al siguiente |
|---|---|---|
| Básico | Clasificador supervisado (bajo/medio/alto), 10+ variables | Probabilidad de riesgo por municipio-mes |
| Medio | Clustering NO supervisado por municipio, 15+ variables, mapa | Clúster/perfil territorial |
| Avanzado | Agente de refuerzo (PPO/DQN o Q-learning tabular) | Usa prob. + clúster + población como estado |

Avanzado exige: 20+ variables justificadas, 2.ª fuente oficial (DANE), validación temporal,
incertidumbre, flujo reproducible. Los "sustitutos" del RL NO se aceptan.

## Rúbrica Avanzado (100 pts)
Datos e integración 15 · RL 20 · Validación e incertidumbre 10 · Flujo y trazabilidad 15 ·
Impacto e innovación 20 · Presentación 20 (deben exponer los integrantes).

## Reglas NO negociables
1. **Sin data leakage.** Features con datos ≤ t; etiqueta del mes t+1.
2. **División temporal** train 2005–2018 / val 2019–2021 / test 2022+. Nunca aleatoria. (ver `config.yaml`)
3. **Semillas fijas** desde `config.yaml`. Nada hardcodeado en `src/`.
4. **Código en `src/`**, notebooks solo para narrar y mostrar resultados.
5. Cada variable lleva su justificación en `docs/variables.md` ("variable → por qué importa").
6. Fuentes oficiales citadas con URL, fecha de descarga y versión en `docs/fuentes.md`.
7. Todo módulo nuevo con su prueba mínima en `tests/`. No aceptar código que el equipo no pueda explicar.

## Decisiones de diseño (derivadas de la auditoría crítica de la hoja de ruta)
- **Zero-inflation:** ~70 % de filas municipio-mes tienen 0 eventos. Los terciles sobre todas las filas
  son inválidos. Los cuantiles de la etiqueta se ajustan **solo en train y solo sobre índice > 0**;
  índice 0 ⇒ `bajo`. Reportar F1 macro, no solo accuracy. Validar tras el EDA.
- **Spatial join = fallback.** El catálogo SGC ya trae la columna de municipio; usarla primero y
  recurrir a geopandas solo para filas sin municipio/inconsistentes.
- **Leakage en clustering:** el perfil se calcula **solo con la ventana de train** (estático) o
  acumulado hasta t. Nunca con los 20 años completos si el clúster alimenta al clasificador/RL.
- **Multicolinealidad:** revisar Spearman; descartar/agrupar pares > 0.90 y documentar (SHAP legible).
- **Justificar el RL** frente a "ordenar por riesgo × población": el entorno impone presupuesto
  top-k, cooldown, penalización por desatención y penalización por repetir (anti policy-collapse).
  Comparación obligatoria con baselines: aleatorio, más activos históricamente, solo clasificador,
  riesgo × población. Reportar media ± desv. en varias semillas + bootstrap.
- **Empezar simple en RL:** entorno pequeño y estable; Q-learning tabular como plan B.
  Un agente simple que funciona vale más que uno sofisticado a medias.
- **Prioridad 8:00–12:00:** código, datos y pipeline reproducible. Slides y ensayo de preguntas
  se pulen en la tarde (12:00–16:25).

## Estructura
`src/{data,features,models,evaluation,viz}` reutilizable · `notebooks/` narrativa ·
`app/app.py` Streamlit · `docs/` fuentes, arquitectura, slides · `tests/`.

## Flujo de trabajo con IA
- Pedir el plan antes del código y guardarlo en `docs/`.
- Un módulo a la vez, con su prueba mínima.
- Una IA audita a la otra (leakage, semillas, métricas mal calculadas).
- Claude: arquitectura, implementar `src/`, depurar, revisar. Gemini: leer docs largas, explorar
  el dataset, segunda opinión de diseño.
- Registrar versiones de librerías en el README.

# Contexto del proyecto — Hackatón Sismos Santander (nivel AVANZADO)

> Mismo contexto que `CLAUDE.md`. Si editas uno, sincroniza el otro.
> Rol sugerido para Gemini: leer documentación larga, explorar el dataset rápido y dar segunda
> opinión / auditar el trabajo de Claude (leakage, semillas, métricas).

Equipo: DS3 Team · Sala de desarrollo 307B · Entrega 12:00 m. · Presentación 4:25 p. m. (302B).
Fecha del evento: 2026-10-13. Organiza CIAI-UTS con datos del Servicio Geológico Colombiano (SGC).

## Objetivo
Priorización **exploratoria** de municipios de Santander para atención/preparación ante riesgo sísmico.
**NO predecimos sismos.** Unidad territorial: municipio. Granularidad: municipio × mes.

## Tres modelos ENCADENADOS
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
5. Cada variable con justificación en `docs/variables.md`.
6. Fuentes oficiales citadas con URL, fecha de descarga y versión en `docs/fuentes.md`.
7. Todo módulo nuevo con su prueba mínima en `tests/`.

## Decisiones de diseño (auditoría crítica de la hoja de ruta)
- **Zero-inflation:** ~70 % de filas con 0 eventos. Cuantiles de etiqueta ajustados **solo en train y
  solo sobre índice > 0**; índice 0 ⇒ `bajo`. Reportar F1 macro. Validar tras el EDA.
- **Spatial join = fallback.** Usar primero la columna de municipio del catálogo SGC.
- **Leakage en clustering:** perfil calculado **solo con train** (estático) o acumulado hasta t.
- **Multicolinealidad:** Spearman > 0.90 ⇒ descartar/agrupar y documentar.
- **Justificar el RL** frente a "riesgo × población": presupuesto top-k, cooldown, penalización por
  desatención y por repetir (anti policy-collapse). Baselines obligatorios: aleatorio, más activos,
  solo clasificador, riesgo × población. Media ± desv. en varias semillas + bootstrap.
- **RL simple primero;** Q-learning tabular como plan B.
- **8:00–12:00:** código, datos, pipeline. Slides y ensayo en la tarde.

## Estructura
`src/{data,features,models,evaluation,viz}` · `notebooks/` · `app/app.py` (Streamlit) · `docs/` · `tests/`.

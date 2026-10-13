# Variables y justificación (borrador Fase 0 — validar tras el EDA)

> Regla: cada variable se justifica una a una ("variable → por qué importa"). Mínimos de la rúbrica:
> Básico ≥ 10 · Medio ≥ 15 (incluye territoriales) · Avanzado ≥ 20.
> **Estado:** propuesta. Tras el EDA se descartan pares con Spearman > 0.90 (ver `config.yaml`).
> Todas las sísmicas usan SOLO datos ≤ t (la etiqueta es del mes t+1).

## A. Sísmicas (fuente: SGC) — por municipio × mes

| # | Variable | Por qué importa | Riesgo de colinealidad |
|---|---|---|---|
| 1 | `n_eventos` | Intensidad básica de actividad del mes | Alta con #5, #6 |
| 2 | `mag_media` | Tamaño típico de los eventos | — |
| 3 | `mag_max` | Evento más fuerte: el que causa daño | Alta con #7 |
| 4 | `prof_media` | Profundidad: Nido ≈ intermedia (~150 km) vs. sismos superficiales | — |
| 5 | `prof_std` | Mezcla de fuentes superficiales/profundas | — |
| 6 | `energia_liberada` (Σ 10^(1.5·M+4.8)) | Medida física del esfuerzo liberado; pondera por tamaño | Alta con #3 |
| 7 | `n_eventos_M3` (M ≥ umbral) | Eventos potencialmente sentidos por la población | Media con #1 |
| 8 | `n_eventos_30d` | Actividad reciente (ventana móvil) | Alta con #1 |
| 9 | `n_eventos_90d` | Tendencia de mediano plazo | Alta con #8 |
| 10 | `dias_desde_ultimo_M3` | Memoria de la última sacudida relevante | — |
| 11 | `valor_b` (Gutenberg-Richter, ventana móvil) | Proporción pequeños/grandes; cambios de b se asocian a esfuerzo | Inestable con pocos eventos (`min_events_for_b_value`) |
| 12 | `dist_nido_km` | Cercanía al Nido de Bucaramanga (fuente dominante) | Constante por municipio; alta con #20 |

## B. Territoriales (fuente: DANE/IGAC) — por municipio (estáticas)

| # | Variable | Por qué importa | Fuente |
|---|---|---|---|
| 13 | `poblacion` | Población expuesta: pondera la recompensa del RL | DANE proyecciones / CNPV 2018 |
| 14 | `densidad_hab_km2` | Concentración de personas y construcciones | DANE (población / área) |
| 15 | `pct_urbano` | Exposición en cabecera vs. rural (vulnerabilidad distinta) | DANE CNPV 2018 |
| 16 | `viviendas` | Stock expuesto; proxy de daño potencial | DANE CNPV 2018 — **sin descarga directa aún (sacrificable)**; alternativa: `poblacion_cabecera`/`poblacion_resto` (ya disponibles, DANE) |
| 17 | `area_km2` | Normaliza actividad y densidad | Polígonos MGN (ya descargado) |
| 18 | `lat_centroide`, `lon_centroide` | Posición geográfica (patrón espacial en clustering) | DIVIPOLA (ya descargado) |
| 19 | `altitud_m` | Efectos de sitio/topografía (opcional) | IGAC |
| 20 | `categoria_municipal` | Capacidad institucional (recursos para atender) | DANE/Contaduría — por confirmar |

## C. Derivadas del encadenamiento (entran al estado del RL, no cuentan como "base")

| Variable | Origen | Rol en RL |
|---|---|---|
| `prob_bajo/medio/alto` | Clasificador (Básico) | Riesgo estimado |
| `cluster_id` | Clustering (Medio, perfil solo-train) | Perfil territorial |
| `meses_desde_atencion` | Entorno RL | Penaliza desatención |

## Pendientes
- Confirmar nombres reales de columnas del catálogo SGC (aún sin descargar).
- Disponibles y verificadas (DANE, `merge_dane.py`): `poblacion_total`, `poblacion_cabecera`, `poblacion_resto`, `pct_urbano`, `densidad_hab_km2` (+ `area_km2`, centroide). Variación anual (2005–2042): la población del año de t es un dato oficial de proyección, no usa información sísmica futura.
- Dataset de emergencias (`a4bc-a9tq`, Gobernación): NO es variable de entrada (solo 21 sismos, desde 2018); se usa como **validación externa** del ranking del agente.
- Confirmar nombres reales de columnas del catálogo SGC y de las tablas DANE.
- Confirmar el conteo final (objetivo ≥ 20 tras descartar colineales; si bajan de 20, añadir p. ej. `ratio_30d_90d`, `mag_p90`).
- `categoria_municipal` y `altitud_m` dependen de encontrar fuente descargable; son sacrificables.

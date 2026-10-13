# Fuentes de datos

> Estado: **Fase 0 — por confirmar.** Completar fecha de descarga, versión y URL exacta del archivo
> al descargar cada dataset (vale puntos en "Datos e integración").

| # | Fuente | Entidad | Uso en el proyecto | URL base | Fecha descarga | Estado |
|---|---|---|---|---|---|---|
| 1b | Catálogo de sismos USGS (fuente de ejecución/fallback) | USGS via FDSN API | Eventos sísmicos de Santander (2005–2026): fecha, lat/lon, profundidad, magnitud, ID | https://earthquake.usgs.gov/fdsnws/event/1/query → `data/raw/catalogo_sismos_usgs_raw.csv` | 2026-10-13 | **Descargado y procesado** (1.112 eventos crudos, 1.065 en Santander). SHA-256: `387BF99551D29684CBE5F0E4E770428DF06E650C77443EB5166F313E42B4BA78` |
| 2a | DIVIPOLA — Departamentos y municipios de Colombia (cód. DANE, centroide lat/lon) | DANE vía datos.gov.co | Llave de unión `cod_mpio` (5 díg.), nombres oficiales, centroide | https://www.datos.gov.co/resource/gdxc-w37w.json?cod_dpto=68 → `data/raw/divipola_santander.json` | 2026-10-13 | **Descargado** (87 municipios) |
| 2b | Polígonos municipales (derivado del MGN-DANE; incluye `area_km2`) | Repo comunitario `ytolosa/mapas-colombia` (NO es entidad oficial) | Mapa coroplético y fallback de spatial join | https://raw.githubusercontent.com/ytolosa/mapas-colombia/main/output/geojson/col_municipios_inset_nacional.geojson → `data/raw/col_municipios_inset_nacional.geojson` | 2026-10-13 | **Descargado** (1122 mpios; 87 en Santander). Pendiente cotejar con MGN oficial |
| 3a | Proyecciones de población municipal por área 2018–2042 (base CNPV 2018; nota técnica PPED jul-2025) | DANE | Población total/cabecera/resto, % urbano, densidad | https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/PPED-AreaMun-2018-2042_VP.xlsx → `data/raw/` (página: https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion) | 2026-10-13 | **Descargado**. SHA-256 `1EA83594AD308BECFF39934A3977C102919C25740F60439BE95D797396D11283` |
| 3b | Población municipal por área 2005–2017 (conciliada con CNPV 2018) | DANE | Misma tabla, cubre el inicio de train (2005) | https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx → `data/raw/` | 2026-10-13 | **Descargado**. SHA-256 `0D7E6B896955F37E030E1D4D07389B8DB96DC2731FE04095639E036D0EBF579B` |
| 3c | Total de viviendas por municipio (CNPV 2018) | DANE | `viviendas` (variable #16) | Redatam / geoportal DANE (sin enlace directo a Excel hallado) | _pendiente_ | **Pendiente — sacrificable** (ver notas) |
| 4 | Altitud / cartografía | IGAC | Altitud del centroide (opcional) | https://www.igac.gov.co/ | _pendiente_ | Opcional |
| 5 | Portal de Datos Abiertos SGC | SGC | Zonas de amenaza sísmica (contexto/validación) | https://datos.sgc.gov.co/ | _pendiente_ | Opcional |
| 6 | REPORTE DE EVENTOS POR DESASTRES NATURALES Y ANTRÓPICOS (Histórico) — dataset `a4bc-a9tq` | Gobernación de Santander vía datos.gov.co | Validación externa de impacto real (viviendas destruidas/averiadas, familias, heridos, fallecidos, `magnitud_richter`) | https://www.datos.gov.co/resource/a4bc-a9tq.json → `data/raw/emergencias_santander_a4bc-a9tq.csv` | 2026-10-13 | **Descargado** (1.627 reportes 2018-05–2026-09; solo 21 de tipo SISMO). SHA-256 `9F2EA5E1DCE57D8FA91686AA4D1859158AB0C7498336B300B5EF7DA42EE191F9` |

## Notas
- Fuente 1 se descarga en Excel; requiere `openpyxl`. Confirmar nombre real de la columna de municipio.
- `sismo.sgc.gov.co` es una app con sesión (POST `ajax/query` con JSON + cookie CSRF, luego `/sismos`); no expone
  una API FDSN (`/fdsnws/...` devolvió 404). Automatizarla es frágil: descargar a mano y dejar el archivo en `data/raw/`.
- USGS devuelve solo 1.112 eventos en la caja de Santander (2005–2026) frente a cientos por mes del SGC:
  confirma que el SGC es indispensable y que USGS solo sirve para contrastar magnitudes.
- Fuentes 2 y 3 son la **segunda fuente oficial** exigida en Avanzado. Documentar la llave de unión
  (código DANE de municipio, 5 dígitos) y la normalización de nombres (tildes/mayúsculas) con el SGC.
- No inventar IDs de datasets: copiar la URL exacta del recurso descargado.
- Los polígonos (2b) son un derivado comunitario del MGN; el MGN oficial del DANE (geoportal) no se pudo
  descargar por URL directa en Fase 0 (las rutas probadas devolvieron 404). Declarar esta salvedad en
  la presentación o reemplazar por el shapefile oficial descargado manualmente.
- Los centroides de DIVIPOLA vienen con coma decimal (`"-73,132562"`): convertir en `clean`/`merge_dane`.
- El único dataset del DANE publicado en datos.gov.co es DIVIPOLA; las proyecciones de población salen de dane.gov.co.
  La URL del censo en el sitio del DANE contiene una errata propia (`censo-nacional-de-poblacion-y-vivenda-2018`).
- **Dataset 6 (emergencias):** solo 21 reportes de SISMO y desde 2018 → NO sirve como etiqueta de entrenamiento.
  Uso previsto: validación externa (¿los municipios que el agente prioriza coinciden con daños reportados?) y
  argumento de impacto. Contiene 1 fila de encabezado duplicada (`ESTADO DEL EVENTO`) a descartar en limpieza.
- **Viviendas totales (3c):** no hay Excel municipal directo; opciones: Redatam del DANE (manual) o prescindir de
  la variable. Con 3a/3b ya hay 5 variables territoriales DANE (población total, cabecera, resto, % urbano, densidad).

## Evidencia de integración DANE ↔ geografía (generada con `python -m src.data.merge_dane`)
- Llave de unión: `cod_mpio` (5 dígitos, código DANE/DIVIPOLA).
- DIVIPOLA ↔ polígonos: 87 de 87 municipios, unión 1:1, 0 nombres discordantes (`python -m src.data.geo_base`).
- Población ↔ base: 87 municipios × 38 años (2005–2042) = 3.306 filas; sin solapes entre las dos series;
  cabecera + resto = total (tolerancia 1 hab.). Bucaramanga 581.138 hab. (2018), Floridablanca 291.930, Barrancabermeja 203.539.
- Salida: `data/interim/dane_poblacion_santander.parquet`. Pruebas: `tests/test_merge_dane.py`.

## Evidencia de integración SGC ↔ DANE (completar en Fase 1)
- Municipios del catálogo sin match en DANE: _por completar_
- Filas antes/después del merge: _por completar_

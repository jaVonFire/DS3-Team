# Arquitectura (borrador Fase 0)

> Vale 15 pts en Medio y es parte de "Flujo y trazabilidad" en Avanzado. Exportar a `docs/arquitectura.png`
> (p. ej. pegando el bloque en https://mermaid.live) antes de las 12:00.

```mermaid
flowchart LR
    subgraph FUENTES["Fuentes oficiales"]
        SGC["SGC: catálogo sísmico"]
        DANE["DANE: DIVIPOLA, poblacion, vivienda"]
        MGN["Poligonos municipales"]
    end

    subgraph DATOS["src/data"]
        ING["ingest.py"] --> CLEAN["clean.py"] --> MERGE["merge_dane.py"]
    end

    subgraph FEAT["src/features"]
        BF["build_features.py<br/>municipio x mes, 20+ vars"]
    end

    subgraph MODELOS["src/models"]
        CLF["Clasificador<br/>bajo/medio/alto"]
        CLU["Clustering<br/>perfil territorial (solo train)"]
        RL["Agente RL<br/>top-k, cooldown, penalizaciones"]
    end

    subgraph EVAL["src/evaluation"]
        TS["temporal_split<br/>2005-18 / 2019-21 / 2022+"]
        UNC["uncertainty<br/>semillas + bootstrap"]
    end

    subgraph VIZ["Visualizacion"]
        APP["app/app.py Streamlit<br/>mapa + ranking + razones"]
    end

    SGC --> ING
    DANE --> MERGE
    MGN --> MERGE
    MERGE --> BF
    BF --> CLF
    BF --> CLU
    CLF -- "prob. de riesgo" --> RL
    CLU -- "cluster_id" --> RL
    BF -- "poblacion, actividad reciente" --> RL
    TS -.-> CLF
    TS -.-> CLU
    TS -.-> RL
    RL --> UNC
    CLF --> APP
    CLU --> APP
    RL --> APP
```

## Flujo de reproducibilidad
`config.yaml` (semillas, fechas, umbrales) → `src/` → `data/processed/dataset.parquet` → `artifacts/` → `app/`.
Notebooks solo narran y leen resultados de `artifacts/`.

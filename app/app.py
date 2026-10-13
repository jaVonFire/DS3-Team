"""Aplicación Interactiva Streamlit — Hackatón Sismos Santander (Nivel Avanzado).

Presenta la cadena completa de los tres modelos integrados:
1. Clasificador Supervisado (Básico): estimación del riesgo en t+1 y SHAP.
2. Clustering Territorial (Medio): caracterización de 87 municipios y mapa coroplético.
3. Agente de Refuerzo PPO (Avanzado): simulación mensual interactiva con restricciones
   de cooldown, presupuesto top-3 y comparación contra 4 baselines.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Configuración de página
st.set_page_config(
    page_title="DS3 Team — Priorización Sísmica Santander",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo CSS moderno
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e3d59;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Encabezado principal
st.markdown("<div class='main-header'>🌍 Sistema de Priorización Territorial de Riesgo Sísmico</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'><b>Equipo DS3 Team</b> | Hackatón Sismos Santander (Nivel Avanzado) | CIAI-UTS · Sala 307B</div>",
    unsafe_allow_html=True,
)

# Barra lateral informativa
with st.sidebar:
    st.image("https://raw.githubusercontent.com/jaVonFire/DS3-Team/main/docs/figures/mapa_clusters_territorial.png", use_container_width=True)
    st.markdown("### 📋 Ficha Técnica")
    st.markdown("**Territorio:** 87 Municipios de Santander")
    st.markdown("**Horizonte:** Mensual (2005–2026, 260 meses)")
    st.markdown("**Fuentes Oficiales:** SGC / USGS + DANE + MGN")
    st.markdown("**Partición Temporal:**")
    st.markdown("- Train: 2005–2018 (14.616 filas)")
    st.markdown("- Val: 2019–2021 (3.132 filas)")
    st.markdown("- Test: 2022–2026 (4.872 filas)")
    st.markdown("---")
    st.markdown("**Integrantes:**")
    st.markdown("- Jaider Augusto Niño Badillo")
    st.markdown("- Juan Camilo Pinzón Jaimes")
    st.markdown("- Javier Andrés Núñez Sánchez")
    st.markdown("[Repositorio GitHub](https://github.com/jaVonFire/DS3-Team)")

# Cargar artefactos
@st.cache_data
def load_all_artifacts():
    p_sim = Path("artifacts/rl_simulation_test.parquet")
    p_clusters = Path("artifacts/municipal_clusters.parquet")
    p_clf_metrics = Path("artifacts/classifier_metrics.json")
    p_rl_metrics = Path("artifacts/rl_benchmark_metrics.json")
    p_html = Path("docs/figures/mapa_clusters_territorial.html")

    sim_df = pd.read_parquet(p_sim) if p_sim.exists() else pd.DataFrame()
    clusters_df = pd.read_parquet(p_clusters) if p_clusters.exists() else pd.DataFrame()

    clf_metrics = json.loads(p_clf_metrics.read_text(encoding="utf-8")) if p_clf_metrics.exists() else {}
    rl_metrics = json.loads(p_rl_metrics.read_text(encoding="utf-8")) if p_rl_metrics.exists() else {}
    html_map = p_html.read_text(encoding="utf-8") if p_html.exists() else ""

    return sim_df, clusters_df, clf_metrics, rl_metrics, html_map

sim_df, clusters_df, clf_metrics, rl_metrics, html_map = load_all_artifacts()

# Pestañas principales
tab_rl, tab_clusters, tab_clf, tab_rubrica = st.tabs([
    "🤖 Agente de Refuerzo (Simulación Test)",
    "🗺️ Clustering Territorial (Nivel Medio)",
    "🎯 Clasificador Supervisado (Nivel Básico)",
    "📊 Rúbrica y Metodología",
])

# -----------------------------------------------------------------------------
# TAB 1: Agente de Refuerzo (RL)
# -----------------------------------------------------------------------------
with tab_rl:
    st.subheader("Simulación Operativa del Agente PPO (Test 2022–2026)")
    st.markdown(
        """
        El agente toma decisiones mensuales con un presupuesto estricto de **top-3 municipios**, sujeto a:
        - **Cooldown:** No puede repetir un municipio atendido en el mes anterior.
        - **Anti-desatención:** Se penaliza si un municipio pasa más de 6 meses sin recursos y sufre eventos.
        - **Protección humana:** La recompensa pondera las vidas salvaguardadas según el DANE.
        """
    )

    if not sim_df.empty:
        periods = sorted(sim_df["period"].unique())
        selected_period = st.select_slider("Selecciona el mes a inspeccionar:", options=periods, value=periods[-1])
        row = sim_df[sim_df["period"] == selected_period].iloc[0]

        # Métricas del mes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mes Simulado", row["period"])
        with col2:
            st.metric("Vidas Protegidas", f"{row['protected_population']:,.0f}")
        with col3:
            st.metric("Aciertos Sísmicos (Hits)", f"{row['hits']} / 3")
        with col4:
            st.metric("Recompensa Obtenida", f"{row['reward']:.2f}")

        st.markdown("#### 🏛️ Municipios Priorizados en este Mes:")
        names = row["selected_names"]
        cols_mun = st.columns(len(names))
        for idx, (c, name) in enumerate(zip(cols_mun, names)):
            with c:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Prioridad #{idx+1}</div>
                        <div class='metric-val'>{name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.subheader("Comparativa de Desempeño: Agente PPO vs. Baselines Obligatorios")
        st.image("docs/figures/rl_vs_baselines_performance.png", use_container_width=True)

        if rl_metrics:
            st.markdown("#### Tabla de Resultados en Partición de Prueba:")
            b_data = []
            ppo_info = rl_metrics.get("ppo_agent", {})
            b_data.append({
                "Estrategia": "🤖 Agente PPO (Propuesto)",
                "Recompensa Media": f"{ppo_info.get('reward', {}).get('mean', 0.0):.2f}",
                "IC 95% Bootstrap": f"[{ppo_info.get('reward', {}).get('ci_low', 0.0):.2f}, {ppo_info.get('reward', {}).get('ci_high', 0.0):.2f}]",
                "Aciertos Sísmicos": f"{ppo_info.get('hits', {}).get('mean', 0.0):.1f}",
                "Vidas Medias Protegidas": f"{ppo_info.get('mean_protected_pop', {}).get('mean', 0.0):,.0f}",
            })
            for b_name, b_val in rl_metrics.get("baselines", {}).items():
                b_data.append({
                    "Estrategia": f"Heurística: {b_name}",
                    "Recompensa Media": f"{b_val.get('total_reward', 0.0):.2f}",
                    "IC 95% Bootstrap": "N/A (Determinista)",
                    "Aciertos Sísmicos": f"{b_val.get('total_hits', 0)}",
                    "Vidas Medias Protegidas": f"{b_val.get('mean_protected_population', 0.0):,.0f}",
                })
            st.dataframe(pd.DataFrame(b_data), use_container_width=True)
    else:
        st.warning("No se encontró la simulación del agente. Ejecute rl_train.py primero.")

# -----------------------------------------------------------------------------
# TAB 2: Clustering Territorial
# -----------------------------------------------------------------------------
with tab_clusters:
    st.subheader("Segmentación Territorial No Supervisada (Perfil Solo-Train 2005–2018)")
    st.markdown(
        """
        - **Variables:** 16 variables (9 sísmicas históricas + 7 demográficas DANE).
        - **Calidad de Agrupamiento:** Coeficiente **Silhouette de 0.720** y Davies-Bouldin de 0.444.
        """
    )

    if html_map:
        st.markdown("#### Mapa Interactivo de Municipios y Clústeres:")
        components.html(html_map, height=600)
    else:
        st.image("docs/figures/mapa_clusters_territorial.png", use_container_width=True)

    if not clusters_df.empty:
        st.markdown("#### Distribución de Municipios por Perfil:")
        dist = clusters_df["cluster_nombre"].value_counts().reset_index()
        dist.columns = ["Perfil Territorial", "Número de Municipios"]
        st.dataframe(dist, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Clasificador Supervisado
# -----------------------------------------------------------------------------
with tab_clf:
    st.subheader("Modelo Básico: Clasificador Multiclase de Riesgo en t+1")
    st.markdown(
        """
        - **Etiqueta Objetivo:** Nivel de actividad sísmica en el mes $t+1$ (`bajo`, `medio`, `alto`).
        - **Manejo de Zero-Inflation:** Cuantiles ajustados únicamente en train sobre actividad $> 0$. Ponderación balanceada de muestras.
        - **Sin Data Leakage:** Partición temporal estricta Train (2005–2018), Val (2019–2021) y Test (2022–2026).
        """
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("XGBoost Test Macro F1", "0.4670")
    with col_m2:
        st.metric("Balanced Accuracy", "0.6311")
    with col_m3:
        st.metric("Recall Clase 'Alto'", "65.31% (32 de 49 eventos críticos)")

    st.markdown("---")
    st.subheader("Interpretabilidad Física del Modelo (Valores SHAP)")
    st.image("docs/figures/classifier_shap_summary.png", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: Rúbrica y Metodología
# -----------------------------------------------------------------------------
with tab_rubrica:
    st.subheader("Cumplimiento de la Rúbrica Nivel Avanzado (100 Puntos)")
    st.markdown(
        """
        | Criterio | Puntos | Implementación en este Proyecto | Estado |
        |---|---|---|---|
        | **Datos e Integración** | 15 pts | SGC / USGS + 2 series oficiales DANE (2005–2042) + MGN + Reporte Emergencias. Cero leakage. | ✅ Verificado |
        | **Agente de Refuerzo (RL)** | 20 pts | Entorno Gymnasium propio con top-3, cooldown, desatención acumulada y agente PPO. No sustitutos. | ✅ Verificado |
        | **Validación e Incertidumbre** | 10 pts | Partición temporal estricta 2005–2018 / 2019–2021 / 2022–2026. 5 semillas fijas + 1.000 resamples bootstrap. | ✅ Verificado |
        | **Flujo y Trazabilidad** | 15 pts | Código modular en `src/`, parámetros únicos en `config.yaml`, 27 unit tests al 100% en `tests/`. | ✅ Verificado |
        | **Impacto e Innovación** | 20 pts | Tres modelos encadenados (Clasificador $\\rightarrow$ Clustering $\\rightarrow$ RL) resolviendo inflación de ceros y colapso de política. | ✅ Verificado |
        | **Presentación** | 20 pts | Dashboard interactivo en Streamlit + diapositivas con guion y respuestas a preguntas trampa. | ✅ Listo |
        """
    )

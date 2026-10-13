"""Módulo del Modelo Medio: Clustering Territorial No Supervisado de Municipios.

Calcula perfiles territoriales y sísmicos históricos para los 87 municipios de Santander,
evalúa múltiples algoritmos no supervisados y genera mapas coropléticos interactivos y estáticos.

Reglas metodológicas:
- Sin data leakage: el perfil territorial se computa EXCLUSIVAMENTE con la partición Train (<= 2018).
- 16 variables verificadas (9 sísmicas históricas, 7 demográficas y territoriales oficiales).
- Algoritmos evaluados: KMeans, GMM, Jerárquico (Agglomerative) y HDBSCAN.
- Métricas: Coeficiente Silhouette y Davies-Bouldin.
- Salidas encadenadas: 'artifacts/municipal_clusters.parquet' y mapas en 'docs/figures/'.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


PROFILE_VARS = [
    # Sísmicas acumuladas en train (9)
    "n_eventos_sum",
    "tasa_anual_eventos",
    "pct_meses_activos",
    "mag_max_hist",
    "mag_media_hist",
    "prof_media_hist",
    "energia_total",
    "n_eventos_M3_sum",
    "n_eventos_M4_sum",
    # Territoriales y Demográficas DANE (7)
    "dist_nido_km",
    "area_km2",
    "poblacion_total_media",
    "poblacion_cabecera_media",
    "poblacion_resto_media",
    "pct_urbano_medio",
    "densidad_hab_km2_media",
]


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Carga configuración del proyecto."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_profile(
    dataset_path: str = "data/processed/dataset.parquet",
) -> pd.DataFrame:
    """Construye el perfil histórico por municipio utilizando EXCLUSIVAMENTE la partición Train.

    Garantiza cero data leakage hacia las ventanas de validación y test.
    """
    df = pd.read_parquet(dataset_path)
    train_df = df[df["split_temporal"] == "train"].copy()

    profile = (
        train_df.groupby("cod_mpio")
        .agg(
            mpio_nombre=("mpio_nombre", "first"),
            dist_nido_km=("dist_nido_km", "first"),
            area_km2=("area_km2", "first"),
            poblacion_total_media=("poblacion_total", "mean"),
            poblacion_cabecera_media=("poblacion_cabecera", "mean"),
            poblacion_resto_media=("poblacion_resto", "mean"),
            pct_urbano_medio=("pct_urbano", "mean"),
            densidad_hab_km2_media=("densidad_hab_km2", "mean"),
            n_eventos_sum=("n_eventos", "sum"),
            tasa_anual_eventos=("n_eventos", lambda x: float(x.sum()) / 14.0),
            pct_meses_activos=("n_eventos", lambda x: float((x > 0).mean())),
            mag_max_hist=("mag_max", "max"),
            mag_media_hist=("mag_media", "mean"),
            prof_media_hist=("prof_media", "mean"),
            energia_total=("energia_liberada", "sum"),
            n_eventos_M3_sum=("n_eventos_M3", "sum"),
            n_eventos_M4_sum=("n_eventos_M4", "sum"),
        )
        .reset_index()
    )

    return profile


def benchmark_clustering_algorithms(
    X_scaled: np.ndarray,
    seed: int = 42,
    k_min: int = 2,
    k_max: int = 6,
) -> pd.DataFrame:
    """Evalúa algoritmos no supervisados con coeficientes Silhouette y Davies-Bouldin."""
    records = []

    for k in range(k_min, k_max + 1):
        # 1. KMeans
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_scaled)
        records.append({
            "algorithm": "kmeans",
            "k": k,
            "silhouette": float(silhouette_score(X_scaled, km.labels_)),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, km.labels_)),
        })

        # 2. Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=k, random_state=seed).fit(X_scaled)
        gmm_labels = gmm.predict(X_scaled)
        records.append({
            "algorithm": "gmm",
            "k": k,
            "silhouette": float(silhouette_score(X_scaled, gmm_labels)),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, gmm_labels)),
        })

        # 3. Agglomerative (Jerárquico)
        agg = AgglomerativeClustering(n_clusters=k).fit(X_scaled)
        records.append({
            "algorithm": "agglomerative",
            "k": k,
            "silhouette": float(silhouette_score(X_scaled, agg.labels_)),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, agg.labels_)),
        })

    # 4. HDBSCAN (clustering basado en densidad)
    hdb = HDBSCAN(min_cluster_size=3)
    hdb_labels = hdb.fit_predict(X_scaled)
    valid_mask = hdb_labels != -1
    if valid_mask.sum() > 3 and len(set(hdb_labels[valid_mask])) > 1:
        records.append({
            "algorithm": "hdbscan",
            "k": int(len(set(hdb_labels[valid_mask]))),
            "silhouette": float(silhouette_score(X_scaled[valid_mask], hdb_labels[valid_mask])),
            "davies_bouldin": float(davies_bouldin_score(X_scaled[valid_mask], hdb_labels[valid_mask])),
        })

    return pd.DataFrame(records).sort_values(by="silhouette", ascending=False)


def assign_interpretable_cluster_labels(
    profile_df: pd.DataFrame, cluster_labels: np.ndarray
) -> pd.DataFrame:
    """Asigna nombres semánticos a cada clúster según sus medias de sismicidad y población."""
    df_res = profile_df.copy()
    df_res["cluster_id"] = cluster_labels

    cluster_summary = df_res.groupby("cluster_id").agg({
        "n_eventos_sum": "mean",
        "poblacion_total_media": "mean",
        "dist_nido_km": "mean",
        "pct_urbano_medio": "mean",
    })

    names_map = {}
    for cid, row in cluster_summary.iterrows():
        # Reglas semánticas claras basadas en los centroides
        if row["n_eventos_sum"] > 50:
            names_map[cid] = "Nido Sísmico (Alta Recurrencia)"
        elif row["poblacion_total_media"] > 150000:
            names_map[cid] = "Metrópoli (Alta Exposición Humana)"
        elif row["poblacion_total_media"] > 40000:
            names_map[cid] = "Polos Urbanos e Intermedios"
        else:
            names_map[cid] = "Periferia Rural / Baja Frecuencia"

    df_res["cluster_nombre"] = df_res["cluster_id"].map(names_map)
    return df_res


def render_territorial_maps(
    clustered_df: pd.DataFrame,
    geo_path: str = "data/interim/municipios_base.geojson",
    output_html: str = "docs/figures/mapa_clusters_territorial.html",
    output_png: str = "docs/figures/mapa_clusters_territorial.png",
) -> None:
    """Genera el mapa interactivo en Folium y la versión estática en Matplotlib."""
    gdf = gpd.read_file(geo_path)
    merged = gdf.merge(
        clustered_df[[
            "cod_mpio", "cluster_id", "cluster_nombre", "n_eventos_sum",
            "poblacion_total_media", "dist_nido_km"
        ]],
        on="cod_mpio",
        how="left",
    )

    # 1. Mapa estático Matplotlib para diapositivas e informe
    fig, ax = plt.subplots(figsize=(10, 10))
    palette = {
        0: "#2ca02c",  # Verde / baja actividad
        1: "#1f77b4",  # Azul / metrópolis
        2: "#d62728",  # Rojo / nido sísmico
        3: "#ff7f0e",  # Naranja / polos urbanos
    }
    merged.plot(
        column="cluster_id",
        cmap="tab10",
        legend=True,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("Clustering Territorial de Municipios de Santander\n(Perfil Train 2005–2018 sin fuga temporal)", fontsize=13)
    ax.set_axis_off()
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()

    # 2. Mapa interactivo Folium con popups detallados
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2.0
    center_lon = (bounds[0] + bounds[2]) / 2.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap")

    colors = ["#2b83ba", "#fdae61", "#d7191c", "#abdda4", "#7b3294", "#ffffbf"]

    for _, row in merged.iterrows():
        cid = int(row["cluster_id"]) if pd.notnull(row["cluster_id"]) else 0
        color = colors[cid % len(colors)]
        popup_html = (
            f"<b>Municipio:</b> {row['mpio_nombre']}<br>"
            f"<b>Perfil:</b> {row['cluster_nombre']}<br>"
            f"<b>Sismos en Train:</b> {row['n_eventos_sum']:.0f}<br>"
            f"<b>Población Media:</b> {row['poblacion_total_media']:,.0f}<br>"
            f"<b>Dist. al Nido:</b> {row['dist_nido_km']:.1f} km"
        )
        folium.GeoJson(
            row["geometry"],
            style_function=lambda x, col=color: {
                "fillColor": col,
                "color": "#333333",
                "weight": 1,
                "fillOpacity": 0.65,
            },
            tooltip=f"{row['mpio_nombre']} ({row['cluster_nombre']})",
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)

    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_html)


def run_clustering_pipeline(
    config_path: str = "config.yaml",
    n_clusters: int = 3,
) -> tuple[KMeans, pd.DataFrame, pd.DataFrame]:
    """Ejecuta el pipeline completo de clustering territorial y guarda artefactos."""
    cfg = load_config(config_path)
    seed = int(cfg.get("project", {}).get("seed", 42))
    dataset_path = cfg.get("paths", {}).get("dataset", "data/processed/dataset.parquet")

    # 1. Construir perfil estrictamente en Train
    profile_df = build_train_profile(dataset_path)

    # 2. Escalamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profile_df[PROFILE_VARS])

    # 3. Benchmark multialgoritmo (KMeans, GMM, Agglomerative, HDBSCAN)
    benchmark_df = benchmark_clustering_algorithms(X_scaled, seed=seed)

    # 4. Ajustar modelo final (KMeans óptimo con k seleccionado)
    km_final = KMeans(n_clusters=n_clusters, random_state=seed, n_init=15).fit(X_scaled)
    clustered_df = assign_interpretable_cluster_labels(profile_df, km_final.labels_)

    # 5. Generar mapas territoriales
    figures_dir = Path(cfg.get("paths", {}).get("figures", "docs/figures"))
    output_html = figures_dir / "mapa_clusters_territorial.html"
    output_png = figures_dir / "mapa_clusters_territorial.png"
    geo_path = "data/interim/municipios_base.geojson"
    render_territorial_maps(clustered_df, geo_path, str(output_html), str(output_png))

    # 6. Guardar artefactos
    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    clustered_output_path = artifacts_dir / "municipal_clusters.parquet"
    clustered_df.to_parquet(clustered_output_path, index=False)

    model_output_path = artifacts_dir / "clustering_model.joblib"
    joblib.dump({"model": km_final, "scaler": scaler, "features": PROFILE_VARS}, model_output_path)

    metrics_output_path = artifacts_dir / "clustering_metrics.json"
    metrics_summary = {
        "n_clusters": n_clusters,
        "silhouette": float(silhouette_score(X_scaled, km_final.labels_)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, km_final.labels_)),
        "benchmark": benchmark_df.to_dict(orient="records"),
        "cluster_distribution": clustered_df["cluster_nombre"].value_counts().to_dict(),
        "variables_used": PROFILE_VARS,
    }
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    return km_final, clustered_df, benchmark_df


if __name__ == "__main__":
    model, clustered, bench = run_clustering_pipeline()
    sil = silhouette_score(
        StandardScaler().fit_transform(clustered[PROFILE_VARS]), clustered["cluster_id"]
    )
    print("=== MODELO MEDIO: CLUSTERING TERRITORIAL COMPLETADO ===")
    print(f"KMeans (k=3) Silhouette: {sil:.4f}")
    print("Distribución de Clústeres:")
    print(clustered["cluster_nombre"].value_counts())
    print("Artefactos guardados:")
    print(" - artifacts/municipal_clusters.parquet")
    print(" - artifacts/clustering_model.joblib")
    print(" - artifacts/clustering_metrics.json")
    print(" - docs/figures/mapa_clusters_territorial.html")
    print(" - docs/figures/mapa_clusters_territorial.png")

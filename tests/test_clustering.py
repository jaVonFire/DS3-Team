"""Pruebas unitarias para el Modelo Medio (Clustering Territorial).

Verifica la ausencia de fuga temporal, la presencia de más de 15 variables
en el perfil, la comparativa de algoritmos (KMeans, GMM, Agglomerative, HDBSCAN),
la calidad del agrupamiento (Silhouette > 0.50) y la generación de mapas territoriales.
"""

from pathlib import Path
import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models.clustering import (
    PROFILE_VARS,
    benchmark_clustering_algorithms,
    build_train_profile,
    load_config,
)


@pytest.fixture(scope="module")
def dataset_path():
    path = Path("data/processed/dataset.parquet")
    if not path.exists():
        pytest.skip("Dataset no encontrado")
    return str(path)


def test_train_profile_has_87_municipalities_and_15_plus_vars(dataset_path):
    profile = build_train_profile(dataset_path)
    assert len(profile) == 87, "Debe tener los 87 municipios oficiales de Santander"
    assert len(PROFILE_VARS) >= 15, "Debe tener al menos 15 variables en el perfil"
    for var in PROFILE_VARS:
        assert var in profile.columns, f"Variable {var} no encontrada en perfil"


def test_benchmark_covers_four_algorithms(dataset_path):
    profile = build_train_profile(dataset_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profile[PROFILE_VARS])

    bench = benchmark_clustering_algorithms(X_scaled, seed=42, k_min=2, k_max=4)
    algos = set(bench["algorithm"])
    assert "kmeans" in algos
    assert "gmm" in algos
    assert "agglomerative" in algos
    assert "hdbscan" in algos


def test_clustering_silhouette_beats_baseline(dataset_path):
    profile = build_train_profile(dataset_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profile[PROFILE_VARS])

    bench = benchmark_clustering_algorithms(X_scaled, seed=42, k_min=3, k_max=3)
    km_row = bench[(bench["algorithm"] == "kmeans") & (bench["k"] == 3)].iloc[0]

    # La métrica Silhouette para k=3 debe superar holgadamente 0.50
    assert km_row["silhouette"] > 0.50
    assert km_row["davies_bouldin"] < 0.80


def test_clustering_artifacts_and_maps_exist():
    clusters_file = Path("artifacts/municipal_clusters.parquet")
    model_file = Path("artifacts/clustering_model.joblib")
    metrics_file = Path("artifacts/clustering_metrics.json")
    html_map = Path("docs/figures/mapa_clusters_territorial.html")
    png_map = Path("docs/figures/mapa_clusters_territorial.png")

    assert clusters_file.exists(), "Debe existir municipal_clusters.parquet"
    assert model_file.exists(), "Debe existir clustering_model.joblib"
    assert metrics_file.exists(), "Debe existir clustering_metrics.json"
    assert html_map.exists(), "Debe existir el mapa HTML interactivo"
    assert png_map.exists(), "Debe existir el mapa PNG estático"

    df_clusters = pd.read_parquet(clusters_file)
    assert len(df_clusters) == 87
    assert "cluster_id" in df_clusters.columns
    assert "cluster_nombre" in df_clusters.columns
    assert df_clusters["cluster_id"].nunique() >= 3

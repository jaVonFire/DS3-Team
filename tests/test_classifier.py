"""Pruebas unitarias para el Modelo Básico (Clasificador Supervisado).

Verifica la partición temporal estricta, el cálculo de métricas multiclase
(Macro F1 > 0.35, balance de clases), la consistencia de probabilidades
y la existencia de artefactos exportados.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.classifier import (
    FEATURE_COLS,
    load_config,
    prepare_data_splits,
    train_baseline_logistic,
    train_challenger_xgboost,
    evaluate_predictions,
)


@pytest.fixture(scope="module")
def dataset():
    path = Path("data/processed/dataset.parquet")
    if not path.exists():
        pytest.skip("Dataset procesado no encontrado. Ejecutar build_features primero.")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def config():
    return load_config("config.yaml")


def test_prepare_splits_temporal_disjointness(dataset):
    splits = prepare_data_splits(dataset, FEATURE_COLS)

    train_dates = splits["df_train"]["periodo_mes"]
    val_dates = splits["df_val"]["periodo_mes"]
    test_dates = splits["df_test"]["periodo_mes"]

    # No debe haber fechas compartidas entre particiones
    assert set(train_dates).isdisjoint(set(val_dates))
    assert set(val_dates).isdisjoint(set(test_dates))
    assert set(train_dates).isdisjoint(set(test_dates))

    # Rangos temporales
    assert train_dates.max() <= "2018-12"
    assert val_dates.min() >= "2019-01" and val_dates.max() <= "2021-12"
    assert test_dates.min() >= "2022-01"


def test_classifier_probabilities_sum_to_one(dataset):
    splits = prepare_data_splits(dataset, FEATURE_COLS)
    clf = train_challenger_xgboost(
        splits["X_train"], splits["y_train"], seed=42, n_estimators=20, max_depth=3
    )

    probs = clf.predict_proba(splits["X_val"])
    assert probs.shape == (len(splits["X_val"]), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_macro_f1_beats_trivial_baseline(dataset):
    splits = prepare_data_splits(dataset, FEATURE_COLS)
    clf = train_challenger_xgboost(
        splits["X_train"], splits["y_train"], seed=42, n_estimators=50, max_depth=3
    )

    preds_val = clf.predict(splits["X_val"])
    metrics = evaluate_predictions(splits["y_val"], preds_val, ["bajo", "medio", "alto"])

    # Con zero-inflation, un clasificador trivial que prediga siempre 'bajo' tendría F1 macro ~0.33
    # El clasificador ponderado debe superar 0.35 y tener balanced accuracy > 0.50
    assert metrics["macro_f1"] > 0.35
    assert metrics["balanced_accuracy"] > 0.50


def test_saved_classifier_artifacts_integrity():
    model_path = Path("artifacts/classifier_model.joblib")
    preds_path = Path("artifacts/predictions_classifier.parquet")
    shap_path = Path("docs/figures/classifier_shap_summary.png")

    assert model_path.exists(), "El archivo del modelo joblib debe existir"
    assert preds_path.exists(), "El archivo parquet de predicciones debe existir"
    assert shap_path.exists(), "El gráfico de interpretabilidad SHAP debe existir"

    # Verificar que el modelo se puede cargar e inferir
    model = joblib.load(model_path)
    assert hasattr(model, "predict_proba")

    # Verificar estructura del dataset de predicciones
    df_preds = pd.read_parquet(preds_path)
    assert len(df_preds) == 22620
    assert "prob_bajo" in df_preds.columns
    assert "prob_medio" in df_preds.columns
    assert "prob_alto" in df_preds.columns
    assert not df_preds[["prob_bajo", "prob_medio", "prob_alto"]].isnull().any().any()

"""Módulo del Modelo Básico: Clasificador Supervisado Multiclase.

Entrena y evalúa un modelo de clasificación supervisada para estimar el riesgo
sísmico municipal del mes siguiente (t+1) en tres clases: 'bajo', 'medio', 'alto'.

Reglas metodológicas:
- Sin data leakage: partición temporal estricta Train (<=2018), Val (2019-2021), Test (2022+).
- Manejo de zero-inflation: ponderación de clases balanceada ('balanced' sample weights).
- Interpretabilidad: importancia de variables globales y análisis SHAP (docs/figures/).
- Encadenamiento: genera probabilidades ('prob_bajo', 'prob_medio', 'prob_alto') para el RL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb


FEATURE_COLS = [
    # Sísmicas (15)
    "n_eventos",
    "mag_media",
    "mag_max",
    "prof_media",
    "prof_std",
    "energia_liberada",
    "n_eventos_M3",
    "n_eventos_M4",
    "n_eventos_3m",
    "n_eventos_6m",
    "energia_3m",
    "mag_max_3m",
    "ratio_actividad_reciente",
    "meses_desde_ultimo_evento",
    "meses_desde_ultimo_M3",
    # Territoriales y Demográficas DANE (7)
    "dist_nido_km",
    "poblacion_total",
    "poblacion_cabecera",
    "poblacion_resto",
    "pct_urbano",
    "densidad_hab_km2",
    "area_km2",
]


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Carga configuración general del proyecto."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_data_splits(
    df: pd.DataFrame, feature_cols: list[str] | None = None
) -> dict[str, Any]:
    """Divide los datos en train, val y test según la partición temporal sin mezclar fechas."""
    cols = feature_cols or FEATURE_COLS

    splits = {}
    for split_name in ["train", "val", "test"]:
        mask = df["split_temporal"] == split_name
        splits[f"X_{split_name}"] = df.loc[mask, cols].copy()
        splits[f"y_{split_name}"] = df.loc[mask, "riesgo_t1_num"].astype(int).to_numpy()
        splits[f"df_{split_name}"] = df.loc[mask].copy()

    return splits


def train_baseline_logistic(
    X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 42
) -> Pipeline:
    """Entrena modelo baseline de regresión logística multinomial con ponderación balanceada."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            ),
        ),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_challenger_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int = 42,
    n_estimators: int = 120,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> xgb.XGBClassifier:
    """Entrena modelo XGBoost multiclase con ponderación balanceada por muestra."""
    sample_weights = compute_sample_weight("balanced", y_train)

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=seed,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    return clf


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Calcula métricas clave para clasificación desbalanceada."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def generate_shap_plot(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame,
    feature_cols: list[str],
    output_path: str = "docs/figures/classifier_shap_summary.png",
) -> dict[str, float]:
    """Genera y guarda el gráfico SHAP summary y retorna la importancia media por variable."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values puede ser array 3D (n_samples, n_features, n_classes) o lista de arrays
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importances = {
        col: float(val) for col, val in zip(feature_cols, mean_abs_shap)
    }
    sorted_importances = dict(
        sorted(importances.items(), key=lambda item: item[1], reverse=True)
    )

    # Graficar con matplotlib
    plt.figure(figsize=(10, 6))
    top_items = list(sorted_importances.items())[:12]
    feats, vals = [k for k, _ in top_items][::-1], [v for _, v in top_items][::-1]
    plt.barh(feats, vals, color="#2b5c8f")
    plt.xlabel("Impacto medio absoluto |SHAP value|")
    plt.title("Interpretabilidad del Clasificador Sísmico: Top Variables (SHAP)")
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()

    return sorted_importances


def run_classifier_pipeline(
    config_path: str = "config.yaml",
) -> tuple[xgb.XGBClassifier, pd.DataFrame, dict[str, Any]]:
    """Ejecuta el pipeline completo del clasificador supervisado y guarda artefactos."""
    cfg = load_config(config_path)
    seed = int(cfg.get("project", {}).get("seed", 42))
    classes = cfg.get("label", {}).get("classes", ["bajo", "medio", "alto"])
    dataset_path = cfg.get("paths", {}).get("dataset", "data/processed/dataset.parquet")

    df = pd.read_parquet(dataset_path)
    splits = prepare_data_splits(df, FEATURE_COLS)

    # 1. Baseline: Logistic Regression
    baseline_model = train_baseline_logistic(splits["X_train"], splits["y_train"], seed=seed)
    y_pred_base_val = baseline_model.predict(splits["X_val"])
    y_pred_base_test = baseline_model.predict(splits["X_test"])

    baseline_metrics = {
        "val": evaluate_predictions(splits["y_val"], y_pred_base_val, classes),
        "test": evaluate_predictions(splits["y_test"], y_pred_base_test, classes),
    }

    # 2. Retador: XGBoost
    challenger_model = train_challenger_xgboost(splits["X_train"], splits["y_train"], seed=seed)
    y_pred_xgb_val = challenger_model.predict(splits["X_val"])
    y_pred_xgb_test = challenger_model.predict(splits["X_test"])

    challenger_metrics = {
        "val": evaluate_predictions(splits["y_val"], y_pred_xgb_val, classes),
        "test": evaluate_predictions(splits["y_test"], y_pred_xgb_test, classes),
    }

    # 3. Predicciones y Probabilidades para TODO el dataset (insumo encadenado)
    all_X = df[FEATURE_COLS]
    probs = challenger_model.predict_proba(all_X)
    preds = challenger_model.predict(all_X)

    predictions_df = df[[
        "cod_mpio", "mpio_nombre", "periodo_mes", "split_temporal", "riesgo_t1", "riesgo_t1_num"
    ]].copy()
    predictions_df["pred_riesgo_num"] = preds
    predictions_df["pred_riesgo"] = [classes[p] for p in preds]
    predictions_df["prob_bajo"] = probs[:, 0]
    predictions_df["prob_medio"] = probs[:, 1]
    predictions_df["prob_alto"] = probs[:, 2]

    # Guardar predicciones enriquecidas
    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    pred_output_path = artifacts_dir / "predictions_classifier.parquet"
    predictions_df.to_parquet(pred_output_path, index=False)

    # 4. Guardar modelo entrenado
    model_output_path = artifacts_dir / "classifier_model.joblib"
    joblib.dump(challenger_model, model_output_path)

    # 5. SHAP Explainability
    figures_dir = Path(cfg.get("paths", {}).get("figures", "docs/figures"))
    shap_png_path = figures_dir / "classifier_shap_summary.png"
    shap_sample = splits["X_test"].sample(min(300, len(splits["X_test"])), random_state=seed)
    shap_rankings = generate_shap_plot(challenger_model, shap_sample, FEATURE_COLS, str(shap_png_path))

    # 6. Guardar métricas completas en JSON
    metrics_summary = {
        "baseline_logistic": baseline_metrics,
        "challenger_xgboost": challenger_metrics,
        "shap_importance_top": list(shap_rankings.items())[:10],
        "features_used": FEATURE_COLS,
    }
    metrics_path = artifacts_dir / "classifier_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    return challenger_model, predictions_df, metrics_summary


if __name__ == "__main__":
    model, preds, metrics = run_classifier_pipeline()
    print("=== MODELO BÁSICO CLASIFICADOR ENTRENADO Y EVALUADO ===")
    print(f"XGBoost Test Macro F1: {metrics['challenger_xgboost']['test']['macro_f1']:.4f}")
    print(f"XGBoost Test Balanced Acc: {metrics['challenger_xgboost']['test']['balanced_accuracy']:.4f}")
    print(f"Predicciones guardadas en artifacts/predictions_classifier.parquet")
    print(f"Modelo guardado en artifacts/classifier_model.joblib")
    print(f"Gráfico SHAP guardado en docs/figures/classifier_shap_summary.png")

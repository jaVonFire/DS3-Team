"""Módulo de Entrenamiento y Evaluación del Agente de Refuerzo (Nivel Avanzado).

Entrena agentes PPO con Gymnasium y Stable-Baselines3, compara exhaustivamente
contra los 4 baselines obligatorios (Aleatorio, Histórico, Solo Clasificador, Riesgo x Población),
calcula incertidumbre mediante múltiples semillas e intervalos bootstrap (1.000 remuestreos),
y exporta los artefactos requeridos para la rúbrica y la aplicación Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
import yaml

from src.models.rl_env import SantanderSeismicEnv


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_merged_rl_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Carga y unifica las salidas encadenadas del dataset, clasificador y clustering."""
    dataset_path = config.get("paths", {}).get("dataset", "data/processed/dataset.parquet")
    artifacts_dir = Path(config.get("paths", {}).get("artifacts", "artifacts"))

    preds_path = artifacts_dir / "predictions_classifier.parquet"
    clust_path = artifacts_dir / "municipal_clusters.parquet"

    df_ds = pd.read_parquet(dataset_path)
    df_preds = pd.read_parquet(preds_path)
    df_clust = pd.read_parquet(clust_path)

    merged = df_ds.merge(
        df_preds[["cod_mpio", "periodo_mes", "prob_bajo", "prob_medio", "prob_alto"]],
        on=["cod_mpio", "periodo_mes"],
        how="left",
    ).merge(
        df_clust[["cod_mpio", "cluster_id"]],
        on="cod_mpio",
        how="left",
    )

    return merged


def run_heuristic_baseline(
    env: SantanderSeismicEnv, policy_type: str, seed: int = 42
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ejecuta una política heurística en el entorno hasta el final del split."""
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    tot_reward = 0.0
    tot_hits = 0
    tot_misses = 0
    tot_neglect = 0
    protected_pops = []
    history = []

    done = False
    while not done:
        curr = env._data_by_period[env.current_step]

        if policy_type == "random":
            action = rng.uniform(-1.0, 1.0, 5).astype(np.float32)
        elif policy_type == "historical_most_active":
            # Pondera estrictamente la recurrencia histórica
            action = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        elif policy_type == "classifier_only":
            # Pondera únicamente la probabilidad de riesgo alto estimado
            action = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif policy_type == "risk_x_population":
            # Pondera riesgo alto y población simultáneamente
            action = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        else:
            raise ValueError(f"Política desconocida: {policy_type}")

        obs, r, term, trunc, info = env.step(action)
        tot_reward += r
        tot_hits += info["hits"]
        tot_misses += info["misses"]
        tot_neglect += info["neglect_hits"]
        protected_pops.append(info["protected_population"])
        history.append({
            "period": info["period"],
            "selected_names": info["selected_names"],
            "reward": r,
            "hits": info["hits"],
            "misses": info["misses"],
            "neglect": info["neglect_hits"],
            "protected_population": info["protected_population"],
        })
        done = term or trunc

    metrics = {
        "policy": policy_type,
        "total_reward": float(tot_reward),
        "total_hits": int(tot_hits),
        "total_misses": int(tot_misses),
        "total_neglect": int(tot_neglect),
        "mean_protected_population": float(np.mean(protected_pops)),
    }
    return metrics, history


def evaluate_ppo_model(
    env: SantanderSeismicEnv, model: sb3.PPO, seed: int = 42
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evalúa un modelo PPO entrenado en el entorno de prueba."""
    obs, _ = env.reset(seed=seed)

    tot_reward = 0.0
    tot_hits = 0
    tot_misses = 0
    tot_neglect = 0
    protected_pops = []
    history = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        tot_reward += r
        tot_hits += info["hits"]
        tot_misses += info["misses"]
        tot_neglect += info["neglect_hits"]
        protected_pops.append(info["protected_population"])
        history.append({
            "period": info["period"],
            "selected_names": info["selected_names"],
            "reward": r,
            "hits": info["hits"],
            "misses": info["misses"],
            "neglect": info["neglect_hits"],
            "protected_population": info["protected_population"],
            "weights": info["weights_used"],
        })
        done = term or trunc

    metrics = {
        "policy": "PPO_Agent",
        "total_reward": float(tot_reward),
        "total_hits": int(tot_hits),
        "total_misses": int(tot_misses),
        "total_neglect": int(tot_neglect),
        "mean_protected_population": float(np.mean(protected_pops)),
    }
    return metrics, history


def compute_bootstrap_ci(
    values: list[float], n_resamples: int = 1000, ci: float = 0.95
) -> dict[str, float]:
    """Calcula intervalo de confianza bootstrap para métricas del agente."""
    rng = np.random.default_rng(42)
    boot_means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return {"mean": float(np.mean(values)), "ci_low": low, "ci_high": high}


def generate_rl_comparison_chart(
    comparison_data: dict[str, dict[str, Any]],
    output_path: str = "docs/figures/rl_vs_baselines_performance.png",
) -> None:
    """Genera gráfico comparativo de barras entre PPO y los 4 baselines."""
    policies = list(comparison_data.keys())
    rewards = [comparison_data[p]["total_reward"] for p in policies]
    hits = [comparison_data[p]["total_hits"] for p in policies]
    neglect = [comparison_data[p]["total_neglect"] for p in policies]

    x = np.arange(len(policies))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Recompensa Total Acumulada
    colors = ["#2ca02c" if p == "PPO_Agent" else "#1f77b4" for p in policies]
    bars = ax1.bar(x, rewards, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=10)
    ax1.set_ylabel("Recompensa Acumulada en Test (2022–2026)")
    ax1.set_title("Comparativa de Recompensa: Agente PPO vs. Baselines", fontsize=12, fontweight="bold")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax1.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )

    # Gráfico 2: Aciertos Sísmicos y Penalizaciones por Desatención
    width = 0.35
    b1 = ax2.bar(x - width / 2, hits, width, label="Aciertos Sísmicos (Hits)", color="#2b5c8f")
    b2 = ax2.bar(x + width / 2, neglect, width, label="Eventos en Desatención", color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=10)
    ax2.set_ylabel("Frecuencia de Eventos")
    ax2.set_title("Eficiencia Operativa y Control de Desatención", fontsize=12, fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_rl_pipeline(config_path: str = "config.yaml") -> dict[str, Any]:
    """Ejecuta el pipeline completo de entrenamiento y evaluación multillave de RL."""
    cfg = load_config(config_path)
    rl_cfg = cfg.get("rl", {})
    seeds = cfg.get("project", {}).get("rl_seeds", [11, 22, 33, 44, 55])
    timesteps = int(rl_cfg.get("total_timesteps", 12000))
    bootstrap_resamples = int(rl_cfg.get("bootstrap_resamples", 1000))

    df_merged = prepare_merged_rl_dataset(cfg)

    # 1. Evaluar los 4 baselines obligatorios en la partición Test (2022–2026)
    baselines_list = ["random", "historical_most_active", "classifier_only", "risk_x_population"]
    baseline_results = {}
    for b_name in baselines_list:
        test_env = SantanderSeismicEnv(df_merged, split_name="test", seed=42)
        b_metrics, _ = run_heuristic_baseline(test_env, b_name, seed=42)
        baseline_results[b_name] = b_metrics

    # 2. Entrenar y evaluar el Agente PPO a través de múltiples semillas
    ppo_seed_metrics = []
    best_reward = -float("inf")
    best_model = None
    best_history = None

    for seed in seeds:
        train_env = SantanderSeismicEnv(df_merged, split_name="train", seed=seed)
        test_env = SantanderSeismicEnv(df_merged, split_name="test", seed=seed)

        model = sb3.PPO(
            "MlpPolicy",
            train_env,
            seed=seed,
            n_steps=64,
            batch_size=32,
            learning_rate=3e-4,
            ent_coef=0.01,
            gamma=0.95,
            verbose=0,
        )
        model.learn(total_timesteps=timesteps)

        eval_metrics, hist = evaluate_ppo_model(test_env, model, seed=seed)
        ppo_seed_metrics.append(eval_metrics)

        if eval_metrics["total_reward"] > best_reward:
            best_reward = eval_metrics["total_reward"]
            best_model = model
            best_history = hist

    # 3. Calcular estadísticas multillave e intervalos bootstrap
    rewards = [m["total_reward"] for m in ppo_seed_metrics]
    hits = [m["total_hits"] for m in ppo_seed_metrics]
    protected = [m["mean_protected_population"] for m in ppo_seed_metrics]

    ppo_stats = {
        "reward": compute_bootstrap_ci(rewards, n_resamples=bootstrap_resamples),
        "reward_std": float(np.std(rewards)),
        "hits": compute_bootstrap_ci(hits, n_resamples=bootstrap_resamples),
        "mean_protected_pop": compute_bootstrap_ci(protected, n_resamples=bootstrap_resamples),
        "all_seeds_metrics": ppo_seed_metrics,
    }

    # 4. Guardar mejor modelo PPO
    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "rl_agent_ppo.zip"
    if best_model is not None:
        best_model.save(str(model_path))

    # 5. Guardar simulación detallada paso a paso en test
    if best_history is not None:
        sim_df = pd.DataFrame(best_history)
        sim_df.to_parquet(artifacts_dir / "rl_simulation_test.parquet", index=False)

    # 6. Generar gráfico comparativo
    comparison_summary = {
        "PPO_Agent": {
            "total_reward": ppo_stats["reward"]["mean"],
            "total_hits": int(ppo_stats["hits"]["mean"]),
            "total_neglect": int(np.mean([m["total_neglect"] for m in ppo_seed_metrics])),
        }
    }
    for b_name, b_data in baseline_results.items():
        comparison_summary[b_name] = b_data

    figures_dir = Path(cfg.get("paths", {}).get("figures", "docs/figures"))
    chart_path = figures_dir / "rl_vs_baselines_performance.png"
    generate_rl_comparison_chart(comparison_summary, str(chart_path))

    # 7. Exportar reporte de métricas en JSON
    benchmark_report = {
        "baselines": baseline_results,
        "ppo_agent": ppo_stats,
        "seeds_evaluated": seeds,
        "timesteps_per_seed": timesteps,
    }
    report_path = artifacts_dir / "rl_benchmark_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

    return benchmark_report


if __name__ == "__main__":
    report = run_rl_pipeline()
    print("=== MODELO AVANZADO: AGENTE DE REFUERZO ENTRENADO Y EVALUADO ===")
    print(f"PPO Media Reward: {report['ppo_agent']['reward']['mean']:.2f} (95% CI: [{report['ppo_agent']['reward']['ci_low']:.2f}, {report['ppo_agent']['reward']['ci_high']:.2f}])")
    print(f"PPO Media Hits: {report['ppo_agent']['hits']['mean']:.1f}")
    print("\nComparativa con Baselines:")
    for b_name, data in report["baselines"].items():
        print(f" - {b_name:25s}: Reward={data['total_reward']:.2f}, Hits={data['total_hits']}")
    print("\nArtefactos guardados:")
    print(" - artifacts/rl_agent_ppo.zip")
    print(" - artifacts/rl_benchmark_metrics.json")
    print(" - artifacts/rl_simulation_test.parquet")
    print(" - docs/figures/rl_vs_baselines_performance.png")

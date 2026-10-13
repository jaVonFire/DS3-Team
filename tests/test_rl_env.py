"""Pruebas unitarias para el Entorno y Agente de Refuerzo (Nivel Avanzado).

Verifica la conformidad del entorno Gymnasium, el cumplimiento de restricciones
dinámicas (cooldown mensual, presupuesto top-3), la evaluación de baselines
y la integridad de los artefactos exportados.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import stable_baselines3 as sb3

from src.models.rl_env import SantanderSeismicEnv
from src.models.rl_train import load_config, prepare_merged_rl_dataset


@pytest.fixture(scope="module")
def merged_df():
    cfg = load_config("config.yaml")
    return prepare_merged_rl_dataset(cfg)


def test_env_observation_and_action_spaces(merged_df):
    env = SantanderSeismicEnv(merged_df, split_name="val", seed=42)
    obs, info = env.reset(seed=42)

    assert obs.shape == (10,), "La observación debe tener dimensión 10"
    assert env.action_space.shape == (5,), "La acción debe tener dimensión 5"
    assert not np.isnan(obs).any(), "No debe haber valores nulos en la observación"


def test_env_enforces_cooldown_constraint(merged_df):
    env = SantanderSeismicEnv(merged_df, split_name="val", top_k=3, seed=42)
    env.reset(seed=42)

    # Paso 1: Acción que prioriza riesgo
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info1 = env.step(action)
    selected_step1 = set(info1["selected_codes"])
    assert len(selected_step1) == 3

    # Paso 2: Acción idéntica en el siguiente mes
    _, _, _, _, info2 = env.step(action)
    selected_step2 = set(info2["selected_codes"])
    assert len(selected_step2) == 3

    # NINGÚN municipio atendido en step 1 puede repetir en step 2 por la regla de cooldown
    assert selected_step1.isdisjoint(selected_step2), (
        f"Violación de cooldown: municipios repetidos consecutivos {selected_step1 & selected_step2}"
    )


def test_env_runs_to_termination(merged_df):
    env = SantanderSeismicEnv(merged_df, split_name="val", seed=42)
    obs, _ = env.reset(seed=42)

    steps = 0
    done = False
    while not done:
        action = np.zeros(5, dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        steps += 1
        done = term or trunc

    assert steps == env.n_steps_max
    assert "hits" in info
    assert "misses" in info
    assert "protected_population" in info


def test_rl_artifacts_integrity():
    model_path = Path("artifacts/rl_agent_ppo.zip")
    metrics_path = Path("artifacts/rl_benchmark_metrics.json")
    sim_path = Path("artifacts/rl_simulation_test.parquet")
    chart_path = Path("docs/figures/rl_vs_baselines_performance.png")

    assert model_path.exists(), "rl_agent_ppo.zip debe existir"
    assert metrics_path.exists(), "rl_benchmark_metrics.json debe existir"
    assert sim_path.exists(), "rl_simulation_test.parquet debe existir"
    assert chart_path.exists(), "rl_vs_baselines_performance.png debe existir"

    # Verificar que el modelo se carga y puede predecir
    model = sb3.PPO.load(str(model_path))
    dummy_obs = np.zeros(10, dtype=np.float32)
    action, _ = model.predict(dummy_obs, deterministic=True)
    assert action.shape == (5,)

    # Verificar simulación
    df_sim = pd.read_parquet(sim_path)
    assert len(df_sim) > 0
    assert "period" in df_sim.columns
    assert "selected_names" in df_sim.columns

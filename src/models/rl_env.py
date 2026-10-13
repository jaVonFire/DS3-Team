"""Módulo del Entorno Gymnasium para Priorización Sísmica Municipal (Nivel Avanzado).

Define el entorno 'SantanderSeismicEnv' que simula la asignación mensual de recursos
de preparación ante riesgo sísmico en los 87 municipios de Santander.

Justificación metodológica del Agente de Refuerzo:
- Frente a ordenamientos deterministas estáticos (como 'riesgo × población' o 'históricamente activos'),
  las decisiones reales de gestión del riesgo están sujetas a restricciones dinámicas:
  1. Presupuesto top-k (k=3 municipios atendidos por mes).
  2. Cooldown operativo (un municipio atendido no puede repetir al mes siguiente).
  3. Penalización por desatención acumulada (> 6 meses de abandono territorial).
  4. Ponderación por vidas humanas expuestas (DANE) y magnitud física de sacudidas en t+1.
- El agente aprende una política óptima de ponderación estratégica [w_alto, w_medio, w_pob, w_neglect, w_cluster]
  que balancea la anticipación de sismos con la equidad territorial y la prevención del colapso de política.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


class SantanderSeismicEnv(gym.Env):
    """Entorno Gymnasium de asignación mensual de recursos de preparación ante riesgo sísmico."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_merged: pd.DataFrame,
        split_name: str = "train",
        top_k: int = 3,
        cooldown_months: int = 1,
        neglect_penalty_months: int = 6,
        miss_penalty: float = 1.0,
        strong_event_threshold: float = 3.0,
        seed: int = 42,
    ):
        super().__init__()

        self.top_k = top_k
        self.cooldown_months = cooldown_months
        self.neglect_penalty_months = neglect_penalty_months
        self.miss_penalty = miss_penalty
        self.strong_event_threshold = strong_event_threshold
        self.split_name = split_name

        # Filtrar datos de la partición temporal sin fuga de información
        df_split = df_merged[df_merged["split_temporal"] == split_name].copy()
        self.periods = sorted(df_split["periodo_mes"].unique())
        self.n_steps_max = len(self.periods) - 1

        # Catálogo único ordenado de los 87 municipios
        self.mun_list = sorted(df_split["cod_mpio"].unique())
        self.n_mun = len(self.mun_list)  # 87
        self.mun_names = df_split.groupby("cod_mpio")["mpio_nombre"].first().to_dict()

        # Cache de datos preindexados por período para velocidad óptima
        self._data_by_period: list[dict[str, np.ndarray]] = []
        for p in self.periods:
            sub = df_split[df_split["periodo_mes"] == p].set_index("cod_mpio").reindex(self.mun_list)

            p_alto = sub["prob_alto"].fillna(0.0).to_numpy(dtype=np.float32)
            p_medio = sub["prob_medio"].fillna(0.0).to_numpy(dtype=np.float32)
            c_id = sub["cluster_id"].fillna(0).to_numpy(dtype=np.float32) / 3.0
            pob = sub["poblacion_total"].fillna(1000.0).to_numpy(dtype=np.float32)
            pob_norm = np.clip((np.log10(np.maximum(pob, 100.0)) - 3.0) / 3.0, 0.0, 1.0)
            act_t1 = sub["target_actividad_t1"].fillna(0.0).to_numpy(dtype=np.float32)
            mag_t1 = sub["target_mag_max_t1"].fillna(0.0).to_numpy(dtype=np.float32)

            self._data_by_period.append({
                "prob_alto": p_alto,
                "prob_medio": p_medio,
                "cluster_norm": c_id,
                "pob_norm": pob_norm,
                "act_t1": act_t1,
                "mag_t1": mag_t1,
                "poblacion_real": pob,
            })

        # Precalcular conteos históricos de entrenamiento para el baseline de más activos
        df_train = df_merged[df_merged["split_temporal"] == "train"]
        train_counts = df_train.groupby("cod_mpio")["n_eventos"].sum()
        self.hist_activity_vector = np.array(
            [float(train_counts.get(c, 0)) for c in self.mun_list], dtype=np.float32
        )

        # Espacio de Observación (10 métricas del estado regional):
        # [max_prob_alto, mean_prob_alto, p90_prob_alto, max_desatencion, mean_desatencion,
        #  pct_desatendidos_criticos, pct_en_cooldown, progreso_temporal, pct_medio, std_prob_alto]
        self.observation_space = gym.spaces.Box(0.0, 5.0, shape=(10,), dtype=np.float32)

        # Espacio de Acción: 5 ponderaciones estratégicas continuas [-1, 1]
        # [w_prob_alto, w_prob_medio, w_pob, w_neglect, w_cluster]
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def _get_obs(self) -> np.ndarray:
        curr = self._data_by_period[self.current_step]
        pa = curr["prob_alto"]
        sa = self.meses_sin_atencion
        return np.array([
            float(pa.max()),
            float(pa.mean()),
            float(np.percentile(pa, 90)),
            float(sa.max() / 12.0),
            float(sa.mean() / 12.0),
            float((sa >= self.neglect_penalty_months).mean()),
            float(self.en_cooldown.mean()),
            float(self.current_step / max(1, self.n_steps_max)),
            float((curr["prob_medio"] > 0.1).mean()),
            float(np.std(pa)),
        ], dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        self.meses_sin_atencion = np.zeros(self.n_mun, dtype=np.float32)
        self.en_cooldown = np.zeros(self.n_mun, dtype=bool)

        return self._get_obs(), {}

    def compute_scores_and_selection(
        self, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcula el vector de prioridad por municipio y selecciona los top-k elegibles."""
        w = np.asarray(weights, dtype=np.float32)
        curr = self._data_by_period[self.current_step]

        # Puntuación paramétrica por municipio
        scores = (
            w[0] * curr["prob_alto"]
            + w[1] * curr["prob_medio"]
            + w[2] * curr["pob_norm"]
            + w[3] * (self.meses_sin_atencion / 12.0)
            + w[4] * curr["cluster_norm"]
        )

        # Enmascarar municipios bajo cooldown estricto (no pueden ser seleccionados)
        scores[self.en_cooldown] = -999.0

        # Seleccionar top-k
        selected_indices = np.argsort(scores)[::-1][: self.top_k]
        return scores, selected_indices

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, selected_indices = self.compute_scores_and_selection(action)
        selected_mask = np.zeros(self.n_mun, dtype=bool)
        selected_mask[selected_indices] = True

        curr = self._data_by_period[self.current_step]
        events_t1 = curr["act_t1"] > 0
        mag_t1 = curr["mag_t1"]
        pob_w = curr["pob_norm"]

        reward = 0.0
        hits = 0
        misses = 0
        neglect_hits = 0
        protected_population = float(curr["poblacion_real"][selected_indices].sum())

        for i in range(self.n_mun):
            if selected_mask[i]:
                # Municipio atendido por el agente
                if events_t1[i]:
                    hits += 1
                    # Acierto ponderado por magnitud de la sacudida y volumen de vidas protegidas
                    reward += (1.0 + mag_t1[i] / 5.0) * (1.0 + 2.0 * pob_w[i])
                else:
                    # Costo de oportunidad menor
                    reward -= 0.05
            else:
                # Municipio no atendido
                if events_t1[i] and mag_t1[i] >= self.strong_event_threshold:
                    misses += 1
                    reward -= self.miss_penalty * (1.0 + pob_w[i])
                if events_t1[i] and self.meses_sin_atencion[i] >= self.neglect_penalty_months:
                    neglect_hits += 1
                    reward -= 1.5 * (1.0 + pob_w[i])

        # Actualizar dinámicas temporales
        self.meses_sin_atencion[selected_mask] = 0.0
        self.en_cooldown[:] = False
        self.en_cooldown[selected_mask] = True
        self.meses_sin_atencion[~selected_mask] += 1.0

        self.current_step += 1
        terminated = self.current_step >= self.n_steps_max
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(10, dtype=np.float32)

        info = {
            "step": self.current_step,
            "period": self.periods[self.current_step - 1],
            "hits": hits,
            "misses": misses,
            "neglect_hits": neglect_hits,
            "protected_population": protected_population,
            "selected_codes": [self.mun_list[idx] for idx in selected_indices],
            "selected_names": [self.mun_names[self.mun_list[idx]] for idx in selected_indices],
            "weights_used": [float(x) for x in action],
        }

        return obs, float(reward), terminated, truncated, info

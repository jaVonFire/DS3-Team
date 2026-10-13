"""Prueba de humo del stack RL: Gymnasium + Stable-Baselines3 (PPO) entrenan unos pasos.

No usa datos reales: solo verifica que el entorno de Windows puede entrenar un agente PPO con
semilla fija. Se omite automáticamente si el stack RL no está instalado.
"""
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")

from src.config import load_config, set_global_seed  # noqa: E402


class _ToyPriorityEnv(gym.Env):
    """Mini-entorno: elegir 1 de N municipios; recompensa si coincide con el de mayor 'riesgo'."""

    def __init__(self, n: int = 5, horizon: int = 12, seed: int = 0):
        super().__init__()
        self.n, self.horizon = n, horizon
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.obs = self._rng.random(self.n).astype(np.float32)
        return self.obs, {}

    def step(self, action):
        reward = float(action == int(np.argmax(self.obs)))
        self.t += 1
        self.obs = self._rng.random(self.n).astype(np.float32)
        return self.obs, reward, False, self.t >= self.horizon, {}


def test_ppo_trains_with_fixed_seed():
    seed = set_global_seed(load_config()["project"]["seed"])
    env = _ToyPriorityEnv(seed=seed)
    model = sb3.PPO("MlpPolicy", env, seed=seed, n_steps=64, batch_size=32, device="cpu", verbose=0)
    model.learn(total_timesteps=256)
    action, _ = model.predict(env.reset(seed=seed)[0], deterministic=True)
    assert 0 <= int(action) < env.n

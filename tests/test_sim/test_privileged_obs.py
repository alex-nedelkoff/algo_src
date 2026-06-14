"""Privileged Dict obs channel + per-env DR hook in GateRaceEnv (COR-127)."""
import numpy as np
from gymnasium import spaces

from sim.envs.gate_race_env import GateRaceEnv
from sim.tracks import Track, GateState


def _track():
    gs = [GateState(position=np.array([8.0 + 8 * i, 0.0, 1.0]),
                    orientation=np.array([1.0, 0, 0, 0])) for i in range(6)]
    return Track(gates=gs, name="c", start_position=np.array([0, 0, 1.0]))


def _env(**kw):
    return GateRaceEnv(n_envs=4, dt=1 / 72.0, max_steps=200, action_mode="vq_rate",
                       max_body_rate=6.0, vq_max_thrust=0.6,
                       tracks=[_track() for _ in range(4)], **kw)


def test_flat_obs_unchanged_when_off():
    env = _env()
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, np.ndarray) and obs.shape[0] == 4


def test_privileged_dict_obs_shape():
    env = _env(privileged_obs=True, dr_width=0.4)
    assert isinstance(env.observation_space, spaces.Dict)
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"policy", "privileged"}
    assert obs["policy"].shape[0] == 4
    assert obs["privileged"].shape == (4, 11)
    # privileged is per-env constant across a step (fixed-at-construction DR)
    a = np.zeros((4, 4))
    obs2, *_ = env.step(a)
    np.testing.assert_array_equal(obs["privileged"], obs2["privileged"])


def test_terminal_obs_dict_when_privileged():
    env = _env(privileged_obs=True, dr_width=0.4)
    env.reset(seed=0)
    a = np.zeros((4, 4))
    _, _, _, _, info = env.step(a)
    t = info["terminal_obs"]
    assert isinstance(t, dict) and set(t.keys()) == {"policy", "privileged"}
    assert t["privileged"].shape == (4, 11)


def test_vec_adapter_dict_passthrough():
    import pytest
    pytest.importorskip("stable_baselines3")   # VecEnvAdapter base is sb3 (pod-only)
    from sim.envs.vec_env_adapter import VecEnvAdapter
    env = _env(privileged_obs=True, dr_width=0.4)
    venv = VecEnvAdapter(env)
    obs = venv.reset()
    assert set(obs.keys()) == {"policy", "privileged"}
    assert obs["policy"].shape == (4, env._obs_dim)

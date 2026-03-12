"""VecEnv wrapper that runs EKF internally and exposes filtered observations.

Wraps any SB3 VecEnv backed by a RateCtrlEnv (or similar), runs the 16-state
EKF at each step, and returns gate-relative 24D observations computed from
the EKF state estimate rather than raw simulator state.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Type, Union

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


class EKFVecEnvWrapper(VecEnv):
    """SB3 VecEnv that provides EKF-filtered observations.

    Wraps an existing VecEnv (backed by RateCtrlEnv via VecEnvAdapter),
    runs the 16-state EKF at each step, and returns gate-relative 24D
    observations computed from the EKF state estimate.

    Args:
        vec_env: The underlying SB3 VecEnv to wrap.
        camera_kwargs: Kwargs for PinholeCamera construction.
        corner_noise_k: Noise scale for corner detector.
        corner_dropout_onset: HMM dropout onset probability.
        provide_teacher_obs: If True, store raw obs as ``last_teacher_obs``.
    """

    def __init__(
        self,
        vec_env: VecEnv,
        camera_kwargs: Optional[dict] = None,
        corner_noise_k: float = 2.0,
        corner_dropout_onset: Optional[float] = None,
        provide_teacher_obs: bool = False,
    ) -> None:
        from perception.camera import PinholeCamera, CornerDetector
        from state_estimation.ekf16 import EKF16

        self._inner = vec_env
        self._raw_env = vec_env.env  # type: ignore[attr-defined]
        n_envs = vec_env.num_envs
        self.provide_teacher_obs = provide_teacher_obs

        cam_kw = camera_kwargs or {}
        self.camera = PinholeCamera(**cam_kw)
        self.corner_detector = CornerDetector(
            self.camera,
            n_envs=n_envs,
            seed=42,
            noise_k=corner_noise_k,
            dropout_onset=corner_dropout_onset,
        )

        dt = getattr(self._raw_env, "dt", 0.002)
        action_repeat = getattr(self._raw_env, "action_repeat", 5)
        self.ekf = EKF16(n_envs=n_envs, dt=dt * action_repeat)

        # Observation space: 24D gate-relative obs from EKF
        observation_space = gymnasium.spaces.Box(
            low=-np.inf * np.ones(24, dtype=np.float32),
            high=np.inf * np.ones(24, dtype=np.float32),
            shape=(24,),
            dtype=np.float32,
        )

        super().__init__(n_envs, observation_space, vec_env.action_space)
        self._actions: Optional[np.ndarray] = None
        self.last_teacher_obs: Optional[np.ndarray] = None

    def reset(self) -> VecEnvObs:  # type: ignore[override]
        obs = self._inner.reset()
        # Initialize EKF with true state
        if hasattr(self._raw_env, "_state"):
            mask = np.ones(self.num_envs, dtype=bool)
            self.ekf.set_state(self._raw_env._state, mask)
            self.corner_detector.reset(mask)
        if self.provide_teacher_obs:
            self.last_teacher_obs = obs.copy()
        return self._ekf_obs().astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        assert self._actions is not None
        self._inner.step_async(self._actions)
        obs, rewards, dones, infos = self._inner.step_wait()
        self._actions = None

        if self.provide_teacher_obs:
            self.last_teacher_obs = obs.copy()

        # Run EKF
        self._run_ekf_update()

        # Auto-reset EKF for done envs
        if np.any(dones) and hasattr(self._raw_env, "_state"):
            self.ekf.set_state(self._raw_env._state[dones], dones)
            self.corner_detector.reset(dones)

        ekf_obs = self._ekf_obs().astype(np.float32)
        return ekf_obs, rewards, dones, infos

    def close(self) -> None:
        self._inner.close()

    def env_method(
        self, method_name: str, *method_args: Any,
        indices: VecEnvIndices = None, **method_kwargs: Any,
    ) -> List[Any]:
        return self._inner.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type[gymnasium.Wrapper],
        indices: VecEnvIndices = None,
    ) -> List[bool]:
        return [False] * self.num_envs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self._inner.get_attr(attr_name, indices)

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None,
    ) -> None:
        self._inner.set_attr(attr_name, value, indices)

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[int, None]]:
        return self._inner.seed(seed)

    # ------------------------------------------------------------------
    # EKF internals
    # ------------------------------------------------------------------

    def _run_ekf_update(self) -> None:
        """Run EKF predict + update using current raw env state."""
        from perception.camera import compute_gate_corners, world_to_camera

        raw = self._raw_env
        if not hasattr(raw, "_state"):
            return

        true_state = raw._state

        # IMU update
        imu_vel = true_state[:, 3:6].copy()
        imu_euler = true_state[:, 6:9].copy()
        imu_rates = true_state[:, 9:12].copy()
        imu_motors = true_state[:, 12:16].copy()
        # Sensor noise
        rng = np.random.default_rng()
        imu_vel += rng.normal(scale=0.05, size=imu_vel.shape)
        imu_euler += rng.normal(scale=0.03, size=imu_euler.shape)
        imu_rates += rng.normal(scale=0.02, size=imu_rates.shape)
        imu_motors += rng.normal(scale=10.0, size=imu_motors.shape)
        self.ekf.update_imu(imu_vel, imu_euler, imu_rates, imu_motors)

        # Predict
        self.ekf.predict()

        # Corner detection + pixel update
        if hasattr(raw, "_gate_positions") and hasattr(raw, "_gate_idx"):
            gate_pos = raw._gate_positions[raw._gate_idx]
            gate_yaw = raw._gate_yaws[raw._gate_idx]

            corners_w = compute_gate_corners(gate_pos, gate_yaw, 0.55, 0.55)
            cam_pts = world_to_camera(
                corners_w, true_state[:, :3], true_state[:, 6:9]
            )
            d2g = np.linalg.norm(true_state[:, :3] - gate_pos, axis=1)
            pixels, visible = self.corner_detector.detect(cam_pts, d2g)
            self.ekf.update(pixels, visible, gate_pos, gate_yaw, self.camera)

    def _ekf_obs(self) -> np.ndarray:
        """Compute 24D gate-relative obs from EKF state estimate."""
        from sim.envs.playground_obs import gate_relative_obs

        raw = self._raw_env
        if not hasattr(raw, "_gate_positions"):
            return np.zeros((self.num_envs, 24))

        gate_pos = raw._gate_positions[raw._gate_idx]
        gate_yaw = raw._gate_yaws[raw._gate_idx]
        next_idx = (raw._gate_idx + 1) % raw._n_gates
        next_gate_pos = raw._gate_positions[next_idx]
        next_gate_yaw = raw._gate_yaws[next_idx]

        return gate_relative_obs(
            self.ekf.x, gate_pos, gate_yaw, next_gate_pos, next_gate_yaw
        )

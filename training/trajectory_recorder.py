"""SB3 callback for recording deterministic policy rollout trajectories.

Periodically creates a temporary single-env evaluation environment (no domain
randomisation), runs the current policy deterministically for a few episodes,
and saves per-timestep state/action/reward data as .npz files for downstream
visualisation.

Environment construction is delegated to an ``EnvFactory`` so the recorder
works with any sim backend (GateRaceEnv, RateCtrlEnv, etc.).

Uses the TrajectoryProvider protocol from metrics.contract for env state
extraction — no env-specific dual-path logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from metrics.contract import (
    ContractViolation,
    TrajectoryProvider,
    validate_trajectory_state,
)

if TYPE_CHECKING:
    from artifacts.uploader import ArtifactUploader
    from sim.envs.base import EnvFactory

try:
    from stable_baselines3.common.callbacks import BaseCallback

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

log = logging.getLogger(__name__)

# Current schema version for the .npz trajectory format.
SCHEMA_VERSION = 2


class TrajectoryRecorderCallback(BaseCallback):
    """Record deterministic rollout trajectories at regular intervals.

    Every ``viz_freq`` total timesteps, spins up a lightweight single-env
    eval environment via the provided ``env_factory`` (no domain randomisation),
    rolls out ``n_viz_episodes`` episodes using the current policy in
    deterministic mode, and writes the per-step data to ``.npz`` files.

    Args:
        viz_freq: Recording frequency in **total timesteps** (not env steps).
        n_envs: Number of parallel training envs (for frequency conversion).
        save_path: Root directory for trajectory output.
        env_factory: Factory for building eval environments.
        eval_reward_cfg: Reward config for eval env.
        n_viz_episodes: Number of episodes to record each time.
        verbose: Verbosity level (0 = silent, 1 = summary line per trigger).
        uploader: Optional artifact uploader.
    """

    def __init__(
        self,
        viz_freq: int,
        n_envs: int,
        save_path: str,
        env_factory: EnvFactory | None = None,
        eval_reward_cfg: Any = None,
        n_viz_episodes: int = 5,
        verbose: int = 1,
        uploader: ArtifactUploader | None = None,
    ) -> None:
        super().__init__(verbose)
        self.viz_freq = viz_freq
        self.check_freq = max(1, viz_freq // n_envs)
        self.save_path = Path(save_path)
        self.n_viz_episodes = n_viz_episodes
        self._uploader = uploader
        self._env_factory = env_factory
        self._eval_reward_cfg = eval_reward_cfg
        self._state_validated = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_eval_env(self) -> Any:
        """Create a single-env evaluation environment.

        Uses the EnvFactory if available, otherwise falls back to cloning
        from the training env (GateRaceEnv legacy path).

        Validates the env implements TrajectoryProvider protocol.
        """
        if self._env_factory is not None:
            from omegaconf import OmegaConf

            # Build a no-DR config
            no_dr_cfg = OmegaConf.create({"enabled": False})
            reward_cfg = self._eval_reward_cfg
            if reward_cfg is None:
                reward_cfg = OmegaConf.create({})
            env = self._env_factory.make_eval_env(no_dr_cfg, reward_cfg, n_envs=1)
        else:
            # Legacy fallback: construct GateRaceEnv directly from training env
            from sim.envs.gate_race_env import GateRaceEnv

            train_env: GateRaceEnv = self.training_env.env  # type: ignore[attr-defined]
            env = GateRaceEnv(
                track=train_env.track,
                params=train_env.params,
                n_envs=1,
                dt=train_env.dt,
                max_steps=train_env.max_steps,
                ceiling=train_env.ceiling,
                reward_weights=train_env.reward_weights,
                gate_passage_radius=train_env.gate_passage_radius,
                v_max=train_env.v_max,
                action_smoothness_threshold=train_env.action_smoothness_threshold,
                esc_nonlinearity=train_env.esc_nonlinearity,
                omega_min=train_env.omega_min,
                max_body_rate=train_env.max_body_rate,
                max_velocity=train_env.max_velocity,
                arena_bounds=train_env.arena_bounds,
                domain_randomizer=None,
            )

        # Validate TrajectoryProvider protocol
        raw_env = getattr(env, "env", env) if hasattr(env, "env") else env
        if not isinstance(raw_env, TrajectoryProvider):
            raise ContractViolation(
                f"{type(raw_env).__name__} does not implement TrajectoryProvider. "
                f"See metrics/contract.py for required methods."
            )
        return env

    def _extract_gate_geometry(self, env: Any, env_idx: int = 0) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Return gate geometry via TrajectoryProvider protocol."""
        raw_env = getattr(env, "env", env)
        geom = raw_env.get_gate_geometry(env_idx=env_idx)
        return geom["positions"], geom["orientations"], geom["half_extents"]

    def _extract_state(self, env: Any, idx: int = 0) -> dict[str, np.ndarray]:
        """Extract state via TrajectoryProvider protocol."""
        raw_env = getattr(env, "env", env)
        state = raw_env.get_state(idx)
        if not self._state_validated:
            validate_trajectory_state(state, type(raw_env).__name__)
            self._state_validated = True
        return state

    def _get_gates_passed(self, env: Any, idx: int = 0) -> int:
        """Get gates_passed counter from raw env.

        Uses the running counter (_gates_passed or _ep_gates_passed) which
        never wraps, unlike _gate_idx which resets to 0 on lap completion.
        """
        raw_env = getattr(env, "env", env)
        if hasattr(raw_env, "_gates_passed"):
            return int(raw_env._gates_passed[idx])
        if hasattr(raw_env, "_ep_gates_passed"):
            return int(raw_env._ep_gates_passed[idx])
        return 0

    def _get_gate_index(self, env: Any, idx: int = 0) -> int:
        """Get current gate index from raw env."""
        raw_env = getattr(env, "env", env)
        if hasattr(raw_env, "_gate_indices"):
            return int(raw_env._gate_indices[idx])
        if hasattr(raw_env, "_gate_idx"):
            return int(raw_env._gate_idx[idx])
        return 0

    def _get_reward_components(self, env: Any, idx: int = 0) -> np.ndarray:
        """Get per-step reward components via TrajectoryProvider."""
        raw_env = getattr(env, "env", env)
        _, values = raw_env.get_step_reward_components(idx)
        return values

    def _get_reward_component_names(self, env: Any) -> list[str]:
        """Get reward component names via TrajectoryProvider."""
        raw_env = getattr(env, "env", env)
        names, _ = raw_env.get_step_reward_components(0)
        return names

    def _rollout_episode(
        self, env: Any, episode_idx: int, out_dir: Path
    ) -> None:
        """Roll out one deterministic episode and save to .npz."""
        # Handle both raw env (reset returns obs, info) and VecEnv (reset returns obs)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        positions_list: list[np.ndarray] = []
        quaternions_list: list[np.ndarray] = []
        velocities_list: list[np.ndarray] = []
        body_rates_list: list[np.ndarray] = []
        motor_rpms_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        rewards_list: list[float] = []
        reward_components_list: list[np.ndarray] = []
        gate_events_list: list[tuple[int, int]] = []

        prev_gates_passed = self._get_gates_passed(env)
        timestep = 0

        while True:
            # Record state BEFORE taking an action
            state = self._extract_state(env)
            positions_list.append(state["position"])
            quaternions_list.append(state["quaternion"])
            velocities_list.append(state["velocity"])
            body_rates_list.append(state["body_rates"])
            motor_rpms_list.append(state["motor_rpms"])

            # Get deterministic action from current policy
            action, _ = self.model.predict(obs, deterministic=True)

            # Snapshot gate index BEFORE step
            pre_step_gate_idx = self._get_gate_index(env)

            step_result = env.step(action)

            # Handle both raw env (5-tuple) and VecEnv (4-tuple) returns
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = (
                    (bool(terminated) if np.ndim(terminated) == 0 else bool(terminated[0]))
                    or (bool(truncated) if np.ndim(truncated) == 0 else bool(truncated[0]))
                )
            else:
                obs, reward, dones, infos = step_result
                done = bool(dones[0]) if np.ndim(dones) > 0 else bool(dones)
                info = infos[0] if isinstance(infos, list) else infos

            reward_scalar = float(reward) if np.ndim(reward) == 0 else float(reward[0])
            actions_list.append(np.asarray(action, dtype=np.float64).flatten()[:4])
            rewards_list.append(reward_scalar)

            rc = self._get_reward_components(env)
            reward_components_list.append(rc)

            # Detect gate passage events
            if done:
                ep = info.get("episode", {})
                gp = ep.get("gates_passed", 0)
                curr_gates_passed = int(gp[0]) if isinstance(gp, np.ndarray) else int(gp)
            else:
                curr_gates_passed = self._get_gates_passed(env)
            if curr_gates_passed > prev_gates_passed:
                gate_events_list.append((timestep, pre_step_gate_idx))
            prev_gates_passed = curr_gates_passed

            timestep += 1

            if done:
                break

        # Build arrays
        gate_positions, gate_orientations, gate_half_extents = (
            self._extract_gate_geometry(env, env_idx=0)
        )

        gate_events = (
            np.array(gate_events_list, dtype=np.int64)
            if gate_events_list
            else np.zeros((0, 2), dtype=np.int64)
        )

        rc_names = self._get_reward_component_names(env)
        raw_env = getattr(env, "env", env)
        dt = getattr(raw_env, "dt", 0.01)

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"eval_ep_{episode_idx}.npz"

        np.savez_compressed(
            path,
            schema_version=np.int64(SCHEMA_VERSION),
            positions=np.array(positions_list, dtype=np.float64),
            quaternions=np.array(quaternions_list, dtype=np.float64),
            velocities=np.array(velocities_list, dtype=np.float64),
            body_rates=np.array(body_rates_list, dtype=np.float64),
            motor_rpms=np.array(motor_rpms_list, dtype=np.float64),
            actions=np.array(actions_list, dtype=np.float64),
            rewards=np.array(rewards_list, dtype=np.float64),
            reward_components=np.array(reward_components_list, dtype=np.float64),
            reward_component_names=np.array(rc_names, dtype=str) if rc_names else np.array([], dtype=str),
            gate_events=gate_events,
            gate_positions=gate_positions,
            gate_orientations=gate_orientations,
            gate_half_extents=gate_half_extents,
            dt=np.float64(dt),
        )

    # ------------------------------------------------------------------
    # SB3 callback interface
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        total_ts = self.num_timesteps
        out_dir = self.save_path / "trajectories" / f"step_{total_ts}"

        if self.verbose >= 1:
            log.info(
                "TrajectoryRecorder: recording %d episodes at step %d",
                self.n_viz_episodes,
                total_ts,
            )

        env = self._make_eval_env()
        try:
            for ep in range(self.n_viz_episodes):
                self._rollout_episode(env, ep, out_dir)
        finally:
            if hasattr(env, "close"):
                env.close()

        if self.verbose >= 1:
            log.info(
                "TrajectoryRecorder: saved %d episodes to %s",
                self.n_viz_episodes,
                out_dir,
            )

        if self._uploader is not None:
            self._uploader.submit(out_dir, total_ts)

        return True

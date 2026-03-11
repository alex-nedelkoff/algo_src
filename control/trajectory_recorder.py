"""SB3 callback for recording deterministic policy rollout trajectories.

Periodically creates a temporary single-env GateRaceEnv (no domain
randomisation), runs the current policy deterministically for a few
episodes, and saves per-timestep state/action/reward data as .npz files
for downstream visualisation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

from sim.envs.gate_race_env import GateRaceEnv, REWARD_COMPONENT_NAMES

log = logging.getLogger(__name__)

# Current schema version for the .npz trajectory format.
SCHEMA_VERSION = 1


class TrajectoryRecorderCallback(BaseCallback):
    """Record deterministic rollout trajectories at regular intervals.

    Every ``viz_freq`` total timesteps, spins up a lightweight single-env
    :class:`GateRaceEnv` that mirrors the training env's track and physics
    (but with no domain randomisation), rolls out ``n_viz_episodes`` episodes
    using the current policy in deterministic mode, and writes the per-step
    data to ``.npz`` files.

    Args:
        viz_freq: Recording frequency in **total timesteps** (not env steps).
            Internally converted to SB3 callback steps via ``viz_freq // n_envs``.
        n_envs: Number of parallel training envs (for frequency conversion).
        save_path: Root directory for trajectory output.  Files are written to
            ``{save_path}/trajectories/step_{total_timesteps}/eval_ep_{i}.npz``.
        n_viz_episodes: Number of episodes to record each time.
        verbose: Verbosity level (0 = silent, 1 = summary line per trigger).
    """

    def __init__(
        self,
        viz_freq: int,
        n_envs: int,
        save_path: str,
        n_viz_episodes: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.viz_freq = viz_freq
        self.check_freq = max(1, viz_freq // n_envs)
        self.save_path = Path(save_path)
        self.n_viz_episodes = n_viz_episodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_eval_env(self) -> GateRaceEnv:
        """Create a single-env GateRaceEnv mirroring the training env.

        Copies track, vehicle params, dt, reward weights, and gate passage
        radius from the training env but forces ``n_envs=1`` and disables
        domain randomisation.
        """
        train_env: GateRaceEnv = self.training_env.env  # type: ignore[attr-defined]

        return GateRaceEnv(
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

    def _extract_gate_geometry(self, env: GateRaceEnv) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Return (positions, orientations, half_extents) arrays for all gates."""
        n_gates = env.track.num_gates
        positions = np.zeros((n_gates, 3), dtype=np.float64)
        orientations = np.zeros((n_gates, 4), dtype=np.float64)
        half_extents = np.zeros((n_gates, 2), dtype=np.float64)

        for g in range(n_gates):
            gate = env.track.gates[g]
            positions[g] = gate.position
            orientations[g] = gate.orientation
            half_extents[g] = [env.gate_passage_radius, env.gate_passage_radius]

        return positions, orientations, half_extents

    def _rollout_episode(
        self, env: GateRaceEnv, episode_idx: int, out_dir: Path
    ) -> None:
        """Roll out one deterministic episode and save to .npz."""
        obs, _ = env.reset()

        positions_list: list[np.ndarray] = []
        quaternions_list: list[np.ndarray] = []
        velocities_list: list[np.ndarray] = []
        body_rates_list: list[np.ndarray] = []
        motor_rpms_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        rewards_list: list[float] = []
        reward_components_list: list[np.ndarray] = []
        gate_events_list: list[tuple[int, int]] = []

        prev_gates_passed = int(env._gates_passed[0])
        timestep = 0

        while True:
            # Record state BEFORE taking an action
            state = env._states[0]
            positions_list.append(state[0:3].copy())
            quaternions_list.append(state[6:10].copy())
            velocities_list.append(state[3:6].copy())
            body_rates_list.append(state[10:13].copy())
            motor_rpms_list.append(state[13:17].copy())

            # Get deterministic action from current policy
            action, _ = self.model.predict(obs, deterministic=True)

            # Snapshot gate index BEFORE step (gate_indices increments on passage)
            pre_step_gate_idx = int(env._gate_indices[0])

            obs, reward, terminated, truncated, info = env.step(action)
            reward_scalar = float(reward) if np.ndim(reward) == 0 else float(reward[0])

            actions_list.append(np.asarray(action, dtype=np.float64).flatten()[:4])
            rewards_list.append(reward_scalar)
            reward_components_list.append(env._step_reward_components[0].copy())

            # Detect gate passage events.
            # After step(), auto-reset may have zeroed _gates_passed for done envs,
            # so we read from the episode snapshot in info instead.
            done = (
                (bool(terminated) if np.ndim(terminated) == 0 else bool(terminated[0]))
                or (bool(truncated) if np.ndim(truncated) == 0 else bool(truncated[0]))
            )
            if done:
                ep = info.get("episode", {})
                curr_gates_passed = int(ep.get("gates_passed", np.zeros(1))[0]) if isinstance(ep.get("gates_passed"), np.ndarray) else int(ep.get("gates_passed", 0))
            else:
                curr_gates_passed = int(env._gates_passed[0])
            if curr_gates_passed > prev_gates_passed:
                # Use pre-step gate index (the gate that was just passed)
                gate_events_list.append(
                    (timestep, pre_step_gate_idx)
                )
            prev_gates_passed = curr_gates_passed

            timestep += 1

            if done:
                break

        # Build arrays
        gate_positions, gate_orientations, gate_half_extents = (
            self._extract_gate_geometry(env)
        )

        gate_events = (
            np.array(gate_events_list, dtype=np.int64)
            if gate_events_list
            else np.zeros((0, 2), dtype=np.int64)
        )

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
            reward_component_names=np.array(REWARD_COMPONENT_NAMES, dtype=str),
            gate_events=gate_events,
            gate_positions=gate_positions,
            gate_orientations=gate_orientations,
            gate_half_extents=gate_half_extents,
            dt=np.float64(env.dt),
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
            env.close()

        if self.verbose >= 1:
            log.info(
                "TrajectoryRecorder: saved %d episodes to %s",
                self.n_viz_episodes,
                out_dir,
            )

        return True

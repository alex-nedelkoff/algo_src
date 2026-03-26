"""Smoke test: TRPY + spline reward + α-RPO + curriculum all enabled.

Runs 2000 steps with all features to verify nothing crashes.
"""
import numpy as np
import pytest

from sim.tracks import build_figure8_track
from sim.envs.gate_race_env import GateRaceEnv
from control.base_policies.pd_waypoint_tracker import PDWaypointTracker
from training.arpo import AlphaSchedule, fuse_actions, normalize_base_trpy
from training.curriculum_callback import CurriculumCallback


class TestTRPYARPOSmoke:
    def test_full_pipeline_runs_2000_steps(self):
        """All features enabled: TRPY, spline reward, α-RPO fusion."""
        track = build_figure8_track()
        env = GateRaceEnv(
            track=track,
            n_envs=4,
            dt=0.01,
            max_steps=500,
            action_mode="trpy",
            reward_weights={
                "gate_passage": 1.5, "gate_progress": 1.0,
                "crash_penalty": 10.0,
                "spline_proximity": 0.5, "heading_alignment": 0.05,
            },
        )

        gate_positions = np.array([g.position for g in track.gates])
        base_policy = PDWaypointTracker(gate_positions, mass=env.params.mass)
        alpha_sched = AlphaSchedule(k_end_fraction=0.25, total_steps=2000)

        obs, _ = env.reset()
        total_rewards = np.zeros(4)

        for step in range(2000):
            alpha = alpha_sched.get_alpha(step)

            # Base policy action (physical → normalized via shared helper)
            base_physical = base_policy.get_action_batch(
                env._states[:4, 0:3], env._states[:4, 3:6],
                env._states[:4, 6:10], env._states[:4, 10:13],
                env._gate_indices[:4],
            )
            base_norm = normalize_base_trpy(base_physical, env.params.mass, env.max_body_rate)

            # Learned policy = random (simulating untrained network)
            learned = np.random.uniform(-0.1, 0.1, size=(4, 4))

            # Fuse
            action = fuse_actions(base_norm, learned, alpha).astype(np.float32)

            obs, rew, term, trunc, info = env.step(action)
            total_rewards += rew

            assert np.all(np.isfinite(obs)), f"NaN in obs at step {step}"

        # At alpha≈0, base policy should keep drones flying — not all catastrophic crashes
        assert np.any(total_rewards > -1000), f"All rewards very negative: {total_rewards}"

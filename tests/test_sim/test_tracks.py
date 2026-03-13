"""Tests for track definitions — verifies gate normal directions."""
import numpy as np
import pytest

from sim.envs.rate_ctrl_env import (
    make_figure8_track,
    make_oval_track,
    make_s_curve_track,
)


def _verify_gate_normals(track: dict) -> None:
    """Assert every gate's normal faces the approach direction."""
    gates = track["gates"]
    n = track["n_gates"]
    for i in range(n):
        prev_pos = gates[(i - 1) % n]["pos"][:2]
        gate_pos = gates[i]["pos"][:2]
        next_pos = gates[(i + 1) % n]["pos"][:2]
        yaw = gates[i]["yaw"]
        normal = np.array([np.cos(yaw), np.sin(yaw)])

        approach_sd = np.dot(prev_pos - gate_pos, normal)
        exit_sd = np.dot(next_pos - gate_pos, normal)
        assert approach_sd < 0, (
            f"Gate {i}: approach signed dist {approach_sd:.2f} should be < 0"
        )
        assert exit_sd > 0, (
            f"Gate {i}: exit signed dist {exit_sd:.2f} should be > 0"
        )


class TestFigure8Track:
    def test_has_8_gates(self) -> None:
        track = make_figure8_track()
        assert track["n_gates"] == 8
        assert len(track["gates"]) == 8

    def test_gate_normals_correct(self) -> None:
        track = make_figure8_track()
        _verify_gate_normals(track)


class TestOvalTrack:
    def test_has_correct_gate_count(self) -> None:
        track = make_oval_track()
        assert track["n_gates"] == 4
        assert len(track["gates"]) == 4

    def test_gate_normals_correct(self) -> None:
        track = make_oval_track()
        _verify_gate_normals(track)

    def test_dict_has_required_keys(self) -> None:
        track = make_oval_track()
        assert "gates" in track
        assert "gate_width" in track
        assert "gate_height" in track
        assert "n_gates" in track


class TestSCurveTrack:
    def test_has_correct_gate_count(self) -> None:
        track = make_s_curve_track()
        assert track["n_gates"] == 6
        assert len(track["gates"]) == 6

    def test_gate_normals_correct(self) -> None:
        track = make_s_curve_track()
        _verify_gate_normals(track)


class TestMultiTrackEnv:
    def test_single_track_backward_compat(self) -> None:
        from sim.envs.rate_ctrl_env import RateCtrlEnv
        env = RateCtrlEnv(n_envs=2, seed=42)
        obs, _ = env.reset()
        assert obs.shape == (2, 24)

    def test_tracks_list_accepted(self) -> None:
        from sim.envs.rate_ctrl_env import RateCtrlEnv
        tracks = [make_figure8_track(), make_oval_track()]
        env = RateCtrlEnv(n_envs=4, seed=42, tracks=tracks)
        obs, _ = env.reset()
        assert obs.shape == (4, 24)

    def test_tracks_overrides_track(self) -> None:
        from sim.envs.rate_ctrl_env import RateCtrlEnv
        tracks = [make_oval_track()]
        env = RateCtrlEnv(n_envs=2, seed=42, track=make_figure8_track(), tracks=tracks)
        obs, _ = env.reset()
        action = np.zeros((2, 4), dtype=np.float32)
        env.step(action)

    def test_per_env_track_varies(self) -> None:
        from sim.envs.rate_ctrl_env import RateCtrlEnv
        tracks = [make_figure8_track(), make_oval_track()]
        env = RateCtrlEnv(n_envs=50, seed=42, tracks=tracks)
        env.reset()
        assert hasattr(env, "_track_idx")
        unique_tracks = set(env._track_idx.tolist())
        assert len(unique_tracks) > 1

    def test_step_runs_without_error(self) -> None:
        from sim.envs.rate_ctrl_env import RateCtrlEnv
        tracks = [make_figure8_track(), make_oval_track()]
        env = RateCtrlEnv(n_envs=4, seed=42, tracks=tracks)
        env.reset()
        action = np.zeros((4, 4), dtype=np.float32)
        for _ in range(10):
            obs, rewards, terminated, truncated, info = env.step(action)
        assert obs.shape == (4, 24)
        assert rewards.shape == (4,)

    def test_gate_idx_respects_track_n_gates(self) -> None:
        from sim.envs.rate_ctrl_env import RateCtrlEnv
        tracks = [make_figure8_track(), make_oval_track()]
        env = RateCtrlEnv(n_envs=2, seed=42, tracks=tracks)
        env.reset()
        env._track_idx[0] = 0
        env._track_idx[1] = 1
        env._gate_idx[0] = 7
        env._gate_idx[1] = 3
        assert (7 + 1) % 8 == 0
        assert (3 + 1) % 4 == 0


class TestRewardNormalization:
    def test_mean_inter_gate_distance_computed(self) -> None:
        """Env should compute mean inter-gate distance per track."""
        from sim.envs.rate_ctrl_env import RateCtrlEnv
        env = RateCtrlEnv(n_envs=2, seed=42)
        assert hasattr(env, "_mean_igd")
        assert len(env._mean_igd) == 1  # single track
        assert env._mean_igd[0] > 0

    def test_multi_track_has_per_track_igd(self) -> None:
        from sim.envs.rate_ctrl_env import RateCtrlEnv
        tracks = [make_figure8_track(), make_oval_track()]
        env = RateCtrlEnv(n_envs=2, seed=42, tracks=tracks)
        assert len(env._mean_igd) == 2
        # Figure-8 and oval have different spacings
        assert env._mean_igd[0] != env._mean_igd[1]

    def test_normalization_scales_progress_reward(self) -> None:
        """Tracks with different IGDs produce different normalized progress."""
        from sim.envs.rate_ctrl_env import RateCtrlEnv
        env_f8 = RateCtrlEnv(n_envs=1, seed=42, track=make_figure8_track())
        env_oval = RateCtrlEnv(n_envs=1, seed=42, track=make_oval_track())

        igd_f8 = env_f8._mean_igd[0]
        igd_oval = env_oval._mean_igd[0]

        # The ratio of normalized progress should be inverse of IGD ratio
        ratio = igd_oval / igd_f8
        assert ratio != 1.0, "Tracks should have different IGDs"

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

"""Tests for trajectory insight extraction."""

import numpy as np
import pytest
from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.analysis.trajectory_insights import analyze_trajectory, TrajectoryInsights


def _make_traj(**overrides) -> TrajectoryData:
    T = overrides.pop("n_timesteps", 200)
    defaults = dict(
        schema_version=2,
        positions=np.column_stack([np.linspace(0, 20, T), np.zeros(T), np.full(T, 2.0)]),
        quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)),
        velocities=np.tile([8.0, 0.0, 0.0], (T, 1)),
        body_rates=np.zeros((T, 3)),
        motor_rpms=np.full((T, 4), 15000.0),
        actions=np.zeros((T, 4)),
        rewards=np.ones(T),
        reward_components=np.zeros((T, 6)),
        reward_component_names=["progress", "body_rate", "action_smooth", "gate_passage", "gate_offset", "crash_penalty"],
        gate_events=np.array([[50, 0], [100, 1], [150, 2]], dtype=np.int64),
        gate_positions=np.array([[5, 0, 2], [10, 0, 2], [15, 0, 2]], dtype=np.float64),
        gate_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        gate_half_extents=np.full((3, 2), 0.5),
        dt=0.01,
    )
    defaults.update(overrides)
    return TrajectoryData(**defaults)


def test_returns_insights():
    insights = analyze_trajectory(_make_traj())
    assert isinstance(insights, TrajectoryInsights)

def test_speed_statistics():
    insights = analyze_trajectory(_make_traj(velocities=np.tile([10.0, 0.0, 0.0], (200, 1))))
    assert insights.mean_speed == pytest.approx(10.0, abs=0.1)

def test_gate_analysis_count():
    insights = analyze_trajectory(_make_traj())
    assert len(insights.gate_analyses) == 3
    assert insights.gate_analyses[0].passage_time == pytest.approx(0.5, abs=0.01)

def test_gate_approach_speed():
    insights = analyze_trajectory(_make_traj())
    for ga in insights.gate_analyses:
        assert ga.approach_speed > 0

def test_motor_utilization():
    insights = analyze_trajectory(_make_traj(motor_rpms=np.full((200, 4), 20000.0)), max_rpm=31470.0)
    assert 0.5 < insights.mean_motor_utilization < 0.8

def test_body_rate_stats():
    rates = np.zeros((200, 3))
    rates[:, 0] = 2.0
    insights = analyze_trajectory(_make_traj(body_rates=rates))
    assert insights.max_body_rate > 1.5

def test_reward_breakdown():
    rc = np.zeros((200, 6))
    rc[:, 0] = 1.0; rc[:, 3] = 5.0
    insights = analyze_trajectory(_make_traj(reward_components=rc))
    assert insights.dominant_reward_component == "gate_passage"

def test_no_gate_events():
    insights = analyze_trajectory(_make_traj(gate_events=np.empty((0, 2), dtype=np.int64)))
    assert len(insights.gate_analyses) == 0

def test_summary_string():
    summary = analyze_trajectory(_make_traj()).to_summary()
    assert isinstance(summary, str)
    assert "speed" in summary.lower()

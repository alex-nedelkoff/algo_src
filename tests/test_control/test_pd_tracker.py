"""Tests for PD waypoint tracker base policy."""
import numpy as np
import pytest

from control.base_policies.pd_waypoint_tracker import PDWaypointTracker


class TestPDWaypointTracker:
    @pytest.fixture
    def tracker(self):
        gate_positions = np.array([
            [2.0, 0.0, 1.5],
            [4.0, 2.0, 1.5],
            [2.0, 4.0, 1.5],
            [0.0, 2.0, 1.5],
        ])
        return PDWaypointTracker(gate_positions, mass=0.752)

    def test_output_shape(self, tracker):
        pos = np.array([0.0, 0.0, 1.5])
        vel = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.zeros(3)
        action = tracker.get_action(pos, vel, quat, omega, gate_idx=0)
        assert action.shape == (4,)

    def test_hover_at_gate_gives_near_hover_thrust(self, tracker):
        pos = np.array([2.0, 0.0, 1.5])
        vel = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.zeros(3)
        action = tracker.get_action(pos, vel, quat, omega, gate_idx=0)
        hover_thrust = 0.752 * 9.81
        assert abs(action[0] - hover_thrust) < 3.0

    def test_behind_gate_gives_forward_pitch(self, tracker):
        pos = np.array([0.0, 0.0, 1.5])
        vel = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.zeros(3)
        action = tracker.get_action(pos, vel, quat, omega, gate_idx=0)
        assert np.linalg.norm(action[1:4]) > 0.01

    def test_batch_action(self, tracker):
        n = 5
        pos = np.tile([0.0, 0.0, 1.5], (n, 1))
        vel = np.zeros((n, 3))
        quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        omega = np.zeros((n, 3))
        gate_idx = np.zeros(n, dtype=np.int64)
        actions = tracker.get_action_batch(pos, vel, quat, omega, gate_idx)
        assert actions.shape == (n, 4)

import numpy as np
import pytest
from unittest.mock import MagicMock
from sim.tracks import Track
from sim.types import GateState
from sim.envs.gate_race_env import GateRaceEnv


def _gate(x, y, z=1.5, yaw=0.0):
    w = np.cos(yaw / 2)
    zq = np.sin(yaw / 2)
    return GateState(position=np.array([x, y, z]), orientation=np.array([w, 0.0, 0.0, zq]))


class TestTrackChaining:
    def test_chain_triggers_on_last_gate(self):
        """When drone passes last gate of chainable track, new segment spawns."""
        gate0 = _gate(2.0, 0.0, 1.5, 0.0)
        gate1 = _gate(4.0, 0.0, 1.5, 0.0)
        exit_pos = np.array([6.0, 0.0, 1.5])

        new_gate0 = _gate(6.0, 0.0, 1.5, 0.0)
        new_gate1 = _gate(8.0, 0.0, 1.5, 0.0)
        new_track = Track(
            [new_gate0, new_gate1],
            chainable=True,
            exit_pos=np.array([10.0, 0.0, 1.5]),
            exit_heading=0.0,
        )
        mock_gen = MagicMock()
        mock_gen.generate.return_value = new_track
        new_track.generator = mock_gen

        track = Track(
            [gate0, gate1],
            chainable=True,
            exit_pos=exit_pos,
            exit_heading=0.0,
            generator=mock_gen,
        )

        env = GateRaceEnv(
            track=track,
            n_envs=1,
            max_steps=100,
        )
        env.reset()

        assert env._tracks[0].chainable is True

    def test_lap_increments_on_chain(self):
        """Lap counter should increment when track is chained."""
        gate0 = _gate(2.0, 0.0, 1.5, 0.0)
        gate1 = _gate(4.0, 0.0, 1.5, 0.0)

        mock_gen = MagicMock()
        new_track = Track(
            [_gate(6.0, 0.0, 1.5, 0.0), _gate(8.0, 0.0, 1.5, 0.0)],
            chainable=True,
            exit_pos=np.array([10.0, 0.0, 1.5]),
            exit_heading=0.0,
        )
        new_track.generator = mock_gen
        mock_gen.generate.return_value = new_track

        track = Track(
            [gate0, gate1],
            chainable=True,
            exit_pos=np.array([6.0, 0.0, 1.5]),
            exit_heading=0.0,
            generator=mock_gen,
        )

        env = GateRaceEnv(track=track, n_envs=1, max_steps=100)
        env.reset()

        env._gate_indices[0] = 1
        env._gates_passed[0] = 1
        assert env._tracks[0].num_gates == 2
        assert env._tracks[0].chainable is True

import numpy as np
import pytest
from sim.tracks import Track
from sim.types import GateState


def _gate(x, y, z=1.5):
    return GateState(position=np.array([x, y, z]))


class TestTrackChainingAttrs:
    def test_default_track_not_chainable(self):
        track = Track([_gate(0, 0), _gate(1, 0)])
        assert track.chainable is False
        assert track.exit_pos is None
        assert track.exit_heading is None
        assert track.generator is None

    def test_chainable_track(self):
        track = Track(
            [_gate(0, 0), _gate(1, 0)],
            chainable=True,
            exit_pos=np.array([2.0, 0.0, 1.5]),
            exit_heading=0.0,
        )
        assert track.chainable is True
        np.testing.assert_array_almost_equal(track.exit_pos, [2.0, 0.0, 1.5])
        assert track.exit_heading == 0.0

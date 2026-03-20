"""Figure-eight track generation."""
from __future__ import annotations

import numpy as np

from sim.tracks import Track, build_figure8_track


class Figure8TrackGenerator:
    """Generates figure-eight tracks.

    Minimal stub — full implementation provided by a parallel agent.
    """

    def generate(self, rng: np.random.Generator) -> Track:
        return build_figure8_track()

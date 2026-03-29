import numpy as np
import pytest
from sim.procedural_tracks import ProceduralTrackGenerator


class TestElevationOscillation:
    def test_oscillation_zero_is_random(self):
        """With oscillation=0, delta_z signs should be roughly balanced."""
        gen = ProceduralTrackGenerator(
            n_gates_min=6, n_gates_max=6,
            elevation_oscillation=0.0,
        )
        rng = np.random.default_rng(42)
        sign_changes = 0
        total = 0
        for _ in range(50):
            track = gen.generate(rng)
            zs = [g.position[2] for g in track.gates]
            for i in range(2, len(zs)):
                d_prev = zs[i - 1] - zs[i - 2]
                d_curr = zs[i] - zs[i - 1]
                if abs(d_prev) > 0.01 and abs(d_curr) > 0.01:
                    total += 1
                    if np.sign(d_prev) != np.sign(d_curr):
                        sign_changes += 1
        ratio = sign_changes / total if total > 0 else 0
        assert 0.25 < ratio < 0.75, f"Expected ~50% sign changes, got {ratio:.1%}"

    def test_oscillation_high_forces_alternating(self):
        """With oscillation=1.0, consecutive delta_z should mostly alternate sign."""
        gen = ProceduralTrackGenerator(
            n_gates_min=6, n_gates_max=6,
            elevation_oscillation=1.0,
        )
        rng = np.random.default_rng(42)
        sign_changes = 0
        total = 0
        for _ in range(50):
            track = gen.generate(rng)
            zs = [g.position[2] for g in track.gates]
            for i in range(2, len(zs)):
                d_prev = zs[i - 1] - zs[i - 2]
                d_curr = zs[i] - zs[i - 1]
                if abs(d_prev) > 0.01 and abs(d_curr) > 0.01:
                    total += 1
                    if np.sign(d_prev) != np.sign(d_curr):
                        sign_changes += 1
        ratio = sign_changes / total if total > 0 else 0
        assert ratio > 0.75, f"Expected >75% sign changes with oscillation=1.0, got {ratio:.1%}"

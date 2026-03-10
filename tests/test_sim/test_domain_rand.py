"""Tests for DomainRandomizer.

Covers bounds enforcement, seed diversity, and identity (no-op) behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.domain_randomization import DomainRandomizer
from sim.dynamics.params import VehicleParams


@pytest.fixture
def nominal_params() -> VehicleParams:
    """Default CrazyFlie 2.1 parameters."""
    return VehicleParams()


class TestBoundsEnforcement:
    """Randomized params should stay within configured bounds."""

    def test_scalar_params_within_bounds(self, nominal_params: VehicleParams) -> None:
        """Scalar parameters stay within [nominal*(1-frac), nominal*(1+frac)]."""
        frac = 0.2  # +/- 20%
        config = {
            "mass": (frac, frac),
            "arm_length": (frac, frac),
            "k_thrust": (frac, frac),
            "k_torque": (frac, frac),
            "tau_motor": (frac, frac),
        }
        randomizer = DomainRandomizer(config)

        rng = np.random.default_rng(42)
        for _ in range(100):
            p = randomizer.apply(nominal_params, rng)

            assert nominal_params.mass * (1 - frac) <= p.mass <= nominal_params.mass * (1 + frac)
            assert (
                nominal_params.arm_length * (1 - frac)
                <= p.arm_length
                <= nominal_params.arm_length * (1 + frac)
            )
            assert (
                nominal_params.k_thrust * (1 - frac)
                <= p.k_thrust
                <= nominal_params.k_thrust * (1 + frac)
            )
            assert (
                nominal_params.k_torque * (1 - frac)
                <= p.k_torque
                <= nominal_params.k_torque * (1 + frac)
            )
            assert (
                nominal_params.tau_motor * (1 - frac)
                <= p.tau_motor
                <= nominal_params.tau_motor * (1 + frac)
            )

    def test_inertia_within_bounds(self, nominal_params: VehicleParams) -> None:
        """Inertia diagonal elements stay within bounds."""
        frac = 0.15
        randomizer = DomainRandomizer({"inertia": (frac, frac)})
        rng = np.random.default_rng(123)

        nominal_diag = np.diag(nominal_params.inertia)

        for _ in range(100):
            p = randomizer.apply(nominal_params, rng)
            rand_diag = np.diag(p.inertia)

            for j in range(3):
                lo = nominal_diag[j] * (1 - frac)
                hi = nominal_diag[j] * (1 + frac)
                assert lo <= rand_diag[j] <= hi, (
                    f"Inertia[{j}]={rand_diag[j]} outside [{lo}, {hi}]"
                )


class TestSeedDiversity:
    """Different seeds should produce different parameters."""

    def test_different_seeds_different_params(self, nominal_params: VehicleParams) -> None:
        """Two different seeds produce different mass values."""
        randomizer = DomainRandomizer.from_percentage(0.2)

        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)

        p1 = randomizer.apply(nominal_params, rng1)
        p2 = randomizer.apply(nominal_params, rng2)

        # At least one param should differ
        assert p1.mass != p2.mass or p1.k_thrust != p2.k_thrust

    def test_same_seed_same_params(self, nominal_params: VehicleParams) -> None:
        """Same seed produces identical parameters."""
        randomizer = DomainRandomizer.from_percentage(0.2)

        p1 = randomizer.apply(nominal_params, np.random.default_rng(42))
        p2 = randomizer.apply(nominal_params, np.random.default_rng(42))

        assert p1.mass == p2.mass
        assert p1.k_thrust == p2.k_thrust
        assert p1.arm_length == p2.arm_length


class TestIdentityRandomizer:
    """No-randomization config should leave parameters unchanged."""

    def test_empty_config_identity(self, nominal_params: VehicleParams) -> None:
        """Empty config produces identical parameters."""
        randomizer = DomainRandomizer.identity()
        rng = np.random.default_rng(42)
        p = randomizer.apply(nominal_params, rng)

        assert p.mass == nominal_params.mass
        assert p.arm_length == nominal_params.arm_length
        assert p.k_thrust == nominal_params.k_thrust
        assert p.k_torque == nominal_params.k_torque
        assert p.tau_motor == nominal_params.tau_motor
        np.testing.assert_array_equal(p.inertia, nominal_params.inertia)
        np.testing.assert_array_equal(p.drag_coeff, nominal_params.drag_coeff)

    def test_identity_deep_copy(self, nominal_params: VehicleParams) -> None:
        """Identity randomizer returns a deep copy (not same object)."""
        randomizer = DomainRandomizer.identity()
        p = randomizer.apply(nominal_params)

        # Should be equal but not the same object
        assert p.mass == nominal_params.mass
        assert p is not nominal_params

        # Modifying the copy should not affect the original
        p.mass = 999.0
        assert nominal_params.mass != 999.0


class TestInvalidConfig:
    """Invalid parameter names should raise errors."""

    def test_unknown_param_raises(self) -> None:
        """Unknown parameter name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            DomainRandomizer({"bogus_param": (0.1, 0.1)})

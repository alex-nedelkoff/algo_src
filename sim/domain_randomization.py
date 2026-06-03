"""Domain randomization for sim-to-real transfer.

Randomizes vehicle parameters per-episode to build robust policies.
Follows the MAVLab approach: percentage-based uniform sampling around nominal values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.dynamics.params import VehicleParams

# Scalar parameters that can be randomized
SCALAR_PARAMS = {"mass", "arm_length", "k_thrust", "k_torque", "tau_motor", "prop_radius", "max_rpm"}

# Array parameters that can be randomized element-wise
ARRAY_PARAMS = {"drag_coeff", "linear_drag_coeff", "weathervane_coeff"}

# Matrix parameters: 'inertia' randomizes the diagonal elements
MATRIX_PARAMS = {"inertia"}

ALL_PARAMS = SCALAR_PARAMS | ARRAY_PARAMS | MATRIX_PARAMS


class DomainRandomizer:
    """Randomizes vehicle parameters within configured bounds.

    Takes a config dict mapping parameter names to (lo_fraction, hi_fraction) tuples,
    where the randomized value is sampled from
        U[nominal * (1 - lo_fraction), nominal * (1 + hi_fraction)]

    Alternatively, bounds can be given as absolute (lo, hi) pairs by setting
    ``absolute_bounds=True``.

    Example config (percentage-based, default)::

        config = {
            "mass": (0.1, 0.1),          # +/- 10%
            "k_thrust": (0.2, 0.2),      # +/- 20%
            "inertia": (0.1, 0.1),       # +/- 10% on diagonal
        }

    Args:
        config: Dict mapping param names to (lo, hi) bound tuples.
        absolute_bounds: If True, interpret bounds as absolute (lo, hi) values
            rather than fractional deviations from nominal.
    """

    def __init__(
        self,
        config: dict[str, tuple[float, float] | float],
        absolute_bounds: bool = False,
    ) -> None:
        # Normalize single floats to symmetric (p, p) tuples
        normalized: dict[str, tuple[float, float]] = {}
        for key, value in config.items():
            if key not in ALL_PARAMS:
                raise ValueError(
                    f"Unknown parameter '{key}'. Valid params: {sorted(ALL_PARAMS)}"
                )
            if isinstance(value, (int, float)):
                normalized[key] = (float(value), float(value))
            else:
                normalized[key] = (float(value[0]), float(value[1]))
        self.config = normalized
        self.absolute_bounds = absolute_bounds

    def apply(
        self,
        params: VehicleParams,
        rng: np.random.Generator | None = None,
    ) -> VehicleParams:
        """Return a randomized copy of the vehicle parameters.

        Args:
            params: Nominal vehicle parameters.
            rng: Random number generator. Creates a new default one if None.

        Returns:
            New VehicleParams with randomized values.
        """
        rng = rng or np.random.default_rng()
        new_params = params.copy()

        for name, bounds in self.config.items():
            lo, hi = bounds

            if name in SCALAR_PARAMS:
                nominal = getattr(params, name)
                if self.absolute_bounds:
                    new_val = rng.uniform(lo, hi)
                else:
                    new_val = rng.uniform(nominal * (1.0 - lo), nominal * (1.0 + hi))
                setattr(new_params, name, new_val)

            elif name in ARRAY_PARAMS:
                nominal = getattr(params, name).copy()
                if self.absolute_bounds:
                    new_val = rng.uniform(lo, hi, size=nominal.shape)
                else:
                    new_val = rng.uniform(
                        nominal * (1.0 - lo), nominal * (1.0 + hi)
                    )
                setattr(new_params, name, new_val)

            elif name == "inertia":
                # Randomize diagonal elements of the inertia tensor
                diag = np.diag(params.inertia).copy()
                if self.absolute_bounds:
                    new_diag = rng.uniform(lo, hi, size=diag.shape)
                else:
                    new_diag = rng.uniform(diag * (1.0 - lo), diag * (1.0 + hi))
                new_params.inertia = np.diag(new_diag)

        return new_params

    @classmethod
    def from_percentage(cls, percentage: float, params: list[str] | None = None) -> DomainRandomizer:
        """Create a randomizer with uniform percentage bounds on all (or selected) params.

        Args:
            percentage: Fractional deviation (e.g. 0.1 for +/- 10%).
            params: List of parameter names. Defaults to all scalar params.

        Returns:
            Configured DomainRandomizer.
        """
        if params is None:
            params_list = sorted(SCALAR_PARAMS)
        else:
            params_list = params
        config: dict[str, tuple[float, float]] = {
            name: (percentage, percentage) for name in params_list
        }
        return cls(config)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DomainRandomizer:
        """Create a DomainRandomizer from a YAML-style config dict.

        Expected format::

            {
                "enabled": True,
                "params": {
                    "mass": 0.3,              # single float = symmetric +-30%
                    "k_thrust": (0.2, 0.4),   # tuple = asymmetric
                }
            }

        If ``enabled`` is False or ``params`` is empty, returns an identity
        (no-op) randomizer.

        Args:
            config: Dict with 'enabled' bool and 'params' dict.

        Returns:
            Configured DomainRandomizer.
        """
        if not config.get("enabled", False):
            return cls.identity()
        params = config.get("params", {})
        if not params:
            return cls.identity()
        return cls(params)

    @classmethod
    def identity(cls) -> DomainRandomizer:
        """Create a no-op randomizer that leaves parameters unchanged.

        Returns:
            DomainRandomizer with empty config.
        """
        return cls({})

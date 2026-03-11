"""Vehicle parameters for quadrotor dynamics simulation.

Default values from CrazyFlie 2.1 system identification.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class VehicleParams:
    """Physical parameters for a quadrotor vehicle.

    Default values correspond to CrazyFlie 2.1 sysid data.
    Motor layout (X-configuration, viewed from above):
        Motor 0: front-right (CW)
        Motor 1: rear-right  (CCW)
        Motor 2: rear-left   (CW)
        Motor 3: front-left  (CCW)

    Attributes:
        mass: Total vehicle mass in kg.
        inertia: 3x3 inertia tensor in kg*m^2.
        arm_length: Distance from center to motor in m.
        k_thrust: Thrust coefficient (N / (rad/s)^2).
        k_torque: Torque coefficient (N*m / (rad/s)^2).
        tau_motor: First-order motor time constant in s.
        prop_radius: Propeller radius in m.
        drag_coeff: Body-frame drag coefficients [dx, dy, dz].
        max_rpm: Maximum motor speed in RPM.
    """

    mass: float = 0.027
    inertia: NDArray[np.float64] = field(
        default_factory=lambda: np.diag([1.395e-5, 1.436e-5, 2.173e-5])
    )
    arm_length: float = 0.0397
    k_thrust: float = 2.245e-8
    k_torque: float = 7.94e-10
    tau_motor: float = 0.02
    prop_radius: float = 0.023
    drag_coeff: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    max_rpm: float = 21702.0

    def __post_init__(self) -> None:
        """Validate and convert fields to numpy arrays."""
        self.inertia = np.asarray(self.inertia, dtype=np.float64)
        self.drag_coeff = np.asarray(self.drag_coeff, dtype=np.float64)

        # Accept 3-element diagonal shorthand: [Jxx, Jyy, Jzz] -> diag matrix
        if self.inertia.shape == (3,):
            self.inertia = np.diag(self.inertia)
        if self.inertia.shape != (3, 3):
            raise ValueError(f"inertia must have shape (3, 3) or (3,), got {self.inertia.shape}")
        if self.drag_coeff.shape != (3,):
            raise ValueError(f"drag_coeff must have shape (3,), got {self.drag_coeff.shape}")
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        if self.arm_length <= 0:
            raise ValueError(f"arm_length must be positive, got {self.arm_length}")

    @property
    def max_omega(self) -> float:
        """Maximum motor speed in rad/s."""
        return self.max_rpm * 2.0 * np.pi / 60.0

    def copy(self) -> VehicleParams:
        """Return a deep copy of this parameter set."""
        return copy.deepcopy(self)

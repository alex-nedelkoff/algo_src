"""Core data types for the drone racing stack.

QuadState uses quaternion (w, x, y, z) convention internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ActionMode(Enum):
    """Action parameterization mode."""

    MOTOR_RPM = "motor_rpm"
    TRPY = "trpy"  # thrust, roll, pitch, yaw


@dataclass
class QuadState:
    """16-state quadrotor state representation.

    State vector layout:
        pos:          [x, y, z]           (3) - position in world frame
        vel:          [vx, vy, vz]        (3) - velocity in world frame
        quat:         [w, x, y, z]        (4) - orientation quaternion
        omega:        [wx, wy, wz]        (3) - angular velocity in body frame
        motor_speeds: [m1, m2, m3, m4]    (4) - motor speeds (RPM or rad/s)

    Total: 16 states (pos=3 + vel=3 + quat=4 + omega=3 + motor_speeds=4 = 17).
    Note: The "16-state" name reflects that the quaternion has only 3 DOF due to
    the unit-norm constraint; the state vector itself has 17 elements.

    Quaternion is normalized on construction.
    """

    pos: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    vel: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    quat: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    omega: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    motor_speeds: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))

    def __post_init__(self) -> None:
        """Validate dimensions and normalize quaternion."""
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.vel = np.asarray(self.vel, dtype=np.float64)
        self.quat = np.asarray(self.quat, dtype=np.float64)
        self.omega = np.asarray(self.omega, dtype=np.float64)
        self.motor_speeds = np.asarray(self.motor_speeds, dtype=np.float64)

        if self.pos.shape != (3,):
            raise ValueError(f"pos must have shape (3,), got {self.pos.shape}")
        if self.vel.shape != (3,):
            raise ValueError(f"vel must have shape (3,), got {self.vel.shape}")
        if self.quat.shape != (4,):
            raise ValueError(f"quat must have shape (4,), got {self.quat.shape}")
        if self.omega.shape != (3,):
            raise ValueError(f"omega must have shape (3,), got {self.omega.shape}")
        if self.motor_speeds.shape != (4,):
            raise ValueError(f"motor_speeds must have shape (4,), got {self.motor_speeds.shape}")

        # Normalize quaternion
        quat_norm = np.linalg.norm(self.quat)
        if quat_norm < 1e-10:
            raise ValueError("Quaternion norm is near zero, cannot normalize")
        self.quat = self.quat / quat_norm

    def to_vector(self) -> NDArray[np.float64]:
        """Flatten state to a 17-element vector."""
        return np.concatenate([self.pos, self.vel, self.quat, self.omega, self.motor_speeds])

    @classmethod
    def from_vector(cls, vec: NDArray[np.float64]) -> QuadState:
        """Construct QuadState from a 17-element vector."""
        vec = np.asarray(vec, dtype=np.float64)
        if vec.shape != (17,):
            raise ValueError(f"Expected 17-element vector, got shape {vec.shape}")
        return cls(
            pos=vec[0:3],
            vel=vec[3:6],
            quat=vec[6:10],
            omega=vec[10:13],
            motor_speeds=vec[13:17],
        )


@dataclass
class Observation:
    """Observation provided to the policy.

    Contains the current state and any additional sensor data.
    """

    state: QuadState
    image: NDArray[np.uint8] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Action output from the policy.

    Can represent either 4 motor RPMs or thrust/roll/pitch/yaw commands.
    """

    values: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    mode: ActionMode = ActionMode.MOTOR_RPM

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.shape != (4,):
            raise ValueError(f"Action values must have shape (4,), got {self.values.shape}")


@dataclass
class GateState:
    """State of a single gate in the track.

    Attributes:
        position: [x, y, z] center of the gate in world frame.
        orientation: [w, x, y, z] quaternion representing gate normal direction.
    """

    position: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    orientation: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)

        if self.position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {self.position.shape}")
        if self.orientation.shape != (4,):
            raise ValueError(f"orientation must have shape (4,), got {self.orientation.shape}")

        # Normalize orientation quaternion
        quat_norm = np.linalg.norm(self.orientation)
        if quat_norm < 1e-10:
            raise ValueError("Orientation quaternion norm is near zero, cannot normalize")
        self.orientation = self.orientation / quat_norm

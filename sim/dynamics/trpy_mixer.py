"""TRPY command → motor speed mixer for quadrotor X-configuration.

Converts collective thrust + body rate commands into individual motor speeds
using the standard quadrotor allocation matrix. Used when the policy outputs
TRPY commands (for Anduril Grand Prix ESC interface) instead of direct motor RPMs.

Allocation for X-config (motor index → label/spin/body-frame position):
    M1 = FR-CW   at (+L/√2, -L/√2, 0)
    M2 = FL-CCW  at (+L/√2, +L/√2, 0)
    M3 = RL-CW   at (-L/√2, +L/√2, 0)
    M4 = RR-CCW  at (-L/√2, -L/√2, 0)

    T_total = k_t * (w1² + w2² + w3² + w4²)
    τ_roll  = k_t * L/√2 * (-w1² + w2² + w3² - w4²)
    τ_pitch = k_t * L/√2 * (-w1² - w2² + w3² + w4²)
    τ_yaw   = k_q * (w1² - w2² + w3² - w4²)

Note: prior versions of this docstring listed M2=RR, M4=FL, which contradicted
the allocation matrix. The matrix is authoritative; positions above are derived
from it (see PyBulletBackend for the consuming side).

We invert this to get w² from [T, τ_roll, τ_pitch, τ_yaw], then scale by the
hover equilibrium speed to obtain motor speeds (rad/s). This linearised form
preserves the exact antisymmetry required for body-rate commands while mapping
zero thrust to zero motor speed and hover thrust to equal hover motor speeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams

_SQRT2_INV = 1.0 / np.sqrt(2.0)
_GRAVITY = 9.81  # m/s²


class TRPYMixer:
    """Convert [thrust, ωx_cmd, ωy_cmd, ωz_cmd] → motor speeds (rad/s).

    The mixer is linearised around hover equilibrium: the quadrotor allocation
    matrix (which maps motor w² to wrench) is inverted and the result is divided
    by the hover equilibrium motor speed. This gives a linear map from wrench to
    motor speeds that satisfies:
      - zero thrust → zero motor speeds
      - hover thrust, zero rates → equal motor speeds at hover equilibrium
      - exact antisymmetry for positive/negative rate commands

    Args:
        params: Vehicle parameters (mass, inertia, arm_length, k_thrust, k_torque).
        rate_gain: Proportional gain for body rate → torque conversion.
    """

    def __init__(self, params: VehicleParams, rate_gain: float = 1.0) -> None:
        self.params = params
        self.rate_gain = rate_gain

        k_t = params.k_thrust
        k_q = params.k_torque
        L = params.arm_length
        Lk = L * k_t * _SQRT2_INV

        # Allocation matrix mapping [w1², w2², w3², w4²] → [T, τ_roll, τ_pitch, τ_yaw]
        self._alloc = np.array([
            [k_t,   k_t,   k_t,   k_t  ],
            [-Lk,   Lk,    Lk,   -Lk   ],
            [-Lk,  -Lk,    Lk,    Lk   ],
            [k_q,  -k_q,   k_q,  -k_q  ],
        ])
        self._alloc_inv = np.linalg.inv(self._alloc)
        self._max_omega = params.max_omega

        # Hover equilibrium motor speed: w_hover = sqrt(m*g / (4*k_t))
        # Used to convert w_sq-space output to motor speed in rad/s.
        hover_thrust = params.mass * _GRAVITY
        self._w_hover = np.sqrt(hover_thrust / (4.0 * k_t))

        # inertia is always 3x3 per VehicleParams.__post_init__; extract diagonal
        self._inertia_diag = np.diag(params.inertia) if params.inertia.ndim == 2 else params.inertia

    def mix(self, trpy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert single TRPY command to motor speeds.

        Applies the inverse allocation matrix to [thrust, τ_roll, τ_pitch, τ_yaw]
        and scales by 1/w_hover to convert from w²-space to w-space. The result
        is clipped to [0, max_omega].

        Args:
            trpy: [thrust_N, ωx_cmd, ωy_cmd, ωz_cmd] shape (4,).

        Returns:
            Motor speeds in rad/s, shape (4,), clipped to [0, max_omega].
        """
        thrust = trpy[0]
        omega_cmd = trpy[1:4]
        desired_torques = self._inertia_diag * self.rate_gain * omega_cmd
        wrench = np.array([thrust, desired_torques[0], desired_torques[1], desired_torques[2]])
        w = self._alloc_inv @ wrench / self._w_hover
        return np.clip(w, 0.0, self._max_omega)

    def mix_batch(self, trpy_batch: NDArray[np.float64]) -> NDArray[np.float64]:
        """Batch version: (N, 4) TRPY → (N, 4) motor speeds."""
        thrust = trpy_batch[:, 0]
        omega_cmd = trpy_batch[:, 1:4]
        desired_torques = omega_cmd * self._inertia_diag * self.rate_gain
        wrench = np.column_stack([thrust, desired_torques])
        w = (self._alloc_inv @ wrench.T).T / self._w_hover
        return np.clip(w, 0.0, self._max_omega)

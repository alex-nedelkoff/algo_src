"""Single-loop attitude P controller feeding the TRPY mixer.

For SET_ATTITUDE_TARGET (q_target, thrust_normalized):
  1. Compute quaternion error in body frame.
  2. P controller on error vector → desired body rates ω_des.
  3. Hand (thrust_n, ω_des) to the existing TRPY mixer for motor speeds.

No inner rate PD: the TRPY mixer's inverted allocation matrix maps desired
rates directly to motor speeds. For a perfect-model sim this is sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams
from sim.dynamics.trpy_mixer import TRPYMixer


def _quat_multiply(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_inverse_unit(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of a unit quaternion (= conjugate)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


@dataclass
class AttitudeController:
    """Quat-error PD controller. Outputs motor speeds via TRPYMixer.

    The D term damps current body rates, preventing the overshoot that a
    pure-P outer loop exhibits in free-running (real-time) mode.
    """
    params: VehicleParams
    k_att: float = 6.0       # attitude P gain (rad/s per rad of error)
    k_damp: float = 0.0      # rate damping gain (subtracts k_damp * omega_body from omega_cmd)

    def __post_init__(self) -> None:
        self._mixer = TRPYMixer(self.params)
        # Total max collective thrust (all 4 motors at max).
        # VehicleParams has no max_thrust_per_motor_n; derive from k_thrust and max_omega.
        max_thrust_per_motor = self.params.k_thrust * self.params.max_omega ** 2
        self._max_thrust_n = 4.0 * max_thrust_per_motor

    def compute(
        self,
        *,
        q_target_enu_wxyz: NDArray[np.float64],
        thrust_normalized: float,
        q_current_enu_wxyz: NDArray[np.float64],
        omega_current_body: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return 4-vector of motor speeds (rad/s)."""
        # 1. Quaternion error in body frame: q_err = q_target ⊗ q_current⁻¹
        q_err = _quat_multiply(q_target_enu_wxyz, _quat_inverse_unit(q_current_enu_wxyz))

        # 2. Take the shorter rotation path.
        sign = 1.0 if q_err[0] >= 0.0 else -1.0

        # 3. PD controller: ω_desired = k_att * 2 * sign(w) * (xyz components)
        #                             − k_damp * omega_current (rate damping)
        omega_desired = (
            self.k_att * 2.0 * sign * q_err[1:4]
            - self.k_damp * omega_current_body
        )

        # 4. Map normalized thrust [0,1] to physical thrust force in Newtons.
        thrust_n = float(np.clip(thrust_normalized, 0.0, 1.0)) * self._max_thrust_n

        # 5. TRPY mixer takes a packed [thrust_n, ωx, ωy, ωz] 4-vector.
        trpy = np.array([thrust_n, omega_desired[0], omega_desired[1], omega_desired[2]])
        return self._mixer.mix(trpy)

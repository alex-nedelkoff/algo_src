"""PyBulletBackend — implements DroneBackend Protocol with a programmatic
280×280×160 mm chassis sized to VADR-TS-002 §3.6.

No URDF, no visual mesh, no propellers — just rigid-body physics. Motor
forces applied at body-frame positions derived from the trpy_mixer's
allocation matrix; per-motor yaw reaction torque summed and applied as a
single body-frame torque. First-order motor lag (params.tau_motor) emulated
in software since PyBullet has no native motor model.

Coordinate convention: PyBullet world is Z-up; we treat it directly as ENU.
Quaternions are xyzw inside PyBullet, wxyz at the DroneState boundary.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneState, ImuSample


# VADR-TS-002 §3.6 chassis bounding box.
_CHASSIS_W = 0.280
_CHASSIS_L = 0.280
_CHASSIS_H = 0.160
_CHASSIS_HALF_DIAGONAL = np.sqrt((_CHASSIS_W / 2) ** 2 + (_CHASSIS_L / 2) ** 2)  # 0.198 m

_G = 9.81


def _rotate_world_to_body(v_world: NDArray[np.float64], q_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])
    return R.T @ v_world


@dataclass
class PyBulletBackend:
    params: VehicleParams
    gui: bool = False
    physics_hz: float = 120.0      # spec §4.4

    _client: int = field(init=False, default=-1)
    _body_id: int = field(init=False, default=-1)
    _motor_positions_body: NDArray[np.float64] = field(init=False, default=None)
    _omega_actual: NDArray[np.float64] = field(init=False, default=None)
    _last_vel_world: NDArray[np.float64] = field(init=False, default=None)
    _last_omega_body: NDArray[np.float64] = field(init=False, default=None)
    _last_accel_body: NDArray[np.float64] = field(init=False, default=None)
    _t0_ns: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.params.arm_length > _CHASSIS_HALF_DIAGONAL:
            warnings.warn(
                f"params.arm_length={self.params.arm_length:.3f} m exceeds "
                f"chassis half-diagonal {_CHASSIS_HALF_DIAGONAL:.3f} m "
                f"(spec §3.6: 280×280 mm). Motors will sit outside the chassis bbox.",
                stacklevel=2,
            )
        self._client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setGravity(0.0, 0.0, -_G, physicsClientId=self._client)
        p.setTimeStep(1.0 / self.physics_hz, physicsClientId=self._client)

        L = float(self.params.arm_length)
        s = L / np.sqrt(2.0)
        # Order: FR, FL, RL, RR — matches trpy_mixer allocation (matrix is truth).
        self._motor_positions_body = np.array([
            [+s, -s, 0.0],
            [+s, +s, 0.0],
            [-s, +s, 0.0],
            [-s, -s, 0.0],
        ])

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[_CHASSIS_W / 2, _CHASSIS_L / 2, _CHASSIS_H / 2],
            physicsClientId=self._client,
        )
        self._body_id = p.createMultiBody(
            baseMass=float(self.params.mass),
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=-1,
            basePosition=[0.0, 0.0, 1.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client,
        )
        diag = np.diag(self.params.inertia)
        p.changeDynamics(
            self._body_id, -1,
            localInertiaDiagonal=[float(diag[0]), float(diag[1]), float(diag[2])],
            physicsClientId=self._client,
        )

        self._omega_actual = np.zeros(4, dtype=np.float64)
        self._last_vel_world = np.zeros(3, dtype=np.float64)
        self._last_omega_body = np.zeros(3, dtype=np.float64)
        self._last_accel_body = np.array([0.0, 0.0, _G], dtype=np.float64)
        self._t0_ns = time.monotonic_ns()

    def reset(self, initial_state: DroneState) -> None:
        q_wxyz = initial_state.quat_wxyz
        q_xyzw = [float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])]
        p.resetBasePositionAndOrientation(
            self._body_id,
            posObj=[float(x) for x in initial_state.pos_enu],
            ornObj=q_xyzw,
            physicsClientId=self._client,
        )
        p.resetBaseVelocity(
            self._body_id,
            linearVelocity=[float(x) for x in initial_state.vel_enu],
            angularVelocity=[float(x) for x in initial_state.angular_vel_body],
            physicsClientId=self._client,
        )
        self._omega_actual = initial_state.motor_speed.copy()
        self._last_vel_world = initial_state.vel_enu.copy()
        self._last_omega_body = initial_state.angular_vel_body.copy()
        self._last_accel_body = np.array([0.0, 0.0, _G], dtype=np.float64)
        self._t0_ns = time.monotonic_ns()

    def step(self, motor_commands: NDArray[np.float64], dt: float) -> DroneState:
        # Implemented in Task 8.
        raise NotImplementedError("PyBulletBackend.step is implemented in Task 8")

    def get_imu(self) -> ImuSample:
        return ImuSample(
            accel_body=self._last_accel_body.copy(),
            gyro_body=self._last_omega_body.copy(),
            timestamp_us=(time.monotonic_ns() - self._t0_ns) // 1000,
        )

    def close(self) -> None:
        if self._client >= 0:
            p.disconnect(self._client)
            self._client = -1

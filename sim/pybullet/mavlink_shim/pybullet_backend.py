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
        sub_dt = 1.0 / self.physics_hz
        n_substeps = max(1, int(round(dt / sub_dt)))
        sub_dt = dt / n_substeps if n_substeps > 0 else sub_dt

        vel_before_world = np.array(p.getBaseVelocity(self._body_id, physicsClientId=self._client)[0])

        # Yaw torque sign convention from the trpy_mixer allocation matrix.
        # M0=FR-CW, M1=FL-CCW, M2=RL-CW, M3=RR-CCW.
        spin_sign = np.array([+1.0, -1.0, +1.0, -1.0])

        for _ in range(n_substeps):
            # First-order motor-lag integration. Closed-form per sub-step:
            # ω_new = ω_cmd + (ω - ω_cmd) · exp(-sub_dt/τ).
            alpha = float(np.exp(-sub_dt / max(self.params.tau_motor, 1e-6)))
            self._omega_actual = motor_commands + (self._omega_actual - motor_commands) * alpha
            omega_sq = self._omega_actual ** 2

            f_motors = self.params.k_thrust * omega_sq           # (4,) N
            tau_yaw = float(np.sum(self.params.k_torque * omega_sq * spin_sign))  # N·m

            for i in range(4):
                p.applyExternalForce(
                    self._body_id, -1,
                    forceObj=[0.0, 0.0, float(f_motors[i])],
                    posObj=[float(self._motor_positions_body[i, 0]),
                            float(self._motor_positions_body[i, 1]),
                            float(self._motor_positions_body[i, 2])],
                    flags=p.LINK_FRAME, physicsClientId=self._client,
                )
            p.applyExternalTorque(
                self._body_id, -1,
                torqueObj=[0.0, 0.0, tau_yaw],
                flags=p.LINK_FRAME, physicsClientId=self._client,
            )
            p.stepSimulation(physicsClientId=self._client)

        # Read back full state.
        pos, quat_xyzw = p.getBasePositionAndOrientation(self._body_id, physicsClientId=self._client)
        vel_world, ang_world = p.getBaseVelocity(self._body_id, physicsClientId=self._client)
        pos_enu = np.array(pos, dtype=np.float64)
        vel_enu = np.array(vel_world, dtype=np.float64)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        ang_body = _rotate_world_to_body(np.array(ang_world, dtype=np.float64), quat_wxyz)

        # IMU derivation — same pattern as NumpyQuadBackend.
        if dt > 0.0:
            world_accel = (vel_enu - vel_before_world) / dt
        else:
            world_accel = np.zeros(3)
        accel_body_kinematic = _rotate_world_to_body(world_accel, quat_wxyz)
        gravity_body = _rotate_world_to_body(np.array([0.0, 0.0, -_G]), quat_wxyz)
        self._last_accel_body = accel_body_kinematic - gravity_body
        self._last_omega_body = ang_body.copy()
        self._last_vel_world = vel_enu.copy()

        return DroneState(
            pos_enu=pos_enu,
            vel_enu=vel_enu,
            quat_wxyz=quat_wxyz,
            angular_vel_body=ang_body,
            motor_speed=self._omega_actual.copy(),
            timestamp_us=(time.monotonic_ns() - self._t0_ns) // 1000,
        )

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

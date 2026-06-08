"""Vectorized VQ-matched quadrotor dynamics (ACRO rate-loop interface).

This replaces the feed-forward ``TRPYMixer`` + bare-physics rotational path of
``NumpyQuadDynamics`` with the *identified VQ (DCL FlightSim) control interface*, so an
RL policy trained here transfers to the real sim. The VQ runs in ACRO mode: it accepts a
body-rate + collective-thrust command and runs its OWN inner rate loop. We therefore do
NOT integrate torque/inertia/motor-lag — the identified first-order rate loop IS the
rotational dynamics (see COR-96 / COR-127).

Validated against the real VQ sim (sysid/vq_model.json, COR-96): the per-step body-frame
specific force matches the on-board accelerometer to ~0.4 m/s^2 on all axes (the larger
"holdout RMSE" in vq_model.json — body-y 3.95 — was an artifact of differentiating the
ODOMETRY velocity, not a model gap; confirmed by comparing model vs IMU vs d/dt-odometry,
2026-06-07). So the grey-box params below are transfer-credible AS-IS.

Model (per env, per step dt), VQ convention = NED world / FRD body, quat [w,x,y,z]:
    rate loop : a=exp(-dt/tau_i); om_i <- a*om_i + (1-a)*G_i*wcmd_i    (first-order ZOH, per axis)
    weathervane moments (rate-space, ang-accel*dt):
        om_roll += (roll_wv0 + roll_wv1*v_body_x) * v_body_y * dt      (directional, COR-127)
        om_yaw  += yaw_wv                          * v_body_y * dt
    attitude  : q <- q (x) exp(0.5*om*dt)
    thrust    : c = -(f0 + df*thr) ;  f_body = [-Dx*vbx, -Dy*vby, -c]   (c>0 at hover; -c = body-up, FRD -z)
    translate : a_world = R @ f_body + [0,0,g] ; v += a*dt ; p += v*dt

Action per env: [thrust_norm in [0,1], wx_cmd, wy_cmd, wz_cmd]  (rad/s setpoints, VQ ACRO).
State per env (13): [pos(3), vel(3), quat(4 wxyz), omega(3)].  NOTE: no motor-speed state —
the rate loop subsumes motor dynamics.

Frame note: the existing RL envs use ENU/FLU numpy_quad. To embed this model there, convert
the body axes FRD<->FLU (negate body y,z) which flips the pitch/yaw rate-gain signs and the
weathervane signs; the "flipped yaw torque sign" the handoff flagged is exactly this FRD/FLU
mismatch. Keeping this module NED-native preserves the validated signs; do the ENU swap in the
env wrapper, not here.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

GRAVITY = 9.81

POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)


def quat_to_rotmat_batch(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quaternion [w,x,y,z] -> body-to-world rotation matrix, batched (N,4)->(N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    n = q.shape[0]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (yy + zz); R[:, 0, 1] = 2 * (xy - wz); R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz); R[:, 1, 1] = 1 - 2 * (xx + zz); R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy); R[:, 2, 1] = 2 * (yz + wx); R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def quat_multiply_batch(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product of two (N,4) [w,x,y,z] quaternion arrays."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    r = np.empty_like(q1)
    r[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    r[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    r[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    r[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return r


class VQMatchedDynamics:
    """Vectorized VQ-matched ACRO rate-loop + grey-box aero dynamics.

    Args:
        model: parsed sysid/vq_model.json dict.
        dt: integration timestep (s).
        roll_wv: include the directional roll weathervane moment (COR-127). Default True.
        latency_s: optional actuation latency (s) — commands are delayed by round(latency_s/dt)
            steps before reaching the rate loop. 0 disables (closed-loop fidelity is ~insensitive
            to this in the controlled regime; comms 19ms + thrust-lag 40-90ms per vq_model).
    """

    def __init__(self, model: dict, dt: float = 1.0 / 72.0, roll_wv: bool = True,
                 latency_s: float = 0.0, frame: str = "NED") -> None:
        rl = model["rate_loop"]
        self.tau = np.array([rl[n]["tau_ms"] / 1000.0 for n in ("roll", "pitch", "yaw")])
        self.Gax = np.array([rl[n]["gain_G"] for n in ("roll", "pitch", "yaw")])
        d = model["drag_linear_body"]
        self.Dx, self.Dy = d["Dx"], d["Dy"]
        t = model["thrust"]
        self.f0, self.df = t["f0"], t["df_dthr"]
        self.yaw_wv = model["weathervane"]["wv_coeff"]
        # directional roll weathervane (COR-127): roll_wv(vx) = c0 + c1*v_body_x
        self.roll_wv0, self.roll_wv1 = (-0.105, -0.019) if roll_wv else (0.0, 0.0)
        self.dt = dt
        self.delay = int(round(latency_s / dt))
        self._buf: list[NDArray[np.float64]] = []  # actuation-delay FIFO of actions

        # Frame. NED/FRD is the native (validated) convention; "ENU" is the env's z-up/FLU
        # convention, related by B=diag(1,-1,-1) (180 deg about body-x, a PROPER rotation applied
        # to BOTH world and body). Under B: rate gains, yaw-wv and drag are INVARIANT; only gravity-z,
        # the thrust z-force, and the roll-wv signs flip. (This is the "flipped yaw torque sign" the
        # port flagged — handled here, not by hand in the env.) Equivalence to NED is unit-tested.
        if frame not in ("NED", "ENU"):
            raise ValueError(f"frame must be 'NED' or 'ENU', got {frame!r}")
        self.frame = frame
        enu = frame == "ENU"
        self._g_vec = np.array([0.0, 0.0, -GRAVITY if enu else GRAVITY])
        self._thrust_z_sign = 1.0 if enu else -1.0   # f_body_z = sign * c  (c = -(f0+df*thr) > 0 hover)
        self._roll_wv_sign = -1.0 if enu else 1.0

    def reset(self, n_envs: int) -> NDArray[np.float64]:
        """Hover initial state (level, at rest, z=0): (n_envs, 13)."""
        s = np.zeros((n_envs, 13))
        s[:, 6] = 1.0  # quat w
        self._buf = []
        return s

    def step(self, states: NDArray[np.float64], actions: NDArray[np.float64],
             dt: float | None = None) -> NDArray[np.float64]:
        """Advance all envs one step. actions (N,4)=[thr,wx,wy,wz].

        states is (N,13) [pos,vel,quat,omega], or (N,17) with a trailing motor-speed
        slice (GateRaceEnv/NumpyQuadDynamics layout) — that slice is preserved untouched
        since the rate loop subsumes motor dynamics.
        """
        dt = self.dt if dt is None else dt
        if states.shape[1] > 13:
            out = states.copy()
            out[:, :13] = self.step(states[:, :13], actions, dt)
            return out
        # actuation latency buffer
        if self.delay > 0:
            self._buf.append(actions.copy())
            act = self._buf.pop(0) if len(self._buf) > self.delay else np.zeros_like(actions)
        else:
            act = actions
        thr = np.clip(act[:, 0], 0.0, 1.0)
        wcmd = act[:, 1:4]

        pos = states[:, POS]; vel = states[:, VEL]; quat = states[:, QUAT]; om = states[:, OMEGA]
        R = quat_to_rotmat_batch(quat)
        vb = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), vel)  # body velocity = R^T v

        # --- rate loop (first-order per axis, EXACT zero-order-hold discretization) ---
        # om[k+1] = a*om[k] + (1-a)*G*wcmd, a=exp(-dt/tau). Exact for held input and dt-correct;
        # plain Euler is wrong here (dt/tau ~ 0.5 at 72 Hz). Matches vq_model alpha (=exp(-dt_fit/tau)).
        a = np.exp(-dt / self.tau)
        om_new = a * om + (1.0 - a) * self.Gax * wcmd
        # --- weathervane moments (ang-accel * dt) ---
        om_new[:, 0] += self._roll_wv_sign * (self.roll_wv0 + self.roll_wv1 * vb[:, 0]) * vb[:, 1] * dt
        om_new[:, 2] += self.yaw_wv * vb[:, 1] * dt   # yaw-wv invariant under the NED<->ENU basis change

        # --- attitude integrate via axis-angle exp(0.5 om dt) ---
        ang = np.linalg.norm(om_new, axis=1) * dt
        dq = np.zeros((states.shape[0], 4)); dq[:, 0] = 1.0
        m = ang > 1e-9
        if np.any(m):
            axis = om_new[m] / np.linalg.norm(om_new[m], axis=1, keepdims=True)
            half = ang[m] / 2.0
            dq[m, 0] = np.cos(half)
            dq[m, 1:4] = np.sin(half)[:, None] * axis
        new_quat = quat_multiply_batch(quat, dq)
        new_quat /= np.maximum(np.linalg.norm(new_quat, axis=1, keepdims=True), 1e-12)
        R = quat_to_rotmat_batch(new_quat)

        # --- translational: thrust + body drag + gravity (frame-aware signs) ---
        # c = -(f0+df*thr) > 0 at hover. Thrust along body-up: f_body_z = -c (NED/FRD) or +c (ENU/FLU).
        c = -(self.f0 + self.df * thr)
        f_body = np.stack([-self.Dx * vb[:, 0], -self.Dy * vb[:, 1], self._thrust_z_sign * c], axis=1)
        a_world = np.einsum("nij,nj->ni", R, f_body) + self._g_vec

        new = np.empty_like(states)
        new[:, VEL] = vel + dt * a_world
        new[:, POS] = pos + dt * new[:, VEL]
        new[:, QUAT] = new_quat
        new[:, OMEGA] = om_new
        return new

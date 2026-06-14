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
    weathervane moments (rate-space, ang-accel*dt), v2 when model["weathervane_v2"] present:
        om_ax  += (a_ax + b_ax*min(|v|,vclip)) * v_body_y * dt         (ax = roll, yaw; 06-10 refit)
    v1 fallback (older vq_model.json):
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
                 latency_s: float = 0.0, frame: str = "NED", thrust_lag_s: float = 0.0) -> None:
        rl = model["rate_loop"]
        self.tau = np.array([rl[n]["tau_ms"] / 1000.0 for n in ("roll", "pitch", "yaw")])
        self.Gax = np.array([rl[n]["gain_G"] for n in ("roll", "pitch", "yaw")])
        # MIMO rate loop (COR-127): om[k+1]=A@om+B@wcmd captures the roll<->yaw cross-coupling a
        # coordinated turn excites, which the diagonal first-order loop MISSES (turn-region 1-step
        # omega RMSE 4-10x lower, eig(A)<1; fit at dt=1/72 on collect_vq doublets + corner runs,
        # weathervane-subtracted). Opt-in via model["rate_loop_mimo"]; falls back to the diagonal
        # loop when absent. NOTE: A,B are dt-specific (fit dt=1/72); the env steps at 1/72.
        _mimo = model.get("rate_loop_mimo")
        self.A_rate = np.array(_mimo["A"], float) if _mimo else None
        self.B_rate = np.array(_mimo["B"], float) if _mimo else None
        if self.A_rate is not None and abs(dt - 1.0 / 72.0) > 1e-9:
            raise ValueError(
                f"rate_loop_mimo A,B are discrete-time at dt=1/72 s; got dt={dt}. "
                "Run the env at dt=1/72 (or refit A,B for this dt).")
        # 2nd-order MIMO rate loop (COR-127 RING-01): the live inner loop is UNDERDAMPED — a
        # complex pole pair at ~4-7 Hz per axis rings under step-like commands. First-order
        # structures cannot overshoot, so the ring showed up as "white" 1-step residual and a
        # policy-killing live limit cycle (the policy feeds back near the ring frequency).
        # om[k+1] = A1@om[k] + A2@om[k-1] + B0@w[k] + B1@w[k-1]; needs per-env history state,
        # kept on the object (reset() clears it; episode-level partial resets leave a 1-step
        # transient, harmless at om~0 hover starts). Opt-in via model["rate_loop_2nd"];
        # supersedes rate_loop_mimo when present. dt-specific (1/72).
        _r2 = model.get("rate_loop_2nd")
        if _r2:
            self.A1_r2 = np.array(_r2["A1"], float); self.A2_r2 = np.array(_r2["A2"], float)
            self.B0_r2 = np.array(_r2["B0"], float); self.B1_r2 = np.array(_r2["B1"], float)
            if abs(dt - 1.0 / 72.0) > 1e-9:
                raise ValueError(f"rate_loop_2nd matrices are discrete-time at dt=1/72 s; got dt={dt}.")
        else:
            self.A1_r2 = None
        self._om_prev: NDArray[np.float64] | None = None
        self._wcmd_prev: NDArray[np.float64] | None = None
        d = model["drag_linear_body"]
        self.Dx, self.Dy = d["Dx"], d["Dy"]
        # quadratic body drag (COR-127 REPLAY-01): the linear fit has no authority at high v —
        # under DEPLOY-03's recorded commands the live VQ caps at |v|=36 m/s where this plant
        # reached 92.7. f_ax += -q_ax*|v|*vb_ax (+ optional linear Dz; the linear fit had NO
        # z-drag at all). Opt-in via model["drag_quadratic_body"]; absent -> exact old behavior.
        _dq = model.get("drag_quadratic_body") or {}
        self.qx = float(_dq.get("qx", 0.0))
        self.qy = float(_dq.get("qy", 0.0))
        self.qz = float(_dq.get("qz", 0.0))
        self.Dz = float(_dq.get("Dz", 0.0))
        t = model["thrust"]
        self.f0, self.df = t["f0"], t["df_dthr"]
        # thrust floor (COR-127 REPLAY-01): the linear map extrapolates to NEGATIVE collective
        # below thr=-f0/df (props pushing down — nonphysical); live braking runs hold |v|~5 at
        # thr_cmd=0 where this plant free-falls. thr_eff = max(thr, thr_floor). Opt-in via
        # model["thrust"]["thr_floor"]; absent -> exact old behavior.
        self.thr_floor = float(t.get("thr_floor", -1.0))
        self.yaw_wv = model["weathervane"]["wv_coeff"]
        # directional roll weathervane (COR-127): roll_wv(vx) = c0 + c1*v_body_x
        self.roll_wv0, self.roll_wv1 = (-0.105, -0.019) if roll_wv else (0.0, 0.0)
        # speed-dependent weathervane v2 (COR-127, 06-10 brake-data joint refit): per axis
        # coeff(|v|) = a + b*min(|v|, vclip);  om_ax += coeff*v_body_y*dt  (roll & yaw). One curve
        # unifies the directional-roll fit (its vbx term was this decay in disguise), the
        # WV-DYNAMIC high-speed sign flip, and the const yaw wv (mid-curve value). Dedicated
        # controlled-brake data REFUTED the hypothesized vbx~0 "braking-wv" blowup — the braking
        # snap gap is the tilt-scaled disturbance field (model["dr_rate_disturbance"]), not a
        # deterministic moment. Opt-in via model["weathervane_v2"]; v1 fallback when absent.
        _wv2 = model.get("weathervane_v2")
        if _wv2:
            self.wv2 = (_wv2["roll"]["a"] if roll_wv else 0.0,
                        _wv2["roll"]["b"] if roll_wv else 0.0,
                        _wv2["yaw"]["a"], _wv2["yaw"]["b"], _wv2["vclip"])
        else:
            self.wv2 = None
        # measured unmodeled-moment field (COR-127 BRAKE-WV-01): sustained per-axis rate biases,
        # RMS scaled by TRUE tilt, ~block_s correlation. This is the field the live plant has and
        # the nominal sim lacks (the DEPLOY-02 snap mechanism). Injected as piecewise-constant
        # per-env bias resampled every block_s, sigma interpolated from the tilt-binned table.
        # ACTIVATION: model["dr_rate_disturbance"]["scale"] > 0 (absent in the canonical file ->
        # off everywhere by default; rl_finetune's per-round DR copies set it).
        _dd = model.get("dr_rate_disturbance") or {}
        self._dist_scale = float(_dd.get("scale", 0.0))
        self._dd_raw = _dd                 # kept so randomize() can inject the field on a scale=0 model
        self.dr_factors = None             # (n_envs, 11) privileged vector; set by randomize()
        if self._dist_scale > 0.0:
            bins = _dd["bins"]
            self._dist_lo = np.array([b["tilt_deg"][0] for b in bins], float)
            self._dist_sig = np.array([b["sigma"] for b in bins], float)   # (nbins, 3) rad/s^2
            self._dist_steps = max(1, int(round(float(_dd.get("block_s", 0.5)) / dt)))
        self._dist_bias: NDArray[np.float64] | None = None
        self._dist_k = 0
        self.dt = dt
        # latency_s = pure comms delay (whole command, FIFO); thrust_lag_s = first-order motor/thrust
        # lag (measured tau~85ms, MOTOR-01) -- a pure delay is the wrong structure for it. The thrust
        # lag state rides the 17-state motor slot [:,13] (env path); inactive on 13-state / when 0.
        self.delay = int(round(latency_s / dt))
        self.thrust_lag_s = thrust_lag_s
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
        if enu and self.A_rate is not None:          # conjugate MIMO matrices by B=diag(1,-1,-1): roll<->{pitch,yaw} cross terms flip, pitch<->yaw invariant
            T = np.diag([1.0, -1.0, -1.0])
            self.A_rate = T @ self.A_rate @ T
            self.B_rate = T @ self.B_rate @ T
        if enu and self.A1_r2 is not None:           # same conjugation for the 2nd-order loop
            T = np.diag([1.0, -1.0, -1.0])
            self.A1_r2 = T @ self.A1_r2 @ T; self.A2_r2 = T @ self.A2_r2 @ T
            self.B0_r2 = T @ self.B0_r2 @ T; self.B1_r2 = T @ self.B1_r2 @ T

    # ---- per-env domain randomization (asymmetric-critic privileged channel) ----
    # Scalar aero/thrust/wv params become per-env arrays (broadcast cleanly in step());
    # the rate-loop matrices stay nominal (dt-specific + validated; per-round DR scatters
    # them for the actor). dr_factors (n_envs, 11) is the privileged vector handed to the
    # critic, centered at 0: factor-1 for the multiplicative params, raw for dist_scale.
    _DR_KEYS = ("Dx", "Dy", "qx", "qy", "qz", "Dz", "f0", "df")

    def randomize(self, rng, n_envs: int, width: float = 0.4) -> None:
        nom = {k: float(getattr(self, k)) for k in self._DR_KEYS}
        fac = {k: 1.0 + rng.uniform(-width, width, n_envs) for k in self._DR_KEYS}
        for k in self._DR_KEYS:
            setattr(self, k, nom[k] * fac[k])
        # weathervane v2 speed-slope (roll b, yaw b) -- the live-divergent term (WV-DYNAMIC)
        if self.wv2 is not None:
            a_r, b_r, a_y, b_y, vclip = self.wv2
            fr = 1.0 + rng.uniform(-width, width, n_envs)
            fy = 1.0 + rng.uniform(-width, width, n_envs)
            self.wv2 = (a_r, b_r * fr, a_y, b_y * fy, vclip)
            wv_rb, wv_yb = fr - 1.0, fy - 1.0
        else:
            wv_rb = wv_yb = np.zeros(n_envs)
        # measured disturbance field magnitude (the DEPLOY-02 snap mechanism); nominal 0..2
        dist = rng.uniform(0.0, 2.0, n_envs)
        self._dist_scale_arr = dist
        if self._dd_raw and not hasattr(self, "_dist_sig"):   # build tables on a scale=0 model
            bins = self._dd_raw["bins"]
            self._dist_lo = np.array([b["tilt_deg"][0] for b in bins], float)
            self._dist_sig = np.array([b["sigma"] for b in bins], float)
            self._dist_steps = max(1, int(round(float(self._dd_raw.get("block_s", 0.5)) / self.dt)))
        self.dr_factors = np.column_stack(
            [fac[k] - 1.0 for k in self._DR_KEYS] + [wv_rb, wv_yb, dist]
        ).astype(np.float32)

    def reset(self, n_envs: int) -> NDArray[np.float64]:
        """Hover initial state (level, at rest, z=0): (n_envs, 13)."""
        s = np.zeros((n_envs, 13))
        s[:, 6] = 1.0  # quat w
        self._buf = []
        self._dist_bias = None
        self._dist_k = 0
        self._om_prev = None
        self._wcmd_prev = None
        return s

    def step(self, states: NDArray[np.float64], actions: NDArray[np.float64],
             dt: float | None = None) -> NDArray[np.float64]:
        """Advance all envs one step. actions (N,4)=[thr,wx,wy,wz].

        states is (N,13) [pos,vel,quat,omega], or (N,17) with a trailing motor-speed
        slice (GateRaceEnv/NumpyQuadDynamics layout) — that slice is preserved untouched
        since the rate loop subsumes motor dynamics.
        """
        dt = self.dt if dt is None else dt
        wide = states.shape[1] > 13
        # comms delay: whole command through a FIFO of `delay` steps
        if self.delay > 0:
            self._buf.append(actions.copy())
            act = self._buf.pop(0) if len(self._buf) > self.delay else np.zeros_like(actions)
        else:
            act = actions
        thr_cmd = np.clip(act[:, 0], 0.0, 1.0)
        wcmd = act[:, 1:4]
        # first-order thrust/motor lag (state in motor slot [:,13]); inactive on 13-state or when 0
        if self.thrust_lag_s > 0.0 and wide:
            a_thr = np.exp(-dt / self.thrust_lag_s)
            thr_prev = np.clip(states[:, 13], 0.0, 1.0)
            thr = a_thr * thr_prev + (1.0 - a_thr) * thr_cmd
        else:
            thr = thr_cmd

        pos = states[:, POS]; vel = states[:, VEL]; quat = states[:, QUAT]; om = states[:, OMEGA]
        R = quat_to_rotmat_batch(quat)
        vb = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), vel)  # body velocity = R^T v

        # --- rate loop (first-order per axis, EXACT zero-order-hold discretization) ---
        # om[k+1] = a*om[k] + (1-a)*G*wcmd, a=exp(-dt/tau). Exact for held input and dt-correct;
        # plain Euler is wrong here (dt/tau ~ 0.5 at 72 Hz). Matches vq_model alpha (=exp(-dt_fit/tau)).
        if self.A1_r2 is not None:                   # 2nd-order underdamped loop (RING-01; dt~=1/72)
            n_env = om.shape[0]
            if self._om_prev is None or self._om_prev.shape[0] != n_env:
                self._om_prev = om.copy(); self._wcmd_prev = wcmd.copy()
            om_new = (om @ self.A1_r2.T + self._om_prev @ self.A2_r2.T
                      + wcmd @ self.B0_r2.T + self._wcmd_prev @ self.B1_r2.T)
            self._om_prev = om.copy(); self._wcmd_prev = wcmd.copy()
        elif self.A_rate is not None:                # MIMO rate loop (roll<->yaw turn coupling, COR-127; dt~=1/72)
            om_new = om @ self.A_rate.T + wcmd @ self.B_rate.T
        else:
            a = np.exp(-dt / self.tau)
            om_new = a * om + (1.0 - a) * self.Gax * wcmd
        # --- weathervane moments (ang-accel * dt) ---
        if self.wv2 is not None:                     # v2: speed-dependent coeff (|v| basis-invariant)
            a_r, b_r, a_y, b_y, vclip = self.wv2
            spd = np.minimum(np.linalg.norm(vb, axis=1), vclip)
            om_new[:, 0] += self._roll_wv_sign * (a_r + b_r * spd) * vb[:, 1] * dt
            om_new[:, 2] += (a_y + b_y * spd) * vb[:, 1] * dt   # vby & om_yaw both flip under ENU -> invariant
        else:                                        # v1 fallback (const yaw + linear-in-vbx roll)
            om_new[:, 0] += self._roll_wv_sign * (self.roll_wv0 + self.roll_wv1 * vb[:, 0]) * vb[:, 1] * dt
            om_new[:, 2] += self.yaw_wv * vb[:, 1] * dt   # yaw-wv invariant under the NED<->ENU basis change
        # --- measured disturbance field (DR): tilt-scaled correlated rate bias ---
        # per-env scale (randomize()) overrides the scalar; nominal model -> scalar path unchanged
        dist_scale = getattr(self, "_dist_scale_arr", None)
        dist_active = dist_scale if dist_scale is not None else self._dist_scale
        if np.any(dist_active > 0.0):
            n = states.shape[0]
            if self._dist_bias is None or self._dist_bias.shape[0] != n:
                self._dist_bias = np.zeros((n, 3)); self._dist_k = 0
            if self._dist_k % self._dist_steps == 0:
                tilt = np.degrees(np.arccos(np.clip(np.abs(R[:, 2, 2]), -1.0, 1.0)))
                idx = np.searchsorted(self._dist_lo, tilt, side="right") - 1
                sig = self._dist_sig[np.clip(idx, 0, len(self._dist_sig) - 1)]
                scl = np.asarray(dist_active)[:, None] if np.ndim(dist_active) else dist_active
                self._dist_bias = np.random.normal(0.0, 1.0, (n, 3)) * sig * scl
            self._dist_k += 1
            om_new += self._dist_bias * dt

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
        c = -(self.f0 + self.df * np.maximum(thr, self.thr_floor))
        vmag = np.linalg.norm(vb, axis=1)
        f_body = np.stack([-(self.Dx + self.qx * vmag) * vb[:, 0],
                           -(self.Dy + self.qy * vmag) * vb[:, 1],
                           -(self.Dz + self.qz * vmag) * vb[:, 2] + self._thrust_z_sign * c], axis=1)
        a_world = np.einsum("nij,nj->ni", R, f_body) + self._g_vec

        new = states.copy()  # preserves any trailing slots (17-state motor slice)
        new[:, VEL] = vel + dt * a_world
        new[:, POS] = pos + dt * new[:, VEL]
        new[:, QUAT] = new_quat
        new[:, OMEGA] = om_new
        if self.thrust_lag_s > 0.0 and wide:
            new[:, 13] = thr  # carry the lagged thrust state
        return new

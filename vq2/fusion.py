"""Deterministic offline IMU + floor-flow (+ optional vision-position)
fusion through PosVelKF, replaying a recorded corpus.

Attitude chain is the flight-proven vq2wp.py one:
  roll += gx dt ; pitch += -gy dt ; yaw += -gz dt   (wfix)
seeded accel-implied at the end of the first rest window. NOT the
estimators.py WFIX (its yaw sign diverges from the flight code).

Vision anchors: GateNet t_cam (camera frame) -> body -> level -> world;
drone position implied by a map gate = gate_w - g_w. Obs Z is biased
(quarantined) so anchors are compared XY-ONLY, but the full 3-vector is
kept for diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import corpus as corpus_mod
from .camera import M_BODY_CAM, R_world_body
from .eskf import PosVelKF, accel_level
from .estimators import accel_implied_attitude, is_at_rest
from .flow_vel import FlowVelocity, Z_FLOOR
from .replay import find_rest_windows

# course map (spawn frame, z down) — HANDOFF_vq2_racing.md course facts
GATES = (
    (11.0, 0.0, -1.3),   # gate 1 (judge gate)
    (10.4, 0.0, -4.0),   # stacked high gate above gate 1
    (30.5, 8.5, -1.5),   # gate 2
)
ACCEPT_R = 3.0  # xy radius for matching an obs to a map gate (as in vq2wp)

import os as _os
_G1C = _os.path.join(_os.path.dirname(__file__), "g1_corners_world.npy")
G1_CORNERS_W = np.load(_G1C) if _os.path.exists(_G1C) else None


@dataclass
class FusionConfig:
    use_flow: bool = True
    use_vision_pos: bool = False
    z_floor: float = Z_FLOOR
    gate_map: tuple = GATES
    # drag-velocity aiding (playbook 0.4): in free flight thrust is body-z
    # only, so body-xy specific force = -D * v_body -- the accelerometer is
    # a scale-free velocity sensor. Dx=Dy=0.1 from sysID. Dz~0: no z info.
    use_drag: bool = False   # holdout verdict: hurts at creep speed (SNR~1
    #                          at 1 m/s); enable for the race-pace speed push
    drag_d: float = 0.1
    drag_sigma: float = 1.0        # m/s per sample; 144 Hz aggregates hard
    drag_vmax: float = 15.0        # implied |v| above this = not drag physics
    # blind-leg holdout: (t_lo, t_hi) on the IMU boot axis. Inside the window
    # vision POSITION/VELOCITY fixes are withheld (anchors still recorded for
    # scoring) -- measures dead-reckoning drift against vision truth.
    deny_vision: tuple | None = None
    # quiescent-gated complementary attitude correction (see AttitudeTracker)
    att_cf_gain: float = 0.0
    # obs acceptance policy: 'radius' = legacy binary ACCEPT_R gate;
    # 'huber' = never discard a matched obs, inflate R with miss instead
    # (ADR-VINS/MonoRace pattern -- keeps the filter correctable at any drift)
    accept_policy: str = "radius"
    huber_delta: float = 1.5   # m: miss below this gets full weight
    # range-ratio identity gate before candidate selection (see
    # range_identity_ok). 'huber' + range_gate=True == the ADR-VINS pair.
    range_gate: bool = False
    # flare veto (EXPERIMENTAL, REFUTED on vq2_accept5: in-flight junk obs
    # occur at LEVEL pitch p50 -0.2deg -- truss/fixture solves in the upper
    # frame at normal attitude, not flare-only; image row overlaps too,
    # junk p50 234 vs good 249. Content-level identity -- corner descriptors
    # / tight corner updates, playbook 0.3 -- is the real fix. Kept as an
    # off-by-default bench knob.) Nose-up beyond this (rad) vetoes obs.
    pitch_veto: float = 0.0
    # constellation identity veto (corner map): an obs matched to G1 must
    # have >=2 high-visibility detected corners landing within ident_px of
    # the projected corner map, else it cannot be G1 (truss/fixture solves
    # fail this regardless of range/miss). Pose fixes stay the update;
    # corners are the ID check (07-06 decision: detector corner noise
    # ~24 px ~= pose-fix accuracy, so identity is the win, not accuracy).
    corner_ident: bool = False
    ident_px: float = 60.0
    ident_min_corners: int = 2
    ident_min_vis: float = 0.5
    # strict mode: pre-G1-tick course context -- any obs that is NOT
    # positively identified as G1 (and is not a genuinely far G2 sighting,
    # rng > ident_far_ok) is vetoed. G2 has no corner map yet; near-range
    # "G2 matches" while flying the G1 leg are hallucination escapes.
    ident_strict: bool = False
    ident_far_ok: float = 18.0
    ident_full_weight: bool = True


@dataclass
class FusionResult:
    t_s: list = field(default_factory=list)
    p: list = field(default_factory=list)
    v: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    flow_updates: int = 0
    flow_rejects: int = 0
    a_w_xy: list = field(default_factory=list)  # gravity-leak diagnostic
    # obs waterfall: one event per detection, stage in
    # {policy_reject, maha_reject, accepted} (+ nis/innovation when updated)
    obs_events: list = field(default_factory=list)


GRAVITY = 9.81


@dataclass
class AttitudeTracker:
    """wfix gyro attitude (vq2wp convention) + optional quiescent-gated
    complementary correction.

    Why: HIGHRES_IMU drops ~24% of samples; integrating across 14 ms gaps
    during violent transients (route-entry corrections hit 350 deg/s) accrues
    1-3 deg of PERMANENT tilt error -> ~0.2-0.5 m/s^2 gravity leak -> the
    estimate runaway seen in every collapsed flight. The CF pulls roll/pitch
    back toward the accel-implied attitude ONLY when quasi-static (low rates,
    |f| ~ g) -- the regime where accel direction IS gravity. Yaw untouched
    (accel carries no yaw)."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    cf_gain: float = 0.0            # per-sample pull toward accel attitude
    cf_gyro_max: float = 0.3        # rad/s: "quiet" rate gate
    cf_acc_tol: float = 0.6         # m/s^2: | |f| - g | gate
    _last_us: int | None = None

    def seed(self, s) -> None:
        self.roll, self.pitch = accel_implied_attitude(s)
        self._last_us = s.t_us

    def update(self, s) -> None:
        if self._last_us is not None and s.t_us > self._last_us:
            dt = (s.t_us - self._last_us) / 1e6
            gx, gy, gz = s.gyr
            self.roll += gx * dt
            self.pitch += -gy * dt   # wfix (vq2wp.py:203)
            self.yaw += -gz * dt     # wfix (vq2wp.py:204)
            if self.cf_gain > 0.0:
                gn = math.sqrt(gx * gx + gy * gy + gz * gz)
                an = math.sqrt(sum(a * a for a in s.acc))
                if gn < self.cf_gyro_max and abs(an - GRAVITY) < self.cf_acc_tol:
                    r_a, p_a = accel_implied_attitude(s)
                    self.roll += self.cf_gain * (r_a - self.roll)
                    self.pitch += self.cf_gain * (p_a - self.pitch)
        self._last_us = s.t_us


def radius_policy(miss: float, rng: float):
    """Legacy binary gate: accept iff within ACCEPT_R of the estimate.
    THE documented failure mode: >3 m drift = uncorrectable filter."""
    return miss <= ACCEPT_R, 1.0


def huber_policy(miss: float, rng: float, delta: float = 1.5):
    """Never discard a matched obs; inflate measurement sigma linearly with
    miss beyond delta. Far-off vision becomes a soft pull instead of noise --
    the filter can always be walked back (ADR-VINS/MonoRace)."""
    if miss <= delta:
        return True, 1.0
    return True, miss / delta


POLICIES = {"radius": radius_policy, "huber": huber_policy}

RANGE_RATIO_MIN = 0.55   # min(rng_m/rng_e, rng_e/rng_m) below this = not that gate


def range_identity_ok(rng_meas: float, rng_expected: float) -> bool:
    """Area-consistency association, in range form (ADR-VINS rho_m): GateNet
    PnP range is size-derived and verified TRUE, so the measured range IS the
    apparent-size cue. A candidate whose expected range disagrees by more
    than ~2x in ratio is a different gate (or a hallucination), regardless
    of how the xy proximity looks. Estimate-LIGHT: G1 vs G2 expected ranges
    differ by ~20 m, so this survives ~10 m of estimator drift where the
    3 m xy proximity gate has long since gone blind."""
    if rng_expected <= 0.1 or rng_meas <= 0.1:
        return False
    ratio = min(rng_meas / rng_expected, rng_expected / rng_meas)
    return ratio >= RANGE_RATIO_MIN


def drag_velocity_update(kf, s, R_wb, cfg) -> bool:
    """One drag-aiding velocity update from an IMU sample. Returns True if
    applied. Gated OFF at rest: on the ground, contact forces put large
    non-drag components in body-xy (the 18-deg pad reads as a false
    ~30 m/s), and the model only holds in free flight."""
    from .estimators import is_at_rest
    if is_at_rest(s):
        return False
    v_bx = -s.acc[0] / cfg.drag_d
    v_by = -s.acc[1] / cfg.drag_d
    if abs(v_bx) > cfg.drag_vmax or abs(v_by) > cfg.drag_vmax:
        return False
    # body-z velocity is unobservable through drag (Dz~0): carry the current
    # estimate so that axis has zero innovation (vq2wp vision-velocity pattern)
    v_b_est = R_wb.T @ kf.v
    v_b_meas = np.array([v_bx, v_by, v_b_est[2]])
    kf.update_velocity(R_wb @ v_b_meas, cfg.drag_sigma)
    return True


def _detection_events(c, bridge_off):
    """[(t_boot_s, g_cam (3,), corners (8,2)|None, vis (8,)|None) ...]."""
    ev = []
    for d in c.detections:
        for inst in d.insts:
            if not inst.get("solved") or inst.get("low_confidence"):
                continue
            t_cam = inst.get("t_cam")
            if not t_cam:
                continue
            cx = inst.get("corner_xy")
            vis = inst.get("visibility")
            ev.append((d.sim_ns / 1e9 + bridge_off, np.asarray(t_cam, float),
                       np.asarray(cx, float) if cx else None,
                       np.asarray(vis, float) if vis else None))
    ev.sort(key=lambda e: e[0])
    return ev


def run_fusion(root: str, cfg: FusionConfig) -> FusionResult:
    c = corpus_mod.load(root)
    seg = c.flight_segment
    res = FusionResult()
    if seg is None or not seg.imu:
        return res
    rests = find_rest_windows(seg.imu)
    if not rests:
        return res
    bridge = corpus_mod.clock_bridge(seg, c.frames)
    off = bridge[0] if bridge else None

    # the recorder's frames/ dir and detections.jsonl accumulate across
    # recording sessions (see corpus.py loader comments) -- bound both event
    # streams to the current flight segment's time span, or stale events
    # mass-drain on the first IMU tick and corrupt the filter
    t_lo = seg.imu[0].t_us / 1e6 - 1.0
    t_hi = seg.imu[-1].t_us / 1e6 + 1.0
    frames = sorted(
        (
            (fr.sim_ns / 1e9 + off, fr.path)
            for fr in c.frames
            if t_lo <= fr.sim_ns / 1e9 + off <= t_hi
        ),
        key=lambda e: e[0],
    ) if off is not None else []
    dets = [
        e for e in _detection_events(c, off) if t_lo <= e[0] <= t_hi
    ] if off is not None else []

    # seed at the end of the first rest window -- unless that window runs to
    # the end of the recording (corpus is at-rest throughout, e.g. vq2_rec),
    # in which case seeding at its end leaves nothing left to integrate.
    seed_idx = rests[0][1] if rests[0][1] < len(seg.imu) - 1 else rests[0][0]
    s0 = seg.imu[seed_idx]
    att = AttitudeTracker(cf_gain=cfg.att_cf_gain)
    att.seed(s0)
    kf = PosVelKF()
    kf.reset_at_rest()
    flow = FlowVelocity()
    fi = di = 0
    last_us = s0.t_us

    for s in seg.imu[seed_idx + 1:]:
        if s.t_us <= last_us:
            continue
        dt = (s.t_us - last_us) / 1e6
        last_us = s.t_us
        t_boot = s.t_us / 1e6
        att.update(s)
        roll, pitch, yaw = att.roll, att.pitch, att.yaw
        a_lvl = accel_level(s.acc, roll, pitch)
        cyw, syw = math.cos(yaw), math.sin(yaw)
        a_w = np.array([cyw * a_lvl[0] - syw * a_lvl[1],
                        syw * a_lvl[0] + cyw * a_lvl[1], a_lvl[2]])
        kf.predict(a_w, dt)
        res.a_w_xy.append(a_w[:2].copy())

        if cfg.use_drag:
            drag_velocity_update(kf, s, R_world_body(roll, pitch, yaw), cfg)

        # frame events up to this IMU stamp
        while cfg.use_flow and fi < len(frames) and frames[fi][0] <= t_boot:
            t_f, path = frames[fi]
            fi += 1
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            h = cfg.z_floor - float(kf.p[2])
            out = flow.process(gray, t_f, (roll, pitch, yaw), h)
            if out is None:
                res.flow_rejects += 1
                continue
            v_w, sigma, ninl, ntr = out
            kf.update_velocity(v_w, sigma)
            res.flow_updates += 1

        # detection events -> anchors (and optional position updates)
        while di < len(dets) and dets[di][0] <= t_boot:
            t_d, g_cam, d_corners, d_vis = dets[di]
            di += 1
            # at-rest exemption: the 18-deg spawn tilt would otherwise veto
            # the pad-lock obs -- the best anchors we get. Flare veto is an
            # airborne concept.
            if (cfg.pitch_veto > 0.0 and pitch < -cfg.pitch_veto
                    and not is_at_rest(s)):
                res.obs_events.append({
                    "t_boot_s": t_d, "rng": float(np.linalg.norm(g_cam)),
                    "miss": float("nan"), "gate": -1, "r_scale": 0.0,
                    "stage": "flare_veto"})
                continue
            g_w = R_world_body(roll, pitch, yaw) @ (M_BODY_CAM @ g_cam)
            rng = float(np.linalg.norm(g_cam))
            p_vis_cands = [np.asarray(g) - g_w for g in cfg.gate_map]
            errs = [np.linalg.norm((pv - kf.p)[:2]) for pv in p_vis_cands]
            if cfg.range_gate:
                feasible = [
                    k for k, g in enumerate(cfg.gate_map)
                    if range_identity_ok(
                        rng, float(np.linalg.norm(
                            np.asarray(g)[:2] - kf.p[:2])))
                ]
                if not feasible:
                    res.obs_events.append({
                        "t_boot_s": t_d, "rng": rng,
                        "miss": float(min(errs)), "gate": -1,
                        "r_scale": 0.0, "stage": "identity_reject"})
                    continue
            else:
                feasible = list(range(len(cfg.gate_map)))
            j = min(feasible, key=lambda k: errs[k])
            miss = float(errs[j])
            if cfg.accept_policy == "huber":
                ok_pol, r_scale = huber_policy(miss, rng, cfg.huber_delta)
            else:
                ok_pol, r_scale = radius_policy(miss, rng)
            ev = {"t_boot_s": t_d, "rng": rng, "miss": miss, "gate": j,
                  "r_scale": r_scale, "stage": "policy_reject"}
            if not ok_pol:
                res.obs_events.append(ev)
                continue
            if (cfg.corner_ident and G1_CORNERS_W is not None
                    and d_corners is not None and d_vis is not None):
                R_wc = R_world_body(roll, pitch, yaw) @ M_BODY_CAM
                rel = (G1_CORNERS_W - kf.p) @ R_wc
                zok = rel[:, 2] > 0.2
                n_match = 0
                for k in range(min(len(d_corners), len(G1_CORNERS_W))):
                    if d_vis[k] < cfg.ident_min_vis or not zok[k]:
                        continue
                    uv = np.array([rel[k, 0] / rel[k, 2] * 226.0 + 319.5,
                                   rel[k, 1] / rel[k, 2] * 226.0 + 179.5])
                    if np.linalg.norm(uv - d_corners[k]) < cfg.ident_px:
                        n_match += 1
                is_g1 = n_match >= cfg.ident_min_corners
                if is_g1 and cfg.ident_full_weight:
                    # content-confirmed obs: huber's miss-based distrust is
                    # estimate-referenced and soft-pedals exactly the fixes
                    # that rescue a drifted filter (run-6/8 approach misses).
                    # Identity replaces it: full weight.
                    r_scale = 1.0
                    ev["r_scale"] = 1.0
                if j == 0 and not is_g1:
                    ev["stage"] = "corner_ident_reject"
                    res.obs_events.append(ev)
                    continue
                if (cfg.ident_strict and not is_g1
                        and not (j == 2 and rng > cfg.ident_far_ok)):
                    ev["stage"] = "corner_ident_reject"
                    res.obs_events.append(ev)
                    continue
            p_vis = p_vis_cands[j]
            res.anchors.append({
                "t_boot_s": t_d,
                "p_vision": p_vis.copy(),
                "p_est": kf.p.copy(),
                "err_xy": miss,
                "range": rng,
            })
            denied = (cfg.deny_vision is not None
                      and cfg.deny_vision[0] <= t_d <= cfg.deny_vision[1])
            if cfg.use_vision_pos and not denied:
                ok = kf.update_position(
                    np.array([p_vis[0], p_vis[1], kf.p[2]]), rng,
                    r_scale=r_scale,
                )
                ev["stage"] = "accepted" if ok else "maha_reject"
                if getattr(kf, "last_nis", None) is not None:
                    ev["nis"] = float(kf.last_nis)
                    ev["innovation_xy"] = [float(kf.last_innovation[0]),
                                           float(kf.last_innovation[1])]
            else:
                ev["stage"] = "accepted"  # scored as anchor; no update path
                ev["nis"] = float("nan")
            res.obs_events.append(ev)

        res.t_s.append(t_boot)
        res.p.append(kf.p.copy())
        res.v.append(kf.v.copy())
    return res

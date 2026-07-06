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
from .estimators import accel_implied_attitude
from .flow_vel import FlowVelocity, Z_FLOOR
from .replay import find_rest_windows

# course map (spawn frame, z down) — HANDOFF_vq2_racing.md course facts
GATES = (
    (11.0, 0.0, -1.3),   # gate 1 (judge gate)
    (10.4, 0.0, -4.0),   # stacked high gate above gate 1
    (30.5, 8.5, -1.5),   # gate 2
)
ACCEPT_R = 3.0  # xy radius for matching an obs to a map gate (as in vq2wp)


@dataclass
class FusionConfig:
    use_flow: bool = True
    use_vision_pos: bool = False
    z_floor: float = Z_FLOOR
    gate_map: tuple = GATES
    # drag-velocity aiding (playbook 0.4): in free flight thrust is body-z
    # only, so body-xy specific force = -D * v_body -- the accelerometer is
    # a scale-free velocity sensor. Dx=Dy=0.1 from sysID. Dz~0: no z info.
    use_drag: bool = True
    drag_d: float = 0.1
    drag_sigma: float = 1.0        # m/s per sample; 144 Hz aggregates hard
    drag_vmax: float = 15.0        # implied |v| above this = not drag physics
    # blind-leg holdout: (t_lo, t_hi) on the IMU boot axis. Inside the window
    # vision POSITION/VELOCITY fixes are withheld (anchors still recorded for
    # scoring) -- measures dead-reckoning drift against vision truth.
    deny_vision: tuple | None = None


@dataclass
class FusionResult:
    t_s: list = field(default_factory=list)
    p: list = field(default_factory=list)
    v: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    flow_updates: int = 0
    flow_rejects: int = 0


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
    """[(t_boot_s, g_cam (3,)) ...] for solved, confident, usable insts."""
    ev = []
    for d in c.detections:
        for inst in d.insts:
            if not inst.get("solved") or inst.get("low_confidence"):
                continue
            t_cam = inst.get("t_cam")
            if not t_cam:
                continue
            ev.append((d.sim_ns / 1e9 + bridge_off, np.asarray(t_cam, float)))
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
    roll, pitch = accel_implied_attitude(s0)
    yaw = 0.0
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
        gx, gy, gz = s.gyr
        roll += gx * dt
        pitch += -gy * dt   # wfix (vq2wp.py:203)
        yaw += -gz * dt     # wfix (vq2wp.py:204)
        a_lvl = accel_level(s.acc, roll, pitch)
        cyw, syw = math.cos(yaw), math.sin(yaw)
        a_w = np.array([cyw * a_lvl[0] - syw * a_lvl[1],
                        syw * a_lvl[0] + cyw * a_lvl[1], a_lvl[2]])
        kf.predict(a_w, dt)

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
            t_d, g_cam = dets[di]
            di += 1
            g_w = R_world_body(roll, pitch, yaw) @ (M_BODY_CAM @ g_cam)
            rng = float(np.linalg.norm(g_cam))
            p_vis_cands = [np.asarray(g) - g_w for g in cfg.gate_map]
            errs = [np.linalg.norm((pv - kf.p)[:2]) for pv in p_vis_cands]
            j = int(np.argmin(errs))
            if errs[j] > ACCEPT_R:
                continue
            p_vis = p_vis_cands[j]
            res.anchors.append({
                "t_boot_s": t_d,
                "p_vision": p_vis.copy(),
                "p_est": kf.p.copy(),
                "err_xy": float(np.linalg.norm((p_vis - kf.p)[:2])),
                "range": rng,
            })
            denied = (cfg.deny_vision is not None
                      and cfg.deny_vision[0] <= t_d <= cfg.deny_vision[1])
            if cfg.use_vision_pos and not denied:
                kf.update_position(
                    np.array([p_vis[0], p_vis[1], kf.p[2]]), rng
                )

        res.t_s.append(t_boot)
        res.p.append(kf.p.copy())
        res.v.append(kf.v.copy())
    return res

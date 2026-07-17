"""vp.py: Manhattan-vertical VP roll/pitch — synthetic-scene recovery,
frame-convention consistency, and degenerate-input rejection.

The synthetic renderer projects world-vertical 3D segments through the
EXACT production chain (camera.R_level_body / M_BODY_CAM / FX,CX), so a
pass here means the module's frame conventions agree with fusion.py's
by construction, not by luck.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from vq2.camera import CX, CY, FX, FY, H, M_BODY_CAM, R_level_body, W
from vq2.vp import (VpConfig, gravity_body, roll_pitch_from_gravity,
                    vp_roll_pitch)

DEG = math.pi / 180.0


# ---------------------------------------------------------------- helpers --
def render_vertical_scene(roll: float, pitch: float, yaw: float = 0.0,
                          n_posts: int = 10, seed: int = 0,
                          horizontal: bool = False) -> np.ndarray:
    """White world-vertical (or horizontal) segments on black, projected at
    the given attitude through the production camera model."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.uint8)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    R_wb = Rz @ R_level_body(roll, pitch)
    for _ in range(n_posts):
        # positions in the yawed (viewing) frame so the scene stays in the
        # frustum at any yaw — the attitude math must not care (yaw invariance)
        x, y = (Rz @ np.array([rng.uniform(3.0, 12.0),
                               rng.uniform(-6.0, 6.0), 0.0]))[:2]
        if horizontal:
            z = rng.uniform(-3.0, -0.5)
            a_w = np.array([x, y - 1.0, z])
            b_w = np.array([x, y + 1.0, z])
        else:
            a_w = np.array([x, y, -3.0])   # z down: post top
            b_w = np.array([x, y, 0.0])    # floor
        uv = []
        for p_w in (a_w, b_w):
            p_cam = M_BODY_CAM.T @ (R_wb.T @ p_w)
            if p_cam[2] < 0.5:
                break
            uv.append((int(round(FX * p_cam[0] / p_cam[2] + CX)),
                       int(round(FY * p_cam[1] / p_cam[2] + CY))))
        if len(uv) == 2:
            cv2.line(img, uv[0], uv[1], 255, 2)
    return img


# ------------------------------------------------------------ conventions --
def test_gravity_body_matches_R_level_body():
    for roll, pitch in [(0.0, 0.0), (0.1, -0.2), (-0.3, 0.31), (0.02, 0.3)]:
        expect = R_level_body(roll, pitch).T @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(gravity_body(roll, pitch), expect, atol=1e-12)


def test_roll_pitch_from_gravity_roundtrip():
    for roll, pitch in [(0.0, 0.0), (0.15, -0.25), (-0.1, 0.3)]:
        r, p = roll_pitch_from_gravity(gravity_body(roll, pitch))
        assert abs(r - roll) < 1e-9 and abs(p - pitch) < 1e-9


# --------------------------------------------------------------- recovery --
@pytest.mark.parametrize("roll,pitch", [
    (0.0, 0.0),
    (3.0 * DEG, 0.0),
    (0.0, -4.0 * DEG),
    (-2.0 * DEG, 3.0 * DEG),
    (5.0 * DEG, -18.0 * DEG),   # spawn-tilt-magnitude attitude
])
def test_recovers_known_attitude(roll, pitch):
    img = render_vertical_scene(roll, pitch)
    # prior offset by the drift magnitudes we care about (1-3 deg)
    res = vp_roll_pitch(img, roll + 2.0 * DEG, pitch - 2.0 * DEG)
    assert res is not None
    assert abs(res.roll - roll) < 0.6 * DEG
    assert abs(res.pitch - pitch) < 0.6 * DEG
    assert res.quality > 0.5


def test_yaw_invariant():
    roll, pitch = 2.0 * DEG, -3.0 * DEG
    for yaw in (0.0, 40.0 * DEG, -110.0 * DEG):
        img = render_vertical_scene(roll, pitch, yaw=yaw, seed=3)
        res = vp_roll_pitch(img, roll, pitch)
        assert res is not None
        assert abs(res.roll - roll) < 0.6 * DEG
        assert abs(res.pitch - pitch) < 0.6 * DEG


def test_survives_horizontal_clutter():
    """Vertical posts + floor-parallel clutter: the prior gate must keep the
    solve on the vertical VP."""
    roll, pitch = 2.0 * DEG, -2.0 * DEG
    img = render_vertical_scene(roll, pitch, n_posts=10, seed=1)
    img |= render_vertical_scene(roll, pitch, n_posts=8, seed=2,
                                 horizontal=True)
    res = vp_roll_pitch(img, roll + 1.5 * DEG, pitch - 1.5 * DEG)
    assert res is not None
    assert abs(res.roll - roll) < 0.7 * DEG
    assert abs(res.pitch - pitch) < 0.7 * DEG


# -------------------------------------------------------------- rejection --
def test_blank_image_rejected():
    assert vp_roll_pitch(np.zeros((H, W), np.uint8), 0.0, 0.0) is None


def test_horizontal_only_rejected():
    """Nothing near-vertical in view: the gate must starve the solve, not
    return a horizontal VP as attitude."""
    img = render_vertical_scene(0.0, 0.0, horizontal=True, n_posts=12)
    assert vp_roll_pitch(img, 0.0, 0.0) is None


def test_far_prior_rejected():
    """A solution further than max_dev_deg from the prior is refused —
    the corrector must not jump to a different VP."""
    img = render_vertical_scene(0.0, 0.0)
    cfg = VpConfig(gate_deg=25.0, max_dev_deg=4.0)
    res = vp_roll_pitch(img, 12.0 * DEG, 0.0, cfg)
    assert res is None or abs(math.degrees(res.roll)) < 16.0

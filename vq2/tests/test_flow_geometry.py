import numpy as np

from vq2 import camera
from vq2.flow_vel import robust_velocity, velocity_from_tracks


def _project(points_w, pos_w, att):
    """World floor points -> pixel coords for a drone at pos_w with att."""
    R_wb = camera.R_world_body(*att)
    r_b = (points_w - pos_w) @ R_wb  # rows: rel vector rotated into body
    r_c = r_b @ camera.M_BODY_CAM    # body -> camera frame
    uv = np.stack(
        [r_c[:, 0] / r_c[:, 2] * camera.FX + camera.CX,
         r_c[:, 1] / r_c[:, 2] * camera.FY + camera.CY], axis=1
    )
    ok = (
        (r_c[:, 2] > 0.1)
        & (uv[:, 0] > 0) & (uv[:, 0] < camera.W - 1)
        & (uv[:, 1] > 0) & (uv[:, 1] < camera.H - 1)
    )
    return uv, ok


def _floor_points():
    xs, ys = np.meshgrid(np.linspace(3.0, 14.0, 12), np.linspace(-3.0, 3.0, 7))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    return pts


def _make_pair(pos0, pos1, att0, att1):
    pts = _floor_points()
    uv0, ok0 = _project(pts, pos0, att0)
    uv1, ok1 = _project(pts, pos1, att1)
    ok = ok0 & ok1
    return uv0[ok], uv1[ok]


def test_forward_motion_recovered_exactly():
    dt = 1.0 / 30.0
    pos0 = np.array([0.0, 0.0, -1.3])          # z down, 1.3 m up
    pos1 = pos0 + np.array([1.5, 0.0, 0.0]) * dt
    att = (0.0, 0.0, 0.0)
    uv0, uv1 = _make_pair(pos0, pos1, att, att)
    h = 0.0 - pos0[2]  # floor at z=0
    v, used = velocity_from_tracks(uv0, uv1, att, att, h, h, dt)
    assert used.sum() >= 10
    est = robust_velocity(v)
    assert est is not None
    v_est, sigma, n = est
    assert np.allclose(v_est, [1.5, 0.0, 0.0], atol=1e-6)


def test_lateral_and_vertical_motion_recovered():
    dt = 1.0 / 30.0
    pos0 = np.array([0.0, 0.0, -1.3])
    v_true = np.array([0.8, -0.6, 0.3])  # includes descent (z down +)
    pos1 = pos0 + v_true * dt
    att = (0.05, -0.1, 0.3)  # tilted + yawed
    uv0, uv1 = _make_pair(pos0, pos1, att, att)
    h0 = 0.0 - pos0[2]
    h1 = 0.0 - pos1[2]
    v, used = velocity_from_tracks(uv0, uv1, att, att, h0, h1, dt)
    v_est, sigma, n = robust_velocity(v)
    assert np.allclose(v_est, v_true, atol=1e-6)


def test_pure_yaw_rotation_gives_zero_velocity():
    dt = 1.0 / 30.0
    pos = np.array([0.0, 0.0, -1.3])
    att0 = (0.0, 0.0, 0.0)
    att1 = (0.0, 0.0, 0.12)  # ~7 deg/frame spin, no translation
    uv0, uv1 = _make_pair(pos, pos, att0, att1)
    h = 1.3
    v, used = velocity_from_tracks(uv0, uv1, att0, att1, h, h, dt)
    v_est, sigma, n = robust_velocity(v)
    assert np.linalg.norm(v_est) < 1e-6


def test_height_error_scales_speed_proportionally():
    dt = 1.0 / 30.0
    pos0 = np.array([0.0, 0.0, -1.3])
    pos1 = pos0 + np.array([1.5, 0.0, 0.0]) * dt
    att = (0.0, 0.0, 0.0)
    uv0, uv1 = _make_pair(pos0, pos1, att, att)
    v, _ = velocity_from_tracks(uv0, uv1, att, att, 1.3 * 1.1, 1.3 * 1.1, dt)
    v_est, _, _ = robust_velocity(v)
    assert abs(v_est[0] / 1.5 - 1.1) < 0.01  # 10% h error -> 10% v error


def test_shallow_rays_rejected():
    # features near the horizon (declination < min) must be masked out
    dt = 1.0 / 30.0
    pos = np.array([0.0, 0.0, -1.3])
    att = (0.0, 0.0, 0.0)
    # top rows of the image look ABOVE the horizon with 20 deg up tilt
    uv_top = np.array([[320.0, 5.0], [100.0, 10.0], [500.0, 8.0]])
    v, used = velocity_from_tracks(uv_top, uv_top, att, att, 1.3, 1.3, dt)
    assert used.sum() == 0


def test_robust_velocity_rejects_outliers_and_small_n():
    good = np.tile([1.0, 0.0, 0.0], (20, 1))
    bad = np.tile([15.0, -9.0, 4.0], (3, 1))
    v_est, sigma, n = robust_velocity(np.vstack([good, bad]))
    assert np.allclose(v_est, [1.0, 0.0, 0.0], atol=1e-9)
    assert n == 20
    assert robust_velocity(good[:4]) is None  # below min_tracks

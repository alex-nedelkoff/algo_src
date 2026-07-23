import math

import numpy as np

from vq2 import camera


def test_center_pixel_ray_is_optical_axis():
    r = camera.pixel_rays_body([[camera.CX, camera.CY]])[0]
    # optical axis 20 deg above horizon in body frame (z down)
    elev = math.degrees(math.asin(-r[2]))
    assert abs(elev - 20.0) < 1e-6
    assert abs(r[1]) < 1e-9  # no lateral component


def test_bottom_center_ray_looks_below_horizon():
    r = camera.pixel_rays_body([[camera.CX, camera.H - 1.0]])[0]
    elev = math.degrees(math.asin(-r[2]))
    # 20 up - atan(179.5/320)=29.29 down => ~-9.3 deg (floor visible)
    assert -10.0 < elev < -8.5


def test_rays_are_unit_norm():
    uv = np.array([[0.0, 0.0], [639.0, 359.0], [320.0, 100.0]])
    r = camera.pixel_rays_body(uv)
    assert np.allclose(np.linalg.norm(r, axis=1), 1.0)


def test_R_level_body_matches_accel_level_composition():
    from vq2.eskf import accel_level
    roll, pitch = 0.21, -0.13
    acc = np.array([1.1, -2.2, -9.0])
    via_helper = camera.R_level_body(roll, pitch) @ acc + np.array([0, 0, 9.81])
    assert np.allclose(via_helper, accel_level(acc, roll, pitch))


def test_R_world_body_yaw_matches_vq2wp_rotation():
    yaw = 0.7
    v_lvl = np.array([1.0, 2.0, 3.0])
    cyw, syw = math.cos(yaw), math.sin(yaw)
    expected = np.array([cyw * v_lvl[0] - syw * v_lvl[1],
                         syw * v_lvl[0] + cyw * v_lvl[1], v_lvl[2]])
    got = camera.R_world_body(0.0, 0.0, yaw) @ v_lvl
    assert np.allclose(got, expected)

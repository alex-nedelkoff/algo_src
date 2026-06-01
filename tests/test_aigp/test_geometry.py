import numpy as np
from aigp.geometry import (
    K, IMG_W, IMG_H, quat_to_R, R_BODY_TO_CAM,
    world_to_camera, project, in_frame,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, body x = world north


def test_intrinsics():
    assert (IMG_W, IMG_H) == (640, 360)
    assert np.allclose(K, [[320, 0, 320], [0, 320, 180], [0, 0, 1]])


def test_quat_to_R_orthonormal():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    R = quat_to_R(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_gate_dead_ahead_projects_to_center():
    # drone at origin facing north (identity), gate 10 m north
    p_cam = world_to_camera(np.array([10.0, 0.0, 0.0]), np.zeros(3), IDENT)
    assert np.allclose(p_cam, [0.0, 0.0, 10.0])  # optical: right, down, forward
    u, v = project(p_cam)
    assert np.isclose(u, 320.0) and np.isclose(v, 180.0)
    assert in_frame(u, v)


def test_gate_to_east_projects_right():
    p_cam = world_to_camera(np.array([10.0, 2.0, 0.0]), np.zeros(3), IDENT)
    u, v = project(p_cam)
    assert u > 320.0 and np.isclose(v, 180.0)


def test_gate_below_projects_down():
    # +z is down in NED
    p_cam = world_to_camera(np.array([10.0, 0.0, 2.0]), np.zeros(3), IDENT)
    u, v = project(p_cam)
    assert np.isclose(u, 320.0) and v > 180.0


def test_gate_behind_is_none():
    p_cam = world_to_camera(np.array([-10.0, 0.0, 0.0]), np.zeros(3), IDENT)
    assert project(p_cam) is None

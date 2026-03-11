"""Pinhole camera projection and wireframe rasterization (pure numpy)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def quat_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert (w, x, y, z) quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion array of shape (4,) in (w, x, y, z) order.

    Returns:
        3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class PinholeCamera:
    """Pinhole camera model with intrinsic parameters.

    Attributes:
        fx: Focal length in pixels (horizontal).
        fy: Focal length in pixels (vertical).
        cx: Principal point x coordinate (pixels).
        cy: Principal point y coordinate (pixels).
        width: Image width in pixels.
        height: Image height in pixels.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 320
    height: int = 240

    @classmethod
    def from_hfov(cls, hfov_deg: float, width: int, height: int) -> PinholeCamera:
        """Create camera from horizontal field of view.

        Args:
            hfov_deg: Horizontal field of view in degrees.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            PinholeCamera with symmetric focal lengths derived from hfov.
        """
        hfov_rad = math.radians(hfov_deg)
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def project_points(
    camera: PinholeCamera,
    points_world: NDArray[np.float64],
    camera_pos: NDArray[np.float64],
    camera_quat: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Project 3D world points to 2D pixel coordinates.

    Args:
        camera: Pinhole camera intrinsics.
        points_world: (N, 3) array of world-frame 3D points.
        camera_pos: (3,) camera position in world frame.
        camera_quat: (4,) camera orientation quaternion (w, x, y, z) in world frame.

    Returns:
        Tuple of:
            uv: (N, 2) pixel coordinates (u, v).
            valid: (N,) boolean mask — True if point is in front of camera and
                   within image bounds.
    """
    # R_body_to_world from the drone orientation quaternion
    R_body_to_world = quat_to_rotation_matrix(camera_quat)
    R_world_to_body = R_body_to_world.T

    # Transform to body frame
    p_rel = points_world - camera_pos[np.newaxis, :]  # (N, 3)
    p_body = (R_world_to_body @ p_rel.T).T  # (N, 3)

    # Body frame: X=forward, Y=left, Z=up
    # Pinhole camera convention: Z=forward (depth), X=right, Y=down
    # Mapping: cam_X = -body_Y, cam_Y = -body_Z, cam_Z = body_X
    x_cam = -p_body[:, 1]
    y_cam = -p_body[:, 2]
    z_cam = p_body[:, 0]

    # Avoid division by zero
    z_safe = np.where(z_cam > 1e-6, z_cam, 1e-6)

    # Project through pinhole
    u = camera.fx * (x_cam / z_safe) + camera.cx
    v = camera.fy * (y_cam / z_safe) + camera.cy

    uv = np.stack([u, v], axis=-1)

    # Valid mask: in front of camera and within image bounds
    valid = (
        (z_cam > 1e-6)
        & (u >= 0)
        & (u < camera.width)
        & (v >= 0)
        & (v < camera.height)
    )

    return uv, valid


def _draw_line(
    img: NDArray[np.uint8],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a line on the image using Bresenham's algorithm.

    Modifies img in-place. Coordinates are (x=column, y=row).
    """
    h, w = img.shape[:2]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _clip_line_to_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[int, int, int, int] | None:
    """Clip a line segment to a rectangle using Cohen-Sutherland.

    Returns clipped integer coordinates or None if entirely outside.
    """
    INSIDE = 0
    LEFT = 1
    RIGHT = 2
    BOTTOM = 4
    TOP = 8

    def _outcode(x: float, y: float) -> int:
        code = INSIDE
        if x < xmin:
            code |= LEFT
        elif x > xmax:
            code |= RIGHT
        if y < ymin:
            code |= BOTTOM
        elif y > ymax:
            code |= TOP
        return code

    outcode0 = _outcode(x0, y0)
    outcode1 = _outcode(x1, y1)

    for _ in range(20):  # max iterations to prevent infinite loops
        if not (outcode0 | outcode1):
            # Both inside
            return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        if outcode0 & outcode1:
            # Both outside same side
            return None

        # Pick the point outside the rectangle
        outcode_out = outcode0 if outcode0 else outcode1
        dx = x1 - x0
        dy = y1 - y0

        if outcode_out & TOP:
            x = x0 + dx * (ymax - y0) / dy if dy != 0 else x0
            y = ymax
        elif outcode_out & BOTTOM:
            x = x0 + dx * (ymin - y0) / dy if dy != 0 else x0
            y = ymin
        elif outcode_out & RIGHT:
            y = y0 + dy * (xmax - x0) / dx if dx != 0 else y0
            x = xmax
        elif outcode_out & LEFT:
            y = y0 + dy * (xmin - x0) / dx if dx != 0 else y0
            x = xmin
        else:
            return None

        if outcode_out == outcode0:
            x0, y0 = x, y
            outcode0 = _outcode(x0, y0)
        else:
            x1, y1 = x, y
            outcode1 = _outcode(x1, y1)

    return None


def render_wireframe(
    camera: PinholeCamera,
    edges: list[tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int, int]]],
    camera_pos: NDArray[np.float64],
    camera_quat: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """Render wireframe edges onto a black background image.

    Args:
        camera: Pinhole camera intrinsics.
        edges: List of (point_a, point_b, color) tuples where point_a and point_b
               are (3,) world coordinates and color is (R, G, B) integers.
        camera_pos: (3,) camera position in world frame.
        camera_quat: (4,) camera orientation quaternion (w, x, y, z).

    Returns:
        (height, width, 3) uint8 image with wireframe lines rendered.
    """
    img = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)

    if not edges:
        return img

    # Collect all unique endpoints for batch projection
    all_points = []
    for pt_a, pt_b, _color in edges:
        all_points.append(pt_a)
        all_points.append(pt_b)

    all_points_arr = np.array(all_points, dtype=np.float64)  # (2*E, 3)
    uv, valid = project_points(camera, all_points_arr, camera_pos, camera_quat)

    for i, (_pt_a, _pt_b, color) in enumerate(edges):
        idx_a = 2 * i
        idx_b = 2 * i + 1

        # Both endpoints must be in front of camera (z > 0)
        if not valid[idx_a] and not valid[idx_b]:
            continue

        u0, v0 = uv[idx_a]
        u1, v1 = uv[idx_b]

        # Clip line to image bounds
        clipped = _clip_line_to_rect(
            u0, v0, u1, v1,
            0, 0, camera.width - 1, camera.height - 1,
        )
        if clipped is None:
            continue

        x0, y0, x1, y1 = clipped
        _draw_line(img, x0, y0, x1, y1, color)

    return img


def render_horizon(
    camera: PinholeCamera,
    camera_pos: NDArray[np.float64],
    camera_quat: NDArray[np.float64],
) -> list[tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int, int]]]:
    """Generate edges for a horizon line at z=0 world plane.

    Samples two far-away points on the z=0 plane to form a horizon line.

    Args:
        camera: Pinhole camera intrinsics.
        camera_pos: (3,) camera position in world frame.
        camera_quat: (4,) camera orientation quaternion (w, x, y, z).

    Returns:
        List of edge tuples [(point_a, point_b, color)] for the horizon line.
    """
    # Place two points far away on the z=0 plane, perpendicular to camera's
    # forward direction projected onto the ground plane.
    R = quat_to_rotation_matrix(camera_quat)
    forward = R[:, 0]  # camera x-axis = forward in body frame

    # Project forward onto z=0 plane
    forward_2d = np.array([forward[0], forward[1], 0.0])
    norm = np.linalg.norm(forward_2d)
    if norm < 1e-6:
        # Camera looking straight down/up — no meaningful horizon
        return []

    forward_2d /= norm
    # Perpendicular direction on ground plane
    right_2d = np.array([-forward_2d[1], forward_2d[0], 0.0])

    far_dist = 100.0  # meters
    ground_center = np.array([camera_pos[0], camera_pos[1], 0.0])

    pt_left = ground_center - right_2d * far_dist
    pt_right = ground_center + right_2d * far_dist

    # Dim green for horizon
    color = (0, 80, 0)
    return [(pt_left, pt_right, color)]

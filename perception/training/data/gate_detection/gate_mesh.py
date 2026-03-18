"""Procedural gate and drone mesh generators.

Produces simple geometric meshes for synthetic data rendering (BlenderProc).
Gate inner corners are used for ground-truth 2D keypoint projection.

All meshes are in gate-local coordinates: XY plane, centered at origin.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger(__name__)


def get_gate_inner_corners(
    inner_width: float = 1.5,
    inner_height: float = 1.5,
) -> np.ndarray:
    """Return the 4 inner corner positions of the gate aperture.

    Corners are in gate-local frame (XY plane, centered at origin).
    Order: TL, TR, BR, BL.

    Args:
        inner_width: Width of the gate opening (X axis).
        inner_height: Height of the gate opening (Y axis).

    Returns:
        (4, 3) float64 array of corner 3D positions.
    """
    w2 = inner_width / 2.0
    h2 = inner_height / 2.0
    return np.array(
        [
            [-w2, +h2, 0.0],  # TL
            [+w2, +h2, 0.0],  # TR
            [+w2, -h2, 0.0],  # BR
            [-w2, -h2, 0.0],  # BL
        ],
        dtype=np.float64,
    )


def generate_gate_mesh(
    inner_width: float = 1.5,
    inner_height: float = 1.5,
    frame_width: float = 0.1,
    frame_depth: float = 0.05,
) -> trimesh.Trimesh:
    """Generate a hollow rectangular gate frame mesh.

    Built from 4 box primitives (top, bottom, left, right bars) concatenated
    together. No boolean operations are used.

    The gate lies in the XY plane, centered at the origin, with the frame
    extending in Z by *frame_depth*.

    Args:
        inner_width: Width of the gate opening (X axis).
        inner_height: Height of the gate opening (Y axis).
        frame_width: Thickness of the frame bars.
        frame_depth: Depth of the frame (Z axis extent).

    Returns:
        A :class:`trimesh.Trimesh` of the gate frame.
    """
    outer_w = inner_width + 2 * frame_width
    outer_h = inner_height + 2 * frame_width

    bars: list[trimesh.Trimesh] = []

    # Top bar: spans full outer width, frame_width tall, frame_depth deep
    # Center Y = inner_height/2 + frame_width/2
    top = trimesh.creation.box(
        extents=[outer_w, frame_width, frame_depth],
    )
    top.apply_translation([0.0, inner_height / 2 + frame_width / 2, 0.0])
    bars.append(top)

    # Bottom bar: same as top, mirrored in Y
    bottom = trimesh.creation.box(
        extents=[outer_w, frame_width, frame_depth],
    )
    bottom.apply_translation([0.0, -(inner_height / 2 + frame_width / 2), 0.0])
    bars.append(bottom)

    # Left bar: spans inner_height (only the gap between top and bottom),
    # frame_width wide, frame_depth deep.
    # Center X = -(inner_width/2 + frame_width/2)
    left = trimesh.creation.box(
        extents=[frame_width, inner_height, frame_depth],
    )
    left.apply_translation([-(inner_width / 2 + frame_width / 2), 0.0, 0.0])
    bars.append(left)

    # Right bar: same as left, mirrored in X
    right = trimesh.creation.box(
        extents=[frame_width, inner_height, frame_depth],
    )
    right.apply_translation([inner_width / 2 + frame_width / 2, 0.0, 0.0])
    bars.append(right)

    mesh = trimesh.util.concatenate(bars)
    log.debug(
        "Generated gate mesh: %d verts, %d faces, outer %.2f x %.2f",
        len(mesh.vertices),
        len(mesh.faces),
        outer_w,
        outer_h,
    )
    return mesh


def generate_drone_mesh(size: float = 0.3) -> trimesh.Trimesh:
    """Generate a simple placeholder drone mesh.

    Cross-shaped body with 4 rotor discs. All dimensions scale with *size*.

    Args:
        size: Overall scale factor for the drone.

    Returns:
        A :class:`trimesh.Trimesh` of the drone.
    """
    parts: list[trimesh.Trimesh] = []

    # Cross-shaped body: two intersecting elongated boxes
    arm_length = size * 1.5
    arm_thickness = size * 0.15
    arm_height = size * 0.1

    # Arm 1: along X
    arm1 = trimesh.creation.box(
        extents=[arm_length, arm_thickness, arm_height],
    )
    parts.append(arm1)

    # Arm 2: along Y
    arm2 = trimesh.creation.box(
        extents=[arm_thickness, arm_length, arm_height],
    )
    parts.append(arm2)

    # Central body hub
    hub = trimesh.creation.box(
        extents=[size * 0.3, size * 0.3, arm_height * 1.5],
    )
    parts.append(hub)

    # 4 rotor discs at arm tips
    rotor_radius = size * 0.25
    rotor_height = size * 0.02
    arm_half = arm_length / 2

    rotor_positions = [
        [+arm_half, 0.0, arm_height / 2],
        [-arm_half, 0.0, arm_height / 2],
        [0.0, +arm_half, arm_height / 2],
        [0.0, -arm_half, arm_height / 2],
    ]

    for pos in rotor_positions:
        rotor = trimesh.creation.cylinder(
            radius=rotor_radius,
            height=rotor_height,
            sections=16,
        )
        rotor.apply_translation(pos)
        parts.append(rotor)

    mesh = trimesh.util.concatenate(parts)
    log.debug(
        "Generated drone mesh: %d verts, %d faces, size=%.2f",
        len(mesh.vertices),
        len(mesh.faces),
        size,
    )
    return mesh


def export_assets(output_dir: str | Path) -> None:
    """Generate and export gate and drone meshes as OBJ files.

    Args:
        output_dir: Directory to write ``gate.obj`` and ``drone.obj`` into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate = generate_gate_mesh()
    gate_path = output_dir / "gate.obj"
    gate.export(str(gate_path))
    log.info("Exported gate mesh to %s", gate_path)

    drone = generate_drone_mesh()
    drone_path = output_dir / "drone.obj"
    drone.export(str(drone_path))
    log.info("Exported drone mesh to %s", drone_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    assets_dir = Path(__file__).parent / "assets"
    export_assets(assets_dir)
    print(f"Assets exported to {assets_dir}")

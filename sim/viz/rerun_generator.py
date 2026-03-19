"""Convert .npz trajectory files to rerun .rrd archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sim.viz.pinhole import PinholeCamera, quat_to_rotation_matrix, render_wireframe, render_horizon
from perception.training.data.gate_detection.gate_mesh import (
    generate_gate_mesh,
    generate_drone_mesh,
)

try:
    import rerun as rr

    _HAS_RERUN = True
except ImportError:
    _HAS_RERUN = False


def _require_rerun() -> None:
    """Raise an informative error if rerun-sdk is not installed."""
    if not _HAS_RERUN:
        raise ImportError(
            "rerun-sdk is required for visualization but is not installed.\n"
            "Install it with:  pip install 'rerun-sdk>=0.22'\n"
            "Or install the viz extras:  pip install -e '.[viz]'"
        )


# Gate color cycle — 8 distinct colors for up to 8 gates
_GATE_COLORS: list[tuple[int, int, int]] = [
    (255, 80, 80),    # red
    (80, 255, 80),    # green
    (80, 80, 255),    # blue
    (255, 255, 80),   # yellow
    (255, 80, 255),   # magenta
    (80, 255, 255),   # cyan
    (255, 160, 80),   # orange
    (160, 80, 255),   # purple
]


def _speed_to_color(speed: float, speed_min: float, speed_max: float) -> tuple[int, int, int]:
    """Map speed scalar to blue-to-red colormap.

    Args:
        speed: Speed value.
        speed_min: Minimum speed for normalization.
        speed_max: Maximum speed for normalization.

    Returns:
        (R, G, B) color tuple.
    """
    if speed_max - speed_min < 1e-6:
        t = 0.5
    else:
        t = np.clip((speed - speed_min) / (speed_max - speed_min), 0.0, 1.0)
    # Blue (0,0,255) -> Red (255,0,0) via white midpoint
    r = int(255 * t)
    g = 0
    b = int(255 * (1.0 - t))
    return (r, g, b)


def _gate_corners(
    position: NDArray[np.float64],
    orientation: NDArray[np.float64],
    half_extents: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute 4 world-frame corners of a gate rectangle.

    Args:
        position: (3,) gate center in world frame.
        orientation: (4,) gate quaternion (w, x, y, z).
        half_extents: (2,) [half_width, half_height].

    Returns:
        (4, 3) array of corner positions in world frame, ordered:
        top-left, top-right, bottom-right, bottom-left.
    """
    hw, hh = half_extents[0], half_extents[1]
    R = quat_to_rotation_matrix(orientation)

    # Local corners in gate frame: gate normal is along local x-axis,
    # width along local y-axis, height along local z-axis.
    local_corners = np.array([
        [0.0, -hw,  hh],  # top-left
        [0.0,  hw,  hh],  # top-right
        [0.0,  hw, -hh],  # bottom-right
        [0.0, -hw, -hh],  # bottom-left
    ])

    world_corners = (R @ local_corners.T).T + position[np.newaxis, :]
    return world_corners


def _gate_wireframe_edges(
    gate_positions: NDArray[np.float64],
    gate_orientations: NDArray[np.float64],
    gate_half_extents: NDArray[np.float64],
) -> list[list[tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int, int]]]]:
    """Compute wireframe edges for all gates.

    Returns:
        List of per-gate edge lists. Each edge is (point_a, point_b, color).
    """
    num_gates = gate_positions.shape[0]
    all_gate_edges: list[list[tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int, int]]]] = []

    for g in range(num_gates):
        corners = _gate_corners(
            gate_positions[g], gate_orientations[g], gate_half_extents[g]
        )
        color = _GATE_COLORS[g % len(_GATE_COLORS)]
        edges = []
        for i in range(4):
            j = (i + 1) % 4
            edges.append((corners[i], corners[j], color))
        all_gate_edges.append(edges)

    return all_gate_edges


def _transform_gate_mesh_verts(
    mesh_verts: NDArray[np.float64],
    position: NDArray[np.float64],
    orientation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform gate mesh vertices from mesh-local to world frame.

    gate_mesh.py produces meshes in XY plane with normal along Z.
    The sim uses gate normal along local X-axis. We apply Ry(+90deg)
    to align Z->X, then the per-gate world rotation and translation.
    """
    R_mesh_to_sim = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    R_gate = quat_to_rotation_matrix(orientation)
    return (mesh_verts @ R_mesh_to_sim.T) @ R_gate.T + position[np.newaxis, :]


def generate_rrd(
    npz_path: str | Path,
    output_path: str | Path | None = None,
    camera_decimation: int = 10,
) -> Path:
    """Convert a single .npz trajectory file to a .rrd rerun archive.

    Args:
        npz_path: Path to the .npz trajectory file.
        output_path: Path for the output .rrd file. If None, places it in a
                     sibling ``rerun/`` directory mirroring the trajectory structure.
        camera_decimation: Render pinhole camera view every N steps.

    Returns:
        Path to the generated .rrd file.
    """
    _require_rerun()

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {npz_path}")

    # Determine output path
    if output_path is None:
        # Default: sibling rerun/ directory
        # e.g. .../trajectories/step_1000000/eval_ep_0.npz
        #   -> .../rerun/step_1000000/eval_ep_0.rrd
        traj_parent = npz_path.parent
        if traj_parent.parent.name == "trajectories":
            step_dir_name = traj_parent.name
            rerun_dir = traj_parent.parent.parent / "rerun" / step_dir_name
        else:
            rerun_dir = traj_parent / "rerun"
        rerun_dir.mkdir(parents=True, exist_ok=True)
        output_path = rerun_dir / npz_path.with_suffix(".rrd").name
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load trajectory data
    data = np.load(npz_path, allow_pickle=True)

    schema_version = int(data["schema_version"])
    if schema_version not in (1, 2):
        raise ValueError(
            f"Unsupported trajectory schema version {schema_version} "
            f"(expected 1 or 2). File: {npz_path}"
        )

    positions = data["positions"]          # (T, 3)
    quaternions = data["quaternions"]       # (T, 4) w,x,y,z
    velocities = data["velocities"]        # (T, 3)
    body_rates = data["body_rates"]        # (T, 3)
    motor_rpms = data["motor_rpms"]        # (T, 4)
    rewards = data["rewards"]              # (T,)
    gate_events = data["gate_events"]      # (E, 2)
    gate_positions = data["gate_positions"]        # (G, 3)
    gate_orientations = data["gate_orientations"]  # (G, 4)
    gate_half_extents = data["gate_half_extents"]  # (G, 2)
    dt = float(data["dt"])

    T = positions.shape[0]

    # Precompute speeds for trail coloring
    speeds = np.linalg.norm(velocities, axis=1)  # (T,)
    speed_min = float(speeds.min())
    speed_max = float(speeds.max())

    # Precompute cumulative reward
    reward_cumsum = np.cumsum(rewards)

    # Precompute accelerations via finite difference of velocities
    accelerations = np.zeros_like(velocities)
    if T > 1:
        accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt
        accelerations[0] = accelerations[1] if T > 1 else 0.0

    # Precompute gate wireframe edges and corners for camera view
    all_gate_edges = _gate_wireframe_edges(
        gate_positions, gate_orientations, gate_half_extents
    )

    # Build gate event lookup: timestep -> gate_idx
    gate_event_map: dict[int, int] = {}
    if gate_events.size > 0:
        for row in gate_events:
            gate_event_map[int(row[0])] = int(row[1])

    # Pinhole camera for synthetic view
    cam = PinholeCamera.from_hfov(90.0, 320, 240)

    # --- Initialize rerun recording ---
    rr.init("drone_racing_viz", spawn=False)
    rr.save(str(output_path))

    # --- 3a. Static scene: gates ---
    _base_gate_mesh = generate_gate_mesh()
    _base_gate_verts = np.array(_base_gate_mesh.vertices, dtype=np.float64)
    _base_gate_faces = np.array(_base_gate_mesh.faces, dtype=np.int32)

    for g in range(gate_positions.shape[0]):
        color = _GATE_COLORS[g % len(_GATE_COLORS)]

        # Solid mesh
        world_verts = _transform_gate_mesh_verts(
            _base_gate_verts, gate_positions[g], gate_orientations[g],
        )
        vert_colors = np.tile(
            np.array(color, dtype=np.uint8), (len(world_verts), 1),
        )
        rr.log(
            f"world/gates/gate_{g}",
            rr.Mesh3D(
                vertex_positions=world_verts.astype(np.float32),
                triangle_indices=_base_gate_faces,
                vertex_colors=vert_colors,
            ),
            static=True,
        )

    # --- 3a-bis. Gate markers: always-visible points + labels + ground projections ---
    # These ensure gates are visible from any viewing angle (top-down, side, etc.)
    # since the wireframe rectangles become edge-on and invisible from some angles.
    _gate_marker_radius = 0.15  # meters
    _ground_cross_arm = 0.5  # half-length of ground cross arms in meters

    for g in range(gate_positions.shape[0]):
        pos = gate_positions[g]
        color = _GATE_COLORS[g % len(_GATE_COLORS)]

        # 1) Center marker sphere — visible from any angle
        rr.log(
            f"world/gates/gate_{g}/marker",
            rr.Points3D(
                [pos],
                radii=[_gate_marker_radius],
                colors=[color],
                labels=[f"G{g}"],
            ),
            static=True,
        )

        # 2) Vertical stalk from gate center down to ground (z=0)
        stalk = np.array([[pos[0], pos[1], 0.0], pos])
        rr.log(
            f"world/gates/gate_{g}/stalk",
            rr.LineStrips3D(
                [stalk],
                colors=[(color[0] // 2, color[1] // 2, color[2] // 2)],
            ),
            static=True,
        )

        # 3) Ground-plane cross at the gate's XY position (z=0)
        gx, gy = pos[0], pos[1]
        cross_lines = np.array([
            [gx - _ground_cross_arm, gy, 0.0],
            [gx + _ground_cross_arm, gy, 0.0],
            [gx, gy - _ground_cross_arm, 0.0],
            [gx, gy + _ground_cross_arm, 0.0],
        ])
        rr.log(
            f"world/gates/gate_{g}/ground",
            rr.LineStrips3D(
                [cross_lines[:2], cross_lines[2:]],
                colors=[color, color],
            ),
            static=True,
        )

    # --- 3a-ter. Static scene: floor plane ---
    _floor_half = 20.0
    floor_verts = np.array([
        [-_floor_half, -_floor_half, 0.0],
        [+_floor_half, -_floor_half, 0.0],
        [+_floor_half, +_floor_half, 0.0],
        [-_floor_half, +_floor_half, 0.0],
    ], dtype=np.float32)
    floor_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    floor_colors = np.full((4, 3), 80, dtype=np.uint8)
    rr.log(
        "world/floor",
        rr.Mesh3D(
            vertex_positions=floor_verts,
            triangle_indices=floor_faces,
            vertex_colors=floor_colors,
        ),
        static=True,
    )

    _drone_mesh = generate_drone_mesh()
    _drone_verts = np.array(_drone_mesh.vertices, dtype=np.float32)
    _drone_faces = np.array(_drone_mesh.faces, dtype=np.int32)
    _drone_colors = np.full((len(_drone_verts), 3), 200, dtype=np.uint8)

    # --- 3b. Per-timestep logging ---
    for t in range(T):
        rr.set_time("step", sequence=t)
        rr.set_time("time", duration=t * dt)

        # Drone transform — swizzle quaternion from (w,x,y,z) to (x,y,z,w) for rerun
        q_wxyz = quaternions[t]
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        rr.log(
            "world/drone",
            rr.Transform3D(
                translation=positions[t],
                rotation=rr.Quaternion(xyzw=q_xyzw),
            ),
        )

        # Drone mesh (body frame, inherits world/drone transform)
        rr.log(
            "world/drone/mesh",
            rr.Mesh3D(
                vertex_positions=_drone_verts,
                triangle_indices=_drone_faces,
                vertex_colors=_drone_colors,
            ),
        )

        # Flight trail segment colored by speed (log only latest segment to avoid O(T^2))
        if t > 0:
            seg_color = _speed_to_color(speeds[t], speed_min, speed_max)
            rr.log(
                f"world/trail/seg_{t}",
                rr.LineStrips3D(
                    [np.array([positions[t - 1], positions[t]])],
                    colors=[seg_color],
                ),
            )

        # Telemetry: motor RPMs
        for m in range(4):
            rr.log(f"telemetry/motors/rpm_{m}", rr.Scalars(float(motor_rpms[t, m])))

        # Telemetry: body rates
        rate_names = ["roll", "pitch", "yaw"]
        for axis in range(3):
            rr.log(
                f"telemetry/body_rates/{rate_names[axis]}",
                rr.Scalars(float(body_rates[t, axis])),
            )

        # Telemetry: acceleration
        accel_names = ["x", "y", "z"]
        for axis in range(3):
            rr.log(
                f"telemetry/acceleration/{accel_names[axis]}",
                rr.Scalars(float(accelerations[t, axis])),
            )

        # Telemetry: reward
        rr.log("telemetry/reward", rr.Scalars(float(rewards[t])))
        rr.log("telemetry/reward_cumulative", rr.Scalars(float(reward_cumsum[t])))

        # Events: gate passages
        if t in gate_event_map:
            gate_idx = gate_event_map[t]
            rr.log("events", rr.TextLog(f"Gate {gate_idx} passed"))

        # Pinhole camera view (decimated)
        if t % camera_decimation == 0:
            # Camera is body-mounted, same pose as drone
            cam_pos = positions[t]
            cam_quat = quaternions[t]  # (w, x, y, z)

            # Collect all gate edges into a flat list
            all_edges: list[tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int, int]]] = []
            for gate_edges in all_gate_edges:
                all_edges.extend(gate_edges)

            # Add horizon edges
            horizon_edges = render_horizon(cam, cam_pos, cam_quat)
            all_edges.extend(horizon_edges)

            # Render wireframe image
            frame = render_wireframe(cam, all_edges, cam_pos, cam_quat)
            rr.log("drone/camera", rr.Image(frame))

    # Flush all pending data and close the file sink before returning.
    # Without this, the ArtifactUploader may upload a partially-written
    # .rrd file (missing chunks that haven't been flushed yet).
    rr.disconnect()

    return output_path


def batch_generate(
    trajectory_dir: str | Path,
    output_dir: str | Path | None = None,
    camera_decimation: int = 10,
) -> list[Path]:
    """Process all .npz trajectory files in a directory.

    Args:
        trajectory_dir: Directory containing .npz files (searched recursively).
        output_dir: Output directory for .rrd files. If None, uses sibling
                    ``rerun/`` directory convention.
        camera_decimation: Render pinhole camera view every N steps.

    Returns:
        List of paths to generated .rrd files.
    """
    _require_rerun()

    trajectory_dir = Path(trajectory_dir)
    if not trajectory_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {trajectory_dir}")

    npz_files = sorted(trajectory_dir.rglob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {trajectory_dir}")
        return []

    results: list[Path] = []
    for npz_file in npz_files:
        print(f"Processing: {npz_file}")
        if output_dir is not None:
            # Mirror subdirectory structure under output_dir
            rel = npz_file.relative_to(trajectory_dir)
            out_path = Path(output_dir) / rel.with_suffix(".rrd")
        else:
            out_path = None

        rrd_path = generate_rrd(
            npz_file,
            output_path=out_path,
            camera_decimation=camera_decimation,
        )
        results.append(rrd_path)
        print(f"  -> {rrd_path}")

    print(f"\nGenerated {len(results)} .rrd file(s)")
    return results

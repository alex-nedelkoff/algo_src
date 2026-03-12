"""Convert .npz trajectory files to rerun .rrd archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sim.viz.pinhole import PinholeCamera, quat_to_rotation_matrix, render_wireframe, render_horizon

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
    if schema_version != 1:
        raise ValueError(
            f"Unsupported trajectory schema version {schema_version} "
            f"(expected 1). File: {npz_path}"
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
    for g in range(gate_positions.shape[0]):
        corners = _gate_corners(
            gate_positions[g], gate_orientations[g], gate_half_extents[g]
        )
        # Close the loop: append first corner to end
        loop = np.vstack([corners, corners[0:1]])  # (5, 3)
        color = _GATE_COLORS[g % len(_GATE_COLORS)]
        rr.log(
            f"world/gates/gate_{g}",
            rr.LineStrips3D([loop], colors=[color]),
            static=True,
        )

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

        # Drone bounding box
        rr.log(
            "world/drone/box",
            rr.Boxes3D(half_sizes=[[0.1, 0.1, 0.03]]),
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

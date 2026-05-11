"""Open3D viewer for the playroom asset bundle.

Shows the transformed mesh + each gate visualized as a colored torus
ring (oriented to match the gate's pose) + the world origin coordinate
frame. Optionally overlays a drone trajectory saved by
``scripts/_playroom_eval_smoke.py`` (as a green-to-red line strip, plus
spawn/end markers). Run from a LOCAL terminal on the laptop.

Drag = rotate, scroll = zoom, Shift+drag = pan.

Usage on laptop::

    cd C:\\Users\\alexj\\Documents\\algo_src\\.claude\\worktrees\\warehouse-tsdf-pybullet-mvp
    python -m scripts.perception._phase2_view_playroom
    # or with trajectory overlay:
    python -m scripts.perception._phase2_view_playroom --trajectory outputs/playroom_eval_traj.npz
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
ASSET_DIR = REPO / "sim" / "assets" / "playroom_v1"
WORLD_OFFSET_Z = 5.475  # match the value the env uses


def quat_to_R(wxyz: np.ndarray) -> np.ndarray:
    """[w, x, y, z] quaternion -> 3x3 rotation matrix."""
    w, x, y, z = wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", type=Path, default=None,
                    help="Path to a .npz saved by _playroom_eval_smoke.py")
    args = ap.parse_args()

    import open3d as o3d

    # ---- Mesh ----------------------------------------------------------
    obj = ASSET_DIR / "warehouse.obj"
    print(f"Loading mesh: {obj}")
    mesh = o3d.io.read_triangle_mesh(str(obj))
    mesh.compute_vertex_normals()
    # Translate mesh to match what WarehouseRaceEnv does at load time
    mesh.translate((0.0, 0.0, WORLD_OFFSET_Z))
    print(f"  {len(mesh.triangles):,} triangles after translate "
          f"(floor now at z=0)")

    geoms = [mesh]

    # ---- World axes at origin (z-up ENU) -------------------------------
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geoms.append(axes)

    # ---- Gates ---------------------------------------------------------
    gates_path = ASSET_DIR / "gates_enu.json"
    print(f"Loading gates: {gates_path}")
    gates = json.loads(gates_path.read_text())
    # IMPORTANT: env loads with reorient_to_racing_line=true, which OVERRIDES
    # each gate's authored orientation with a yaw-only quaternion pointing
    # toward the next gate (closed loop). We reproduce that here so the
    # visualization matches what the policy actually sees.
    gate_names = list(gates.keys())
    gate_positions = np.array([gates[n]["position_enu"] for n in gate_names])
    n_gates = len(gate_names)
    reoriented_quats = []
    for i in range(n_gates):
        nxt = gate_positions[(i + 1) % n_gates]
        dx = float(nxt[0] - gate_positions[i, 0])
        dy = float(nxt[1] - gate_positions[i, 1])
        yaw = np.arctan2(dy, dx)
        reoriented_quats.append(np.array([
            np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2),
        ]))
    colors = [
        (1.0, 0.2, 0.2),   # red    Gate_00
        (0.2, 1.0, 0.2),   # green  Gate_01
        (0.2, 0.4, 1.0),   # blue   Gate_02
        (1.0, 1.0, 0.2),   # yellow Gate_03
        (1.0, 0.5, 0.0),   # orange Gate_04
        (0.7, 0.2, 1.0),   # purple Gate_05
    ]
    for i, (name, g) in enumerate(gates.items()):
        pos = np.array(g["position_enu"]) + np.array([0.0, 0.0, WORLD_OFFSET_Z])
        quat = reoriented_quats[i]   # env-actual orientation
        R = quat_to_R(quat)
        r_inner = g["inner_radius_m"]
        r_outer = g["outer_radius_m"]
        # Torus with major = inner radius, tube = (outer - inner) / 2
        torus = o3d.geometry.TriangleMesh.create_torus(
            torus_radius=r_inner, tube_radius=(r_outer - r_inner) / 2.0,
            radial_resolution=32, tubular_resolution=16,
        )
        # Gate's local frame: x = forward (drone passes through +x), z = up.
        # create_torus produces a ring in the XY plane (normal = +z). We
        # want the ring's plane perpendicular to gate-forward (i.e. ring
        # normal = gate's +x). So rotate the torus's +z to align with the
        # gate's +x by a -90 deg rotation about the Y axis, *then* apply
        # the gate's full rotation.
        Rz_to_x = np.array([
            [0,  0, 1],
            [0,  1, 0],
            [-1, 0, 0],
        ])
        torus.rotate(R @ Rz_to_x, center=(0, 0, 0))
        torus.translate(pos)
        torus.paint_uniform_color(colors[i % len(colors)])
        torus.compute_vertex_normals()
        geoms.append(torus)
        print(f"  {name}: pos={pos.round(2)} color={colors[i % len(colors)]}")

        # Also add a small arrow at the gate showing forward direction
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.05, cone_radius=0.12,
            cylinder_height=0.6, cone_height=0.2,
        )
        # Arrow default points +z; rotate so it points +x of gate frame
        arrow.rotate(R @ Rz_to_x, center=(0, 0, 0))
        arrow.translate(pos)
        arrow.paint_uniform_color(colors[i % len(colors)])
        geoms.append(arrow)

    # ---- Trajectory (optional) -----------------------------------------
    if args.trajectory is not None and args.trajectory.exists():
        print(f"\nLoading trajectory: {args.trajectory}")
        data = np.load(args.trajectory)
        positions = data["positions"]  # (N, 3), already in world frame WITH offset
        # The eval saves env._states[0, 0:3] which is in WORLD coords (world
        # offset already baked in by GateRaceEnv.reset/step). Do NOT add
        # WORLD_OFFSET_Z again — it's already there.
        n = len(positions)
        print(f"  {n} positions, "
              f"reward sum={float(data['total_reward']):.2f}, "
              f"gates_passed={int(data['gates_passed'])}, "
              f"termination_reason={int(data['termination_reason'])}")

        # Line strip colored by time: green at start -> red at end
        lines = [[i, i + 1] for i in range(n - 1)]
        cmap = np.zeros((n - 1, 3))
        t = np.linspace(0, 1, n - 1)
        cmap[:, 0] = t        # R grows
        cmap[:, 1] = 1 - t    # G shrinks
        cmap[:, 2] = 0.2
        traj = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines),
        )
        traj.colors = o3d.utility.Vector3dVector(cmap)
        geoms.append(traj)

        # Spawn marker (blue) + end marker (red)
        spawn = data["spawn"]
        # Spawn is stored as world-frame (the script passes spawn_pos in
        # world coords pre-offset; reset() then applies offset). Let's
        # use positions[0] which is the post-reset position to avoid
        # offset confusion.
        spawn_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        spawn_sphere.translate(positions[0])
        spawn_sphere.paint_uniform_color((0.2, 0.4, 1.0))   # blue
        spawn_sphere.compute_vertex_normals()
        geoms.append(spawn_sphere)

        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        end_sphere.translate(positions[-1])
        end_sphere.paint_uniform_color((1.0, 0.0, 0.0))     # red (crash/end)
        end_sphere.compute_vertex_normals()
        geoms.append(end_sphere)

        # Gate-pass markers (yellow spheres at the exact pass positions)
        gp_pos = data["gate_pass_positions"]
        for p in gp_pos:
            m = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
            m.translate(p)
            m.paint_uniform_color((1.0, 1.0, 0.2))           # yellow
            m.compute_vertex_normals()
            geoms.append(m)
        print(f"  added {n}-point line strip + spawn (blue) + end (red) + "
              f"{len(gp_pos)} gate-pass markers (yellow)")

    print(f"\nOpening Open3D viewer "
          f"({len(geoms)} geometries: 1 mesh + 1 axes + {len(gates)} gates "
          f"+ {len(gates)} arrows{' + trajectory' if args.trajectory else ''})")
    o3d.visualization.draw_geometries(
        geoms, window_name="playroom + gates (ENU, Z-up)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

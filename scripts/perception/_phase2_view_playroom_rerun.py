"""Rerun version of the playroom viewer.

Logs the playroom mesh + 4 gates + drone trajectory + reward time-series
into a .rrd file. Two ways to view afterward:

  1. scp the .rrd to a machine with display, open with `rerun <path>.rrd`
  2. On the laptop:
       rerun outputs/playroom_eval.rrd --web-viewer --bind 0.0.0.0
     Then from Mac browser via Tailscale:  http://laptop-tc658s39:9876

The --bind 0.0.0.0 is important — without it rerun binds to localhost
only and Tailscale can't reach it.

Usage::

    python -m scripts.perception._phase2_view_playroom_rerun \\
        --trajectory outputs/playroom_eval_traj.npz \\
        --out outputs/playroom_eval.rrd
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
ASSET_DIR = REPO / "sim" / "assets" / "playroom_v1"
WORLD_OFFSET_Z = 5.475


def quat_to_R(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", type=Path, default=None,
                    help="Path to a .npz saved by _playroom_eval_smoke.py")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/playroom_eval.rrd"),
                    help="Output .rrd file")
    args = ap.parse_args()

    rr.init("playroom_racing", spawn=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(args.out))

    # Right-hand Z-up (matches our ENU world frame)
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ---- Mesh ----------------------------------------------------------
    obj = ASSET_DIR / "warehouse.obj"
    print(f"[mesh] loading {obj}")
    mesh = o3d.io.read_triangle_mesh(str(obj))
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices) + np.array([0.0, 0.0, WORLD_OFFSET_Z])
    tris = np.asarray(mesh.triangles)
    print(f"       {len(verts)} verts, {len(tris)} tris -> rerun.Mesh3D")
    rr.log("/world/playroom", rr.Mesh3D(
        vertex_positions=verts,
        triangle_indices=tris,
        vertex_normals=np.asarray(mesh.vertex_normals),
        albedo_factor=[180, 180, 180],
    ), static=True)

    # ---- Gates (env-actual reoriented orientations) --------------------
    gates_path = ASSET_DIR / "gates_enu.json"
    gates = json.loads(gates_path.read_text())
    gate_names = list(gates.keys())
    gate_positions_raw = np.array([gates[n]["position_enu"] for n in gate_names])
    n_gates = len(gate_names)

    colors = np.array([
        [255,  50,  50],   # red
        [ 50, 255,  50],   # green
        [ 50, 100, 255],   # blue
        [255, 255,  60],   # yellow
        [255, 130,   0],   # orange
        [180,  60, 255],   # purple
    ], dtype=np.uint8)

    for i in range(n_gates):
        pos = gate_positions_raw[i] + np.array([0, 0, WORLD_OFFSET_Z])
        nxt = gate_positions_raw[(i + 1) % n_gates]
        dx, dy = float(nxt[0] - gate_positions_raw[i, 0]), \
                 float(nxt[1] - gate_positions_raw[i, 1])
        yaw = np.arctan2(dy, dx)
        # Forward-pointing rotation (gate's +X = direction toward next gate)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        # Build a ring in the gate's local YZ plane (perpendicular to forward)
        r_inner = gates[gate_names[i]]["inner_radius_m"]
        angles = np.linspace(0, 2 * np.pi, 48)
        ring_local = np.stack([
            np.zeros_like(angles),
            np.cos(angles) * r_inner,
            np.sin(angles) * r_inner,
        ], axis=-1)
        ring_world = (ring_local @ R.T) + pos
        # Close the loop
        ring_world = np.vstack([ring_world, ring_world[0:1]])
        rr.log(f"/world/gates/{gate_names[i]}", rr.LineStrips3D(
            strips=[ring_world],
            colors=[colors[i % len(colors)].tolist()],
            radii=0.04,
        ), static=True)
        # Add a forward arrow as a separate line strip
        arrow_world = np.stack([pos, pos + R @ np.array([0.8, 0, 0])])
        rr.log(f"/world/gates/{gate_names[i]}_arrow", rr.LineStrips3D(
            strips=[arrow_world],
            colors=[colors[i % len(colors)].tolist()],
            radii=0.025,
        ), static=True)
        # Label at gate center
        rr.log(f"/world/gates/{gate_names[i]}_label", rr.Points3D(
            positions=[pos],
            colors=[colors[i % len(colors)].tolist()],
            radii=0.04,
            labels=[gate_names[i]],
        ), static=True)

    # ---- Trajectory ----------------------------------------------------
    if args.trajectory is not None and args.trajectory.exists():
        data = np.load(args.trajectory)
        positions = data["positions"]  # (N+1, 3)
        rewards = data["rewards"]      # (N,)
        n = len(positions)
        print(f"[traj] {n} steps  reward={float(data['total_reward']):.2f}  "
              f"gates_passed={int(data['gates_passed'])}  "
              f"term_reason={int(data['termination_reason'])}")

        # Whole path as a static line, colored green->red
        # Rerun LineStrips3D doesn't support per-vertex colors, so split into
        # short segments with progressive coloring instead.
        n_segments = min(200, n - 1)
        seg_size = max(1, (n - 1) // n_segments)
        for k in range(n_segments):
            i0 = k * seg_size
            i1 = min((k + 1) * seg_size + 1, n)
            t = k / max(1, n_segments - 1)
            r, g, b = int(255 * t), int(255 * (1 - t)), 60
            rr.log(f"/world/path/seg_{k:03d}", rr.LineStrips3D(
                strips=[positions[i0:i1]],
                colors=[[r, g, b]],
                radii=0.025,
            ), static=True)

        # Spawn (blue) + end (red) markers
        rr.log("/world/spawn", rr.Points3D(
            positions=[positions[0]],
            colors=[[60, 100, 255]], radii=0.18, labels=["spawn"],
        ), static=True)
        rr.log("/world/end", rr.Points3D(
            positions=[positions[-1]],
            colors=[[255,  20,  20]], radii=0.18,
            labels=[f"end (reason={int(data['termination_reason'])})"],
        ), static=True)

        # Gate-pass markers
        gp = data.get("gate_pass_positions") if hasattr(data, "get") else None
        try:
            gp = data["gate_pass_positions"]
            if len(gp) > 0:
                rr.log("/world/gate_passes", rr.Points3D(
                    positions=gp,
                    colors=[[255, 255, 60]] * len(gp),
                    radii=0.15,
                    labels=[f"pass {i}" for i in range(len(gp))],
                ), static=True)
        except (KeyError, ValueError):
            pass

        # Animated drone — per-step pose + scalar reward
        for t in range(n):
            rr.set_time("step", sequence=t)
            rr.log("/world/drone", rr.Points3D(
                positions=[positions[t]],
                colors=[[255, 200, 80]],
                radii=0.18,
            ))
            if t > 0 and t - 1 < len(rewards):
                rr.log("/scalars/reward", rr.Scalars(float(rewards[t - 1])))

    # ---- Blueprint -----------------------------------------------------
    rr.send_blueprint(rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(name="3D scene", origin="/world"),
            rrb.TimeSeriesView(name="Reward", origin="/scalars/reward"),
            row_shares=[4, 1],
        ),
    ), make_active=True, make_default=True)

    print(f"\n.rrd saved -> {args.out.resolve()}")
    print(f"View with one of:")
    print(f"  Mac (local):    rerun {args.out.name}  (after scp)")
    print(f"  Laptop (web):   rerun {args.out} --web-viewer --bind 0.0.0.0")
    print(f"                  open http://laptop-tc658s39:9876 from Mac via Tailscale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

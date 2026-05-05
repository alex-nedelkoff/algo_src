"""Week-2 vision-aware planner demo (M4 of COR-106).

Extends ``week1_demo.py`` with:
    - A 2.5D occupancy grid built by raycasting the warehouse mesh at
      drone altitude (M4.1).
    - A synthetic 1 m-radius cylinder dropped between gates 4 and 5
      (the off-axis ones from week 1) — the obstacle the planner has
      to detour around.
    - A* path planning between consecutive gates over the dilated
      occupancy (M4.2). The full-lap trajectory is the concatenation
      of these segments, time-parameterized at constant speed.
    - Top-down occupancy view + obstacle + planned path drawn in
      rerun, alongside the existing M1/M2/M3 overlays.

Compared to the week-1 demo the drone now actually flies a planned
path around an obstacle, instead of a parametric oval that ignores the
scene.

Usage::

    python -m scripts.perception.week2_planner_demo
    rerun outputs/perception/week2_planner_demo.rrd
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402  preload before transformers

import argparse  # noqa: E402
import io  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pybullet as pb  # noqa: E402
from PIL import Image  # noqa: E402

import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402

from perception.localization.synthetic_vio import (  # noqa: E402
    SyntheticVIO, _quat_yaw,
)
from perception.planning.cost_map_planner import plan as plan_path  # noqa: E402
from perception.planning.occupancy_grid import (  # noqa: E402
    GridExtent, OccupancyGrid2D5,
)
from sim.tracks.waypoint import waypoints_to_track  # noqa: E402
from scripts.perception.week1_demo import (  # noqa: E402
    WarehouseRenderer, quat_to_rotmat, rgb_to_jpeg_bytes,
    warehouse_centered_oval, build_off_axis_waypoints,
)


APP_ID = "vision_aug_racing_week2_planner_demo_v4"
DRONE_RADIUS_M = 0.25
OBSTACLE_RADIUS_M = 1.0
SPEED_MPS = 3.0


# --------------------------------------------------------------------- planner


def plan_full_lap(
    occupancy: OccupancyGrid2D5,
    gates_xyz: np.ndarray,
    target_spacing_m: float = 4.0,
) -> np.ndarray:
    """Plan A* paths between consecutive gates and concatenate into a lap.

    Args:
        occupancy: dilated grid (caller dilated by drone radius).
        gates_xyz: (N, 3) gate positions; we use xy and pin altitude.
        target_spacing_m: per-segment resampling target.

    Returns:
        (M, 3) waypoints walking the concatenated path.
    """
    n_gates = gates_xyz.shape[0]
    full = []
    for k in range(n_gates):
        a = gates_xyz[k, 0:2]
        b = gates_xyz[(k + 1) % n_gates, 0:2]
        wps = plan_path(
            occupancy, start_xy=a, goal_xy=b,
            altitude_m=float(gates_xyz[k, 2]),
            smooth=True, target_spacing_m=target_spacing_m,
        )
        if wps is None:
            print(f"  WARNING: planner failed for segment {k}->{(k+1)%n_gates}, "
                  f"falling back to direct line")
            # Direct line as fallback (shouldn't happen with the demo geometry).
            wps = np.stack([
                np.append(a, gates_xyz[k, 2]),
                np.append(b, gates_xyz[k, 2]),
            ], axis=0)
        # Drop the duplicate join cell between segments (keep starts, drop ends).
        if k > 0:
            wps = wps[1:]
        full.append(wps)
    return np.concatenate(full, axis=0)


def parameterize_constant_speed(
    waypoints: np.ndarray,
    speed_mps: float,
    dt: float,
) -> np.ndarray:
    """Time-parameterize a polyline at constant tangent speed.

    Returns (T, 7) poses [x, y, z, qw, qx, qy, qz] with yaw aligned to
    the path tangent. Linear in xyz between waypoints; yaw is computed
    on the fly from velocity (so corners produce sharp yaw transitions
    — good enough for visualization).
    """
    seg_vec = np.diff(waypoints, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = float(cum_len[-1])
    if total_len < 1e-9:
        # Degenerate; just hover.
        T = max(1, int(round(1.0 / dt)))
        out = np.zeros((T, 7), dtype=np.float64)
        out[:, 0:3] = waypoints[0]
        out[:, 3] = 1.0
        return out

    duration_s = total_len / speed_mps
    T = max(2, int(round(duration_s / dt)))
    target_s = np.linspace(0.0, total_len, T)

    out = np.zeros((T, 7), dtype=np.float64)
    seg_idx = np.searchsorted(cum_len, target_s, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_len) - 1)
    for i, (s, k) in enumerate(zip(target_s, seg_idx)):
        seg_start = cum_len[k]
        if seg_len[k] < 1e-12:
            t = 0.0
        else:
            t = (s - seg_start) / seg_len[k]
        out[i, 0:3] = waypoints[k] + t * seg_vec[k]
        # Yaw from tangent.
        v = seg_vec[k]
        yaw = math.atan2(float(v[1]), float(v[0]))
        out[i, 3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return out


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/perception/week2_planner_demo.rrd"))
    ap.add_argument("--dt", type=float, default=0.033)
    ap.add_argument("--cell-size-m", type=float, default=0.25)
    ap.add_argument("--fpv-step-skip", type=int, default=3)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- Scene + occupancy ---------------------------------------------------

    print("Loading warehouse...")
    renderer = WarehouseRenderer()
    aabb_min, aabb_max = renderer.aabb()
    altitude_m = float(aabb_min[2] + 1.5)
    print(f"  warehouse AABB: min {aabb_min.round(2).tolist()}  max {aabb_max.round(2).tolist()}")
    print(f"  flight altitude: {altitude_m:.2f} m")

    # ---- Gates first (we need them to know where to put the obstacle) -------

    print("Building gates from warehouse-centered oval (M3 reuse)...")
    seed_traj = warehouse_centered_oval(
        duration_s=30.0, dt=args.dt, aabb_min=aabb_min, aabb_max=aabb_max,
        speed_mps=SPEED_MPS,
    )
    gates_xyz = build_off_axis_waypoints(seed_traj, n_gates=12)
    print(f"  {gates_xyz.shape[0]} gates")

    # ---- Spawn a real cylinder body in PyBullet so it shows up in
    #      RGB / depth / raycast occupancy alike. The previous version
    #      only painted occupancy cells, so the obstacle was invisible
    #      in the rendered FPV view — misleading for verification.

    obstacle_xy = 0.5 * (gates_xyz[4, 0:2] + gates_xyz[5, 0:2])
    obstacle_height_m = 4.0
    obstacle_z_center = float(aabb_min[2]) + obstacle_height_m / 2.0
    print(f"Spawning cylinder body: center xy {obstacle_xy.round(2).tolist()}, "
          f"radius {OBSTACLE_RADIUS_M} m, height {obstacle_height_m} m")
    obstacle_body_id = renderer.spawn_cylinder(
        center_xy=obstacle_xy,
        radius_m=OBSTACLE_RADIUS_M,
        height_m=obstacle_height_m,
        z_center=obstacle_z_center,
        rgba=(0.85, 0.25, 0.25, 1.0),
    )

    # ---- Occupancy: raycast against warehouse + cylinder so the planner
    #      sees what the renderer renders.

    print("Building 2.5D occupancy grid by raycasting warehouse + obstacle...")
    extent = GridExtent(
        x_min=float(aabb_min[0]) - 1.0, x_max=float(aabb_max[0]) + 1.0,
        y_min=float(aabb_min[1]) - 1.0, y_max=float(aabb_max[1]) + 1.0,
        cell_size_m=args.cell_size_m,
    )
    print(f"  extent: {extent.n_cells_x} x {extent.n_cells_y} cells "
          f"@ {extent.cell_size_m} m → {extent.n_cells_x * extent.n_cells_y} rays")
    occ_raw = OccupancyGrid2D5.from_pybullet_scene(
        client_id=renderer.cid,
        body_ids=[renderer.handles.warehouse_body_id, obstacle_body_id],
        extent=extent, altitude_m=altitude_m, thickness_m=2.0,
    )
    print(f"  raw occupancy: {int(occ_raw.grid.sum())} cells "
          f"({100 * occ_raw.grid.sum() / occ_raw.grid.size:.1f} %)")

    # Dilate by drone radius.
    occ = occ_raw.dilate(DRONE_RADIUS_M)
    print(f"  after dilate ({DRONE_RADIUS_M} m drone radius): "
          f"{int(occ.grid.sum())} cells occupied")

    # ---- Plan + parameterize -------------------------------------------------

    print("Planning A* paths between consecutive gates...")
    full_path = plan_full_lap(occ, gates_xyz, target_spacing_m=4.0)
    print(f"  {full_path.shape[0]} waypoints over the full lap")

    print(f"Time-parameterizing at {SPEED_MPS} m/s...")
    gt_poses = parameterize_constant_speed(full_path, speed_mps=SPEED_MPS, dt=args.dt)
    T = gt_poses.shape[0]
    duration_s = T * args.dt
    print(f"  {T} steps @ {args.dt} s = {duration_s:.1f} s of flight")

    # ---- VIO shadows ---------------------------------------------------------

    print("Running synthetic VIO across 3 profiles (M1)...")
    profiles = {
        "with_map": ("orb_slam3_with_map_matching", "#7ee787"),
        "no_map": ("orb_slam3_mono_inertial", "#ffa657"),
        "dead_reckoning": ("imu_dead_reckoning", "#ff7b72"),
    }
    vio_estimates = {}
    for name, (profile_name, _) in profiles.items():
        vio = SyntheticVIO(profile=profile_name, seed=0)
        vio_estimates[name] = vio.run(gt_poses, dt=args.dt)
        final = float(np.linalg.norm(vio_estimates[name][-1, 0:3] - gt_poses[-1, 0:3]))
        print(f"  {name:>15s}: final drift {final:.3f} m")

    # ---- DA V2 pipeline ------------------------------------------------------

    print("Loading DA V2 Small for per-frame depth (M2)...")
    from transformers import pipeline
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # ---- Logging -------------------------------------------------------------

    print(f"\nWriting {args.out}...")
    rr.init(APP_ID, spawn=False)
    rr.save(str(args.out))

    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Log occupancy as a Boxes3D — one box per occupied cell, lying flat
    # at altitude. Subtle alpha to keep the 3D scene readable.
    print("Logging occupancy grid + obstacle + planned path...")
    occ_indices = np.argwhere(occ_raw.grid != 0)
    cs = occ_raw.extent.cell_size_m
    if occ_indices.size > 0:
        centers = []
        for i, j in occ_indices:
            xy = occ_raw.grid_to_world((int(i), int(j)))
            centers.append([float(xy[0]), float(xy[1]), altitude_m - 0.05])
        rr.log(
            "/world/occupancy",
            rr.Boxes3D(
                centers=centers,
                half_sizes=[[cs / 2, cs / 2, 0.025]] * len(centers),
                colors=[(80, 100, 130)] * len(centers),
            ),
            static=True,
        )

    # Cylinder obstacle: now a real PyBullet body, so we draw it as a
    # solid 3D representation (stack of rings spanning the height) — what
    # the rerun view shows is the same physical thing the renderer sees.
    n_ring_pts = 32
    obstacle_z_min = float(aabb_min[2])
    obstacle_z_max = obstacle_z_min + obstacle_height_m
    z_levels = np.linspace(obstacle_z_min, obstacle_z_max, 5)
    rings = []
    for z in z_levels:
        ring = []
        for k in range(n_ring_pts + 1):
            ang = 2 * math.pi * k / n_ring_pts
            ring.append([
                float(obstacle_xy[0]) + OBSTACLE_RADIUS_M * math.cos(ang),
                float(obstacle_xy[1]) + OBSTACLE_RADIUS_M * math.sin(ang),
                float(z),
            ])
        rings.append(ring)
    rr.log(
        "/world/obstacle",
        rr.LineStrips3D(
            strips=rings,
            colors=[(255, 80, 80)] * len(rings),
            radii=0.04,
            labels=[f"{OBSTACLE_RADIUS_M} m × {obstacle_height_m} m cylinder"] + [""] * (len(rings) - 1),
        ),
        static=True,
    )

    # Synthetic gates (M3) — same as week 1.
    GATE_HALF = 0.75
    frame_local = np.array([
        [0.0,  GATE_HALF,  GATE_HALF],
        [0.0, -GATE_HALF,  GATE_HALF],
        [0.0, -GATE_HALF, -GATE_HALF],
        [0.0,  GATE_HALF, -GATE_HALF],
        [0.0,  GATE_HALF,  GATE_HALF],
    ])
    track = waypoints_to_track(
        gates_xyz, yaw_lookahead=2, yaw_smoothing_alpha=0.5, closed_loop=True,
    )
    for i, gate in enumerate(track.gates):
        rotmat = quat_to_rotmat(gate.orientation)
        is_off_axis = i in (4, 5)
        color = (255, 165, 100) if is_off_axis else (90, 160, 255)
        frame_world = (rotmat @ frame_local.T).T + gate.position
        rr.log(
            f"/world/track/gate_{i:02d}",
            rr.LineStrips3D(
                strips=[frame_world.tolist()],
                colors=[color], radii=0.025,
                labels=[f"wp{i}{' (off-axis)' if is_off_axis else ''}"],
            ),
            static=True,
        )

    # Planned path (the trajectory the drone follows).
    rr.log(
        "/world/planned_path",
        rr.LineStrips3D(
            strips=[full_path.tolist()],
            colors=[(140, 230, 160)], radii=0.04,
        ),
        static=True,
    )

    # FPV camera (Pinhole + body→cam transform — same as week 1).
    fpv_w, fpv_h = 512, 384
    rr.log(
        "/world/drone/fpv",
        rr.Pinhole(
            focal_length=fpv_w / (2 * math.tan(math.radians(70) / 2)),
            width=fpv_w, height=fpv_h,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
        static=True,
    )
    R_body_to_cam = np.array([
        [0.0,  0.0,  1.0],
        [-1.0, 0.0,  0.0],
        [0.0, -1.0,  0.0],
    ])
    rr.log(
        "/world/drone/fpv",
        rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=R_body_to_cam),
        static=True,
    )

    # ---- Time-varying logs ---------------------------------------------------

    print("Rendering + DA V2 inference per FPV frame...")
    fpv_count = 0
    t0 = time.perf_counter()

    for t in range(T):
        sim_time_s = t * args.dt
        rr.set_time("sim_time", duration=sim_time_s)

        gt_pos = gt_poses[t, 0:3]
        gt_q = gt_poses[t, 3:7]
        gt_yaw = _quat_yaw(gt_q)
        gt_rotmat = quat_to_rotmat(gt_q)
        rr.log("/world/drone", rr.Transform3D(translation=gt_pos.tolist(), mat3x3=gt_rotmat))
        rr.log(
            "/world/drone/body",
            rr.Points3D(positions=[[0, 0, 0]], colors=[(140, 220, 250)], radii=0.18),
        )
        rr.log(
            "/world/drone/axes",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 3,
                vectors=[[0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]],
                colors=[(255, 100, 100), (100, 255, 100), (100, 150, 255)],
                radii=0.02,
            ),
        )

        for name, (_, hex_color) in profiles.items():
            est = vio_estimates[name][t]
            color_rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
            rr.log(
                f"/world/vio_{name}",
                rr.Points3D(positions=[est[0:3].tolist()], colors=[color_rgb], radii=0.10),
            )
            drift = float(np.linalg.norm(est[0:3] - gt_pos))
            rr.log(f"/scalars/pos_drift_m/{name}", rr.Scalars(drift))
            yaw_err = abs(_quat_yaw(est[3:7]) - gt_yaw)
            yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
            rr.log(f"/scalars/yaw_drift_deg/{name}",
                   rr.Scalars(math.degrees(abs(yaw_err))))

        if t % args.fpv_step_skip == 0:
            rgb = renderer.render_from_drone(gt_pos, gt_yaw)
            jpeg_bytes = rgb_to_jpeg_bytes(rgb, quality=78)
            depth_out = depth_pipe(Image.fromarray(rgb))
            depth_arr = np.asarray(depth_out["depth"], dtype=np.float32)

            rr.log(
                "/world/drone/fpv/rgb",
                rr.EncodedImage(contents=jpeg_bytes, media_type="image/jpeg"),
            )
            rr.log("/world/drone/fpv/depth", rr.DepthImage(depth_arr, meter=1.0))
            fpv_count += 1
            if fpv_count % 30 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  FPV frame {fpv_count}: {elapsed:.1f}s elapsed "
                      f"({fpv_count / elapsed:.1f} fps)")

    total = time.perf_counter() - t0
    print(f"  done — {fpv_count} FPV frames in {total:.1f}s "
          f"({fpv_count / total:.1f} fps)")
    renderer.close()
    del depth_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Blueprint -----------------------------------------------------------

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="World — track + obstacle + planned path",
                    origin="/world",
                    contents=["/world/**", "/scalars/**:exclude"],
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="FPV (RGB)",
                        origin="/world/drone/fpv",
                        contents=["/world/drone/fpv/rgb"],
                    ),
                    rrb.Spatial2DView(
                        name="DA V2 depth",
                        origin="/world/drone/fpv",
                        contents=["/world/drone/fpv/depth"],
                    ),
                ),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Position drift (m) — VIO profiles vs GT",
                    origin="/scalars/pos_drift_m",
                ),
                rrb.TimeSeriesView(
                    name="Yaw drift (deg)",
                    origin="/scalars/yaw_drift_deg",
                ),
            ),
            row_shares=[3, 2],
        ),
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    print(f"\n  Demo .rrd → {args.out.resolve()}")
    print(f"  Open with: rerun {args.out}")
    print()
    print("What's new vs week 1:")
    print(f"  - Red ring at gates 4-5 = {OBSTACLE_RADIUS_M} m cylinder obstacle")
    print(f"  - Grey patches = warehouse occupancy at altitude (raycast on mesh)")
    print(f"  - Green line = A*-planned path between consecutive gates")
    print(f"  - Drone follows the planned path (detours around the obstacle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

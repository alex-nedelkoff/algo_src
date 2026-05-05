"""Week-1 verification demo (COR-106).

Builds a single rerun .rrd that renders all three week-1 milestones in
one scrubbable timeline:

    M3 — WaypointTrack adapter
        Synthetic gates rendered as 3D rectangles at each waypoint.
        Off-axis waypoint placement is visible — demonstrates that the
        adapter correctly handles non-trivial geometry.

    M1 — Localization drift
        Three "shadow drones" follow the ground-truth trajectory but
        accumulate drift per the synthetic VIO profiles. The augmentation-
        case (with map matching) sticks tight to GT; the no-map case
        wanders; dead reckoning floats away. Drift over time is in the
        scalar panel.

    M2 — Depth source
        DA V2 Small inference on warehouse renders, shown alongside RGB
        in the FPV camera view. RGB cycles through the 3 captured renders
        every 5 s. Demonstrates the model produces sensible depth.

Scrub the timeline to verify behavior. Rotate the 3D view to inspect
the trajectory + waypoints. The 2D FPV view stays locked to the
drone's heading (per ``reference_rerun_drone_fpv.md`` conventions).

Usage::

    python -m scripts.perception.week1_demo
    # Then open outputs/perception/week1_demo.rrd in the rerun viewer
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402

import argparse  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402

from perception.localization.synthetic_vio import (  # noqa: E402
    SyntheticVIO, _quat_yaw,
)
from sim.tracks.waypoint import waypoints_to_track  # noqa: E402
from scripts.perception.benchmark_localization_drift import (  # noqa: E402
    parametric_oval_trajectory,
)


# Conventions: world frame is ENU (X right, Y forward, Z up). Drone body is
# FLU (X forward, Y left, Z up). See reference_rerun_drone_fpv.md for the
# camera-attachment specifics that work in this codebase.
APP_ID = "vision_aug_racing_week1_demo_v3"  # bump on blueprint changes
WAREHOUSE_RENDERS = [
    "outputs/render/warehouse_top.png",
    "outputs/render/warehouse_perspective.png",
    "outputs/render/warehouse_inside.png",
]


def build_off_axis_waypoints(gt_poses: np.ndarray, n_gates: int = 12) -> np.ndarray:
    """Sample N waypoints from the oval, with deliberate off-axis excursions.

    The first 3 waypoints sit on the oval (clean racing line). Waypoints
    4-5 are pushed laterally outward by 1.5 m — the synthetic detour
    that the M3 adapter must represent. The remaining waypoints return
    to the oval. Visualizes that off-axis waypoints are recognised.
    """
    T = gt_poses.shape[0]
    sample_indices = np.linspace(0, T - 1, n_gates, dtype=int)
    waypoints = gt_poses[sample_indices, 0:3].copy()
    # Push waypoints 4 + 5 outward (a synthetic "detour around an obstacle").
    for k in (4, 5):
        # Outward radial direction in the XY plane.
        r = waypoints[k, 0:2]
        if np.linalg.norm(r) > 1e-6:
            r_unit = r / np.linalg.norm(r)
            waypoints[k, 0:2] += 1.5 * r_unit
    return waypoints


def run_da_v2_on_renders(image_paths: list[str], resolution=(384, 512)) -> tuple[list[bytes], list[np.ndarray]]:
    """Compute DA V2 Small depth for each render.

    Returns ``(jpeg_bytes_per_render, depth_arrays)``. We pre-encode RGB
    as JPEG so per-frame logging fits in a few MB instead of 150+ —
    rerun stores image data once per (entity, time) pair and doesn't
    dedupe identical blobs across timestamps, so shrinking each blob
    is the right lever.

    Resolution matches the M2 decision (Small @ 384x512).
    """
    import io

    from transformers import pipeline

    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    rgb_jpegs = []
    depths = []
    h, w = resolution
    for p in image_paths:
        img = Image.open(p).convert("RGB").resize((w, h), Image.BILINEAR)
        out = pipe(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        rgb_jpegs.append(buf.getvalue())
        depths.append(np.asarray(out["depth"], dtype=np.float32))
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rgb_jpegs, depths


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/perception/week1_demo.rrd"))
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=0.033)  # ~30 Hz log rate
    ap.add_argument("--rgb-cycle-s", type=float, default=5.0,
                    help="how often to switch RGB+depth view among the 3 renders")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Generating ground-truth trajectory...")
    gt_poses, dt = parametric_oval_trajectory(
        duration_s=args.duration_s, dt=args.dt,
    )
    T = gt_poses.shape[0]
    print(f"  {T} steps @ {dt}s ({args.duration_s}s)")

    print("Building waypoints + WaypointTrack (M3)...")
    waypoints = build_off_axis_waypoints(gt_poses, n_gates=12)
    track = waypoints_to_track(
        waypoints, yaw_lookahead=2, yaw_smoothing_alpha=0.5, closed_loop=True,
    )
    print(f"  {track.num_gates} synthetic gates (waypoints 4-5 pushed off-axis)")

    print("Running synthetic VIO across 3 profiles (M1)...")
    profiles = {
        "with_map": ("orb_slam3_with_map_matching", "#7ee787"),
        "no_map": ("orb_slam3_mono_inertial", "#ffa657"),
        "dead_reckoning": ("imu_dead_reckoning", "#ff7b72"),
    }
    vio_estimates = {}
    for name, (profile_name, _) in profiles.items():
        vio = SyntheticVIO(profile=profile_name, seed=0)
        vio_estimates[name] = vio.run(gt_poses, dt=dt)
        final_drift = float(np.linalg.norm(
            vio_estimates[name][-1, 0:3] - gt_poses[-1, 0:3]
        ))
        print(f"  {name:>15s}: final drift {final_drift:.3f} m")

    print("Running DA V2 Small on warehouse renders (M2)...")
    rgb_jpegs, depths = run_da_v2_on_renders(WAREHOUSE_RENDERS, resolution=(384, 512))
    print(f"  {len(rgb_jpegs)} (rgb-jpeg, depth) pairs cached at 384x512")

    print(f"\nWriting {args.out}...")
    rr.init(APP_ID, spawn=False)
    rr.save(str(args.out))

    # ---- Static logs (independent of time) -----------------------------------

    # World coordinate axes for sanity reference.
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Synthetic gates (M3): rectangular frame outlines (openings, not filled
    # walls) so the drone visibly flies *through* the opening rather than
    # clipping into a solid face. 1.5 m square in the body Y-Z plane (the
    # gate's "front" is its local +X axis; the opening is perpendicular to it).
    GATE_HALF = 0.75   # half-side of the 1.5 m opening
    # Frame corners in gate-local frame (X = forward / passage axis, Y = lateral,
    # Z = vertical). Frame lies in the Y-Z plane at X=0.
    frame_local = np.array([
        [0.0,  GATE_HALF,  GATE_HALF],
        [0.0, -GATE_HALF,  GATE_HALF],
        [0.0, -GATE_HALF, -GATE_HALF],
        [0.0,  GATE_HALF, -GATE_HALF],
        [0.0,  GATE_HALF,  GATE_HALF],   # close the loop
    ])
    for i, gate in enumerate(track.gates):
        rotmat = quat_to_rotmat(gate.orientation)
        # Off-axis waypoints get a different colour so they stand out.
        is_off_axis = i in (4, 5)
        color = (255, 165, 100) if is_off_axis else (90, 160, 255)
        # Transform the local frame corners into world coordinates.
        frame_world = (rotmat @ frame_local.T).T + gate.position
        rr.log(
            f"/world/track/gate_{i:02d}",
            rr.LineStrips3D(
                strips=[frame_world.tolist()],
                colors=[color],
                radii=0.025,
                labels=[f"wp{i}{' (off-axis)' if is_off_axis else ''}"],
            ),
            static=True,
        )
        # Also log the gate's forward normal as a small arrow so the
        # passage direction is visually unambiguous.
        normal = rotmat @ np.array([0.5, 0.0, 0.0])
        rr.log(
            f"/world/track/gate_{i:02d}_normal",
            rr.Arrows3D(
                origins=[gate.position.tolist()],
                vectors=[normal.tolist()],
                colors=[color],
                radii=0.012,
            ),
            static=True,
        )

    # GT trajectory as a single static line for context.
    rr.log(
        "/world/trajectory_gt",
        rr.LineStrips3D(
            strips=[gt_poses[:, 0:3].tolist()],
            colors=[(140, 220, 250)],
            radii=0.02,
        ),
        static=True,
    )

    # Pinhole camera definition for the FPV view (rooted on the drone).
    fpv_w, fpv_h = 512, 384
    fpv_focal = 350.0   # ~70° FOV horizontally; just for visualisation
    rr.log(
        "/world/drone/fpv",
        rr.Pinhole(
            focal_length=fpv_focal,
            width=fpv_w,
            height=fpv_h,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
        static=True,
    )
    # Body-to-camera rotation per reference_rerun_drone_fpv.md
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

    rgb_steps_per_cycle = max(1, int(round(args.rgb_cycle_s / dt)))

    for t in range(T):
        sim_time_s = t * dt
        rr.set_time("sim_time", duration=sim_time_s)

        # GT drone pose: parent transform of /world/drone.
        gt_pos = gt_poses[t, 0:3]
        gt_q = gt_poses[t, 3:7]
        gt_rotmat = quat_to_rotmat(gt_q)
        rr.log(
            "/world/drone",
            rr.Transform3D(translation=gt_pos.tolist(), mat3x3=gt_rotmat),
        )
        rr.log(
            "/world/drone/body",
            rr.Points3D(
                positions=[[0, 0, 0]],
                colors=[(140, 220, 250)],
                radii=0.18,
            ),
        )
        # Body-frame axes for orientation sanity.
        rr.log(
            "/world/drone/axes",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 3,
                vectors=[[0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]],
                colors=[(255, 100, 100), (100, 255, 100), (100, 150, 255)],
                radii=0.02,
            ),
        )

        # VIO shadow drones — one Points3D per profile so they show up
        # even when collapsed.
        for name, (_, hex_color) in profiles.items():
            est = vio_estimates[name][t]
            est_pos = est[0:3]
            color_rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
            rr.log(
                f"/world/vio_{name}",
                rr.Points3D(
                    positions=[est_pos.tolist()],
                    colors=[color_rgb],
                    radii=0.10,
                ),
            )
            # Per-step drift scalar.
            drift = float(np.linalg.norm(est_pos - gt_pos))
            rr.log(f"/scalars/pos_drift_m/{name}", rr.Scalars(drift))

            # Yaw drift in degrees.
            yaw_err = abs(_quat_yaw(est[3:7]) - _quat_yaw(gt_q))
            yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
            rr.log(f"/scalars/yaw_drift_deg/{name}",
                   rr.Scalars(math.degrees(abs(yaw_err))))

        # RGB + depth cycle: switch every rgb_cycle_s seconds. Log every
        # 10th frame (≈3 Hz update rate) — enough that the viewer always
        # has a recent entity to render even during scrub jumps, while
        # keeping the float32 depth blobs from blowing up file size.
        if t % 10 == 0:
            rgb_idx = (t // rgb_steps_per_cycle) % len(rgb_jpegs)
            rr.log(
                "/world/drone/fpv/rgb",
                rr.EncodedImage(contents=rgb_jpegs[rgb_idx], media_type="image/jpeg"),
            )
            rr.log("/world/drone/fpv/depth", rr.DepthImage(depths[rgb_idx], meter=1.0))

    # ---- Blueprint: orbital 3D + FPV (locked) + scalars ---------------------

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="World — track + drone + VIO shadows",
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
    print("What to look for in the viewer:")
    print("  3D world view (top-left):")
    print("    - 12 synthetic gates from M3, two pushed laterally outward (orange)")
    print("    - GT drone (blue sphere + axes) flying the oval")
    print("    - Three VIO shadows (green=with-map, orange=no-map, red=dead-reckoning)")
    print("  FPV camera + depth (top-right): cycles through 3 warehouse renders, M2 DA V2")
    print("  Drift time series (bottom): green stays flat, orange grows, red explodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

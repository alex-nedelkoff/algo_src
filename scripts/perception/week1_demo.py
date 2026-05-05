"""Week-1 verification demo (COR-106).

Builds a single rerun .rrd that renders all three week-1 milestones in
one scrubbable timeline:

    M3 — WaypointTrack adapter
        Synthetic gates rendered as 3D rectangle outlines at each
        waypoint. Off-axis waypoint placement is visible — demonstrates
        that the adapter handles non-trivial geometry. Drone passes
        through openings cleanly (frames, not solid faces).

    M1 — Localization drift
        Three "shadow drones" follow the ground-truth trajectory but
        accumulate drift per the synthetic VIO profiles. The augmentation
        case (with map matching) sticks tight to GT; the no-map case
        wanders; dead reckoning floats away. Drift over time is in the
        scalar panel.

    M2 — Depth source
        DA V2 Small inference on warehouse views *rendered from the
        drone's actual pose at every step* via PyBullet's TINY
        renderer. RGB + depth update at 10 Hz throughout the run.

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
from sim.pybullet.warehouse_loader import WarehouseScene  # noqa: E402
from sim.tracks.waypoint import waypoints_to_track  # noqa: E402


# Conventions: world frame is ENU (X right, Y forward, Z up). Drone body is
# FLU (X forward, Y left, Z up). See reference_rerun_drone_fpv.md for the
# camera-attachment specifics that work in this codebase.
APP_ID = "vision_aug_racing_week1_demo_v6"  # bump on blueprint changes
WAREHOUSE_ASSETS = Path("sim/assets/warehouse_fab_v1")


# --------------------------------------------------------------------- traj


def warehouse_centered_oval(
    duration_s: float,
    dt: float,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    margin_m: float = 2.5,
    altitude_above_floor_m: float = 1.5,
    speed_mps: float = 3.0,
    altitude_amplitude_m: float = 0.3,
) -> np.ndarray:
    """Generate a parametric oval that stays inside the warehouse AABB.

    Returns:
        (T, 7) poses [x, y, z, qw, qx, qy, qz].
    """
    # Center in xy; altitude = floor + offset (z floor = aabb_min[2]).
    cx = 0.5 * (aabb_min[0] + aabb_max[0])
    cy = 0.5 * (aabb_min[1] + aabb_max[1])
    alt = aabb_min[2] + altitude_above_floor_m
    rx = 0.5 * (aabb_max[0] - aabb_min[0]) - margin_m
    ry = 0.5 * (aabb_max[1] - aabb_min[1]) - margin_m
    rx = max(rx, 1.0)
    ry = max(ry, 1.0)
    eff_radius = math.sqrt(0.5 * (rx ** 2 + ry ** 2))
    omega = speed_mps / eff_radius

    T = int(round(duration_s / dt))
    poses = np.zeros((T, 7), dtype=np.float64)
    for i in range(T):
        t = i * dt
        theta = omega * t
        x = cx + rx * math.cos(theta)
        y = cy + ry * math.sin(theta)
        z = alt + altitude_amplitude_m * math.sin(2 * math.pi * 0.3 * t)

        vx = -rx * omega * math.sin(theta)
        vy = ry * omega * math.cos(theta)
        yaw = math.atan2(vy, vx)
        # Yaw-only quaternion [w, x, y, z].
        poses[i, 0:3] = [x, y, z]
        poses[i, 3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return poses


# --------------------------------------------------------------------- waypts


def build_off_axis_waypoints(gt_poses: np.ndarray, n_gates: int = 12) -> np.ndarray:
    """Sample waypoints from the oval, with deliberate off-axis excursions
    on waypoints 4 and 5 (1.5 m radially outward from the trajectory's
    center axis). Visualises that off-axis waypoints are recognised by M3.
    """
    T = gt_poses.shape[0]
    sample_indices = np.linspace(0, T - 1, n_gates, dtype=int)
    waypoints = gt_poses[sample_indices, 0:3].copy()
    # Center of mass of the trajectory in xy (used as the radial origin).
    centroid = gt_poses[:, 0:2].mean(axis=0)
    for k in (4, 5):
        r = waypoints[k, 0:2] - centroid
        if np.linalg.norm(r) > 1e-6:
            r_unit = r / np.linalg.norm(r)
            waypoints[k, 0:2] += 1.5 * r_unit
    return waypoints


# --------------------------------------------------------------------- render


class WarehouseRenderer:
    """Headless PyBullet TINY renderer of the warehouse from a drone pose.

    The warehouse + ground are loaded once into a DIRECT-mode client; per
    frame, we just call ``getCameraImage`` with a fresh view matrix.
    """

    def __init__(
        self,
        assets: Path = WAREHOUSE_ASSETS,
        width: int = 512,
        height: int = 384,
        fov_deg: float = 70.0,
    ) -> None:
        self.cid = pb.connect(pb.DIRECT)
        self.scene = WarehouseScene(asset_dir=assets)
        self.handles = self.scene.load_into(self.cid)
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.proj = pb.computeProjectionMatrixFOV(
            fov=fov_deg, aspect=width / height, nearVal=0.1, farVal=100.0,
        )

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = pb.getAABB(self.handles.warehouse_body_id, physicsClientId=self.cid)
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    def render_from_drone(self, position: np.ndarray, yaw_rad: float) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB image looking forward from the drone."""
        # Look 5 m forward from the drone's position along its yaw direction.
        forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
        eye = position
        target = position + forward * 5.0
        up = np.array([0.0, 0.0, 1.0])
        view = pb.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        # TINY renderer is the only one that works headless on Windows
        # without EGL set up.
        _, _, rgba, _, _ = pb.getCameraImage(
            self.width, self.height,
            viewMatrix=view, projectionMatrix=self.proj,
            renderer=pb.ER_TINY_RENDERER,
            physicsClientId=self.cid,
        )
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
        return rgba[:, :, :3]

    def close(self) -> None:
        pb.disconnect(self.cid)


# --------------------------------------------------------------------- helpers


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rgb_to_jpeg_bytes(rgb: np.ndarray, quality: int = 80) -> bytes:
    """Encode an HxWx3 uint8 array to JPEG bytes."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/perception/week1_demo.rrd"))
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=0.033,
                    help="sim time step (3D + scalars logged here)")
    ap.add_argument("--fpv-step-skip", type=int, default=3,
                    help="render + DA V2 every N sim-steps (default 3 → 10 Hz)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading warehouse + computing trajectory...")
    renderer = WarehouseRenderer()
    aabb_min, aabb_max = renderer.aabb()
    print(f"  warehouse AABB: min {aabb_min.round(2).tolist()}  max {aabb_max.round(2).tolist()}")

    gt_poses = warehouse_centered_oval(
        duration_s=args.duration_s, dt=args.dt,
        aabb_min=aabb_min, aabb_max=aabb_max,
    )
    T = gt_poses.shape[0]
    print(f"  {T} steps @ {args.dt}s ({args.duration_s}s); FPV every {args.fpv_step_skip} steps")

    print("Building waypoints + WaypointTrack (M3)...")
    waypoints = build_off_axis_waypoints(gt_poses, n_gates=12)
    track = waypoints_to_track(
        waypoints, yaw_lookahead=2, yaw_smoothing_alpha=0.5, closed_loop=True,
    )
    print(f"  {track.num_gates} synthetic gates; waypoints 4-5 pushed off-axis (orange)")

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

    print("Loading DA V2 Small for per-frame depth (M2)...")
    from transformers import pipeline
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    print(f"\nWriting {args.out}...")
    rr.init(APP_ID, spawn=False)
    rr.save(str(args.out))

    # ---- Static scene ---------------------------------------------------------

    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Synthetic gates (M3): rectangle outlines + forward-normal arrows.
    GATE_HALF = 0.75
    frame_local = np.array([
        [0.0,  GATE_HALF,  GATE_HALF],
        [0.0, -GATE_HALF,  GATE_HALF],
        [0.0, -GATE_HALF, -GATE_HALF],
        [0.0,  GATE_HALF, -GATE_HALF],
        [0.0,  GATE_HALF,  GATE_HALF],
    ])
    for i, gate in enumerate(track.gates):
        rotmat = quat_to_rotmat(gate.orientation)
        is_off_axis = i in (4, 5)
        color = (255, 165, 100) if is_off_axis else (90, 160, 255)
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

    rr.log(
        "/world/trajectory_gt",
        rr.LineStrips3D(
            strips=[gt_poses[:, 0:3].tolist()],
            colors=[(140, 220, 250)],
            radii=0.02,
        ),
        static=True,
    )

    # FPV camera Pinhole + body-to-camera rotation (per ref doc).
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

    # ---- Per-step logs (3D + scalars at full rate, FPV at 10 Hz) -------------

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
        rr.log("/world/drone/body",
               rr.Points3D(positions=[[0, 0, 0]], colors=[(140, 220, 250)], radii=0.18))
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
                rr.Points3D(
                    positions=[est[0:3].tolist()], colors=[color_rgb], radii=0.10,
                ),
            )
            drift = float(np.linalg.norm(est[0:3] - gt_pos))
            rr.log(f"/scalars/pos_drift_m/{name}", rr.Scalars(drift))
            yaw_err = abs(_quat_yaw(est[3:7]) - gt_yaw)
            yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
            rr.log(f"/scalars/yaw_drift_deg/{name}",
                   rr.Scalars(math.degrees(abs(yaw_err))))

        # FPV: render warehouse from drone pose, run DA V2, log both.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Easy-mode diagnostic of the MASt3R-SLAM Phase 1 smoke test.

Same plumbing as ``mast3r_slam_smoketest.py`` but with three diagnostic
changes that isolate whether the localizers can track at all under
generous conditions, before we worry about racing dynamics:

  1. **Slow flight** (~1 m/s instead of 7.37 m/s). With the per-step
     correspondence search radius (5 cm here), motion-per-frame stays
     well within the matching threshold.

  2. **Configurable trajectory mode**: ``static`` (drone doesn't move,
     useful to confirm the localizer returns the anchor on a fixed
     view) | ``line`` (slow forward translation, no rotation) |
     ``oval`` (the original lap, just slowed down).

  3. **GT prior every frame for ICP** (no carry-forward of the refined
     pose). If ICP fails when handed the right answer every step,
     it's a structural issue, not hysteresis.

Render is also brighter (ambient=1.0, broader directional component)
so MASt3R-SLAM has actual feature contrast to work with.

Usage on the pod::

    python -m scripts.perception.mast3r_slam_smoketest_easy --mode oval
    # or static, or line
    bash scripts/cloud/run_smoke.sh   # serves rerun if you want web-viewer

Output: ``outputs/perception/mast3r_smoketest_easy_<mode>.rrd``
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import open3d as o3d
import pybullet as pb
import rerun as rr
import rerun.blueprint as rrb

from perception.localization.icp_localizer import (
    CameraIntrinsics, IcpLocalizer, sample_mesh_surface,
)
from perception.localization.mast3r_localizer import Mast3rLocalizer


APP_ID = "mast3r_slam_smoketest_easy_v1"
ROOM_DIR = Path("sim/assets/textured_room_v1")
ROOM_OBJ = ROOM_DIR / "room.obj"
ROOM_URDF = ROOM_DIR / "room.urdf"

WIDTH, HEIGHT, VFOV_DEG = 512, 384, 70.0
ALTITUDE = 1.5

R_BODY_TO_CAM = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0],
])


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def static_trajectory(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Drone parked at room centre, looking +x (toward metal_plate wall)."""
    pos = np.tile(np.array([0.0, 0.0, ALTITUDE]), (n_frames, 1))
    yaw = np.zeros(n_frames)
    return pos, yaw


def line_trajectory(
    n_frames: int, dt: float, speed_mps: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow forward translation along +x, no rotation. Drone starts at
    (-3, 0, 1.5) and walks toward the +x wall.
    """
    t = np.arange(n_frames) * dt
    x = -3.0 + speed_mps * t
    pos = np.stack([x, np.zeros(n_frames), np.full(n_frames, ALTITUDE)], axis=1)
    yaw = np.zeros(n_frames)
    return pos, yaw


def oval_trajectory(
    n_frames: int, dt: float,
    radius_x: float = 4.0, radius_y: float = 3.0,
    n_laps: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow oval flight — defaults to ~1 m/s mean speed (15 % of a lap
    in 3 s), well below the 7.37 m/s of the original racing smoke test.
    """
    t_total = n_frames * dt
    omega = 2 * math.pi * n_laps / t_total
    t = np.arange(n_frames) * dt
    theta = omega * t
    pos = np.stack([
        radius_x * np.cos(theta),
        radius_y * np.sin(theta),
        np.full(n_frames, ALTITUDE),
    ], axis=1)
    vx = -radius_x * np.sin(theta) * omega
    vy =  radius_y * np.cos(theta) * omega
    yaw = np.arctan2(vy, vx)
    return pos, yaw


def pose_to_matrix(pos: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = pos
    return T


# ---------------------------------------------------------------------------
# PyBullet rendering — brighter than the original smoke test
# ---------------------------------------------------------------------------

def setup_scene(use_egl: bool) -> int:
    cid = pb.connect(pb.DIRECT)
    if use_egl:
        plugin_id = pb.loadPlugin("eglRendererPlugin", physicsClientId=cid)
        if plugin_id < 0:
            print("[scene] EGL plugin failed; falling back to TINY_RENDERER")
        else:
            print(f"[scene] EGL plugin loaded (id={plugin_id})")
    pb.setAdditionalSearchPath(str(ROOM_DIR), physicsClientId=cid)
    pb.loadURDF(str(ROOM_URDF), basePosition=[0, 0, 0],
                useFixedBase=True, physicsClientId=cid)
    return cid


def render_rgb_depth(
    cid: int, pos: np.ndarray, yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    eye = pos.tolist()
    target = (pos + 5.0 * forward).tolist()
    up = [0.0, 0.0, 1.0]
    view = pb.computeViewMatrix(eye, target, up)
    proj = pb.computeProjectionMatrixFOV(
        VFOV_DEG, WIDTH / HEIGHT, 0.1, 100.0,
    )
    # Brighter lighting than the original. Ambient=1.0 + diffuse=0.7
    # gives ~75 % more luminance, which makes the metal/brick textures
    # visible in their actual colors instead of muddy brown.
    _, _, rgba, depth, _ = pb.getCameraImage(
        WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj,
        renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        lightAmbientCoeff=1.0,
        lightDiffuseCoeff=0.7,
        lightSpecularCoeff=0.05,
        lightDirection=[0, 0, -1],
        lightColor=[1.0, 1.0, 1.0],
        shadow=0, physicsClientId=cid,
    )
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[..., :3].copy()
    depth = np.asarray(depth, dtype=np.float32).reshape(HEIGHT, WIDTH).copy()
    return rgb, depth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "line", "oval"],
                    default="oval")
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.03,
                    help="time step (s). 100 frames @ 30 ms = 3 s run.")
    ap.add_argument("--icp-max-corr-m", type=float, default=0.05,
                    help="ICP correspondence distance — tight to force "
                         "matches against the *real* nearby surface")
    ap.add_argument("--reset-icp-prior", action="store_true", default=True,
                    help="Use GT-derived prior every frame instead of "
                         "carrying ICP refined pose forward (default ON)")
    ap.add_argument("--no-egl", action="store_true")
    ap.add_argument("--skip-mast3r", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = Path(f"outputs/perception/"
                        f"mast3r_smoketest_easy_{args.mode}.rrd")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Trajectory ---------------------------------------------------
    print(f"\n[1/5] Trajectory mode={args.mode}, "
          f"{args.n_frames} frames @ dt={args.dt}s "
          f"({args.n_frames * args.dt:.1f} s total)")
    if args.mode == "static":
        positions, yaws = static_trajectory(args.n_frames)
    elif args.mode == "line":
        positions, yaws = line_trajectory(args.n_frames, args.dt)
    else:
        positions, yaws = oval_trajectory(args.n_frames, args.dt)

    if args.n_frames >= 2:
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=1).mean() / args.dt
    else:
        speed = 0.0
    print(f"     mean speed: {speed:.2f} m/s")

    # ---- 2. Render -------------------------------------------------------
    print(f"\n[2/5] Rendering RGB+depth (EGL={not args.no_egl})...")
    cid = setup_scene(use_egl=not args.no_egl)
    rgb_buf = np.zeros((args.n_frames, HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth_buf = np.zeros((args.n_frames, HEIGHT, WIDTH), dtype=np.float32)
    t0 = time.perf_counter()
    for t in range(args.n_frames):
        rgb, depth = render_rgb_depth(cid, positions[t], yaws[t])
        rgb_buf[t] = rgb
        depth_buf[t] = depth
        if t == 0 or t == args.n_frames - 1:
            print(f"     t={t:>3}: RGB mean=[{rgb[..., 0].mean():.0f},"
                  f"{rgb[..., 1].mean():.0f},{rgb[..., 2].mean():.0f}] "
                  f"std={rgb.std():.1f}  "
                  f"depth zbuf=[{depth.min():.3f}, {depth.max():.3f}]")
    pb.disconnect(cid)
    print(f"     render took {time.perf_counter() - t0:.1f} s")

    # ---- 3. Map cloud (ICP) ---------------------------------------------
    print(f"\n[3/5] ICP map cloud...")
    map_pcd = sample_mesh_surface(ROOM_OBJ, n_points=30_000)
    print(f"     {len(map_pcd.points)} points, has_normals={map_pcd.has_normals()}")

    intr = CameraIntrinsics.from_vfov(VFOV_DEG, WIDTH, HEIGHT)
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]])
    gt_T = np.stack([pose_to_matrix(positions[t], yaws[t])
                     for t in range(args.n_frames)])

    # ---- 4a. ICP ---------------------------------------------------------
    print(f"\n[4a/5] ICP (max_corr={args.icp_max_corr_m*100:.0f} cm, "
          f"reset_prior={args.reset_icp_prior})...")
    icp_loc = IcpLocalizer(
        map_pcd=map_pcd, intrinsics=intr, R_body_to_cam=R_BODY_TO_CAM,
        max_iterations=50,
        max_correspondence_distance_m=args.icp_max_corr_m,
        min_inlier_fraction=0.30,
        body_cloud_voxel_size_m=0.05,
        depth_stride=2,
    )
    icp_T = np.zeros_like(gt_T)
    icp_fitness = np.zeros(args.n_frames)
    icp_T[0] = gt_T[0].copy()
    prior = gt_T[0].copy()
    t0 = time.perf_counter()
    for t in range(args.n_frames):
        # If --reset-icp-prior, every frame uses GT as init. Otherwise
        # carry refined pose forward like the original smoke test.
        if args.reset_icp_prior:
            prior = gt_T[t].copy()
        elif t > 0:
            vel = (positions[t] - positions[t - 1]) / args.dt
            prior_pos = prior[:3, 3] + vel * args.dt
            prior = gt_T[t].copy()  # GT yaw
            prior[:3, 3] = prior_pos
        res = icp_loc.localize(depth_buf[t], prior,
                               is_zbuffer=True, near=0.1, far=100.0)
        icp_T[t] = res.refined_T_world_body
        icp_fitness[t] = res.fitness
        err = np.linalg.norm(icp_T[t][:3, 3] - gt_T[t][:3, 3])
        # Verbose per-frame log so we can see exactly where it goes wrong.
        print(f"     t={t:>3}  fit={res.fitness:.3f}  rmse={res.inlier_rmse_m*100:5.1f} cm  "
              f"corr_n={res.correspondence_count:>5}  "
              f"pos_err={err*100:6.1f} cm  "
              f"GT={gt_T[t,:3,3].round(2)}  est={icp_T[t,:3,3].round(2)}")
        prior = icp_T[t].copy()
    print(f"     ICP took {time.perf_counter() - t0:.1f} s")

    # ---- 4b. MASt3R-SLAM -------------------------------------------------
    mast_T = gt_T.copy()
    mast_mode = ["skipped"] * args.n_frames
    if not args.skip_mast3r:
        print(f"\n[4b/5] MASt3R-SLAM...")
        mast_loc = Mast3rLocalizer(
            K=K, R_body_to_cam=R_BODY_TO_CAM, img_size_wh=(WIDTH, HEIGHT),
        )
        mast_loc.reset(gt_T[0])
        t0 = time.perf_counter()
        for t in range(args.n_frames):
            res = mast_loc.localize(rgb_buf[t], t=t)
            mast_T[t] = res.refined_T_world_body
            mast_mode[t] = res.mode
            err = np.linalg.norm(mast_T[t][:3, 3] - gt_T[t][:3, 3])
            print(f"     t={t:>3}  mode={res.mode:<14}  "
                  f"pos_err={err*100:6.1f} cm  "
                  f"GT={gt_T[t,:3,3].round(2)}  est={mast_T[t,:3,3].round(2)}")
        print(f"     MASt3R took {time.perf_counter() - t0:.1f} s")

    # ---- 5. Rerun --------------------------------------------------------
    print(f"\n[5/5] Logging to {args.out}")
    rr.init(APP_ID, spawn=False)
    rr.save(str(args.out))
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    map_arr = np.asarray(map_pcd.points)
    sub = np.random.default_rng(0).choice(
        len(map_arr), min(8000, len(map_arr)), replace=False)
    rr.log("/world/map_cloud",
           rr.Points3D(positions=map_arr[sub].tolist(),
                       colors=[(80, 90, 110)] * len(sub), radii=0.025),
           static=True)
    rr.log("/world/path/gt",
           rr.LineStrips3D(strips=[gt_T[:, :3, 3].tolist()],
                           colors=[(140, 220, 250)], radii=0.03,
                           labels=["GT"]),
           static=True)
    rr.log("/world/path/icp",
           rr.LineStrips3D(strips=[icp_T[:, :3, 3].tolist()],
                           colors=[(140, 230, 160)], radii=0.025,
                           labels=["ICP"]),
           static=True)
    if not args.skip_mast3r:
        rr.log("/world/path/mast3r",
               rr.LineStrips3D(strips=[mast_T[:, :3, 3].tolist()],
                               colors=[(255, 180, 90)], radii=0.025,
                               labels=["MASt3R"]),
               static=True)

    for t in range(args.n_frames):
        rr.set_time("sim_time", duration=t * args.dt)
        rr.log("/world/drone_gt",
               rr.Points3D(positions=[gt_T[t, :3, 3].tolist()],
                           colors=[(140, 220, 250)], radii=0.16))
        rr.log("/world/drone_icp",
               rr.Points3D(positions=[icp_T[t, :3, 3].tolist()],
                           colors=[(140, 230, 160)], radii=0.12))
        rr.log("/scalars/icp/pos_err_cm",
               rr.Scalars(np.linalg.norm(icp_T[t, :3, 3] - gt_T[t, :3, 3]) * 100))
        rr.log("/scalars/icp/fitness", rr.Scalars(float(icp_fitness[t])))
        if not args.skip_mast3r:
            rr.log("/world/drone_mast3r",
                   rr.Points3D(positions=[mast_T[t, :3, 3].tolist()],
                               colors=[(255, 180, 90)], radii=0.12))
            rr.log("/scalars/mast3r/pos_err_cm",
                   rr.Scalars(np.linalg.norm(mast_T[t, :3, 3] - gt_T[t, :3, 3]) * 100))
        if t % 5 == 0:
            rr.log("/camera/rgb", rr.Image(rgb_buf[t]))

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="World", origin="/world",
                                   contents=["/world/**", "/scalars/**:exclude"]),
                rrb.Spatial2DView(name="Camera RGB", origin="/camera"),
                column_shares=[3, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="ICP error (cm)",
                                   origin="/scalars/icp/pos_err_cm"),
                rrb.TimeSeriesView(name="MASt3R error (cm)",
                                   origin="/scalars/mast3r/pos_err_cm"),
                rrb.TimeSeriesView(name="ICP fitness",
                                   origin="/scalars/icp/fitness"),
            ),
            row_shares=[3, 1],
        ),
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    # ---- Summary ---------------------------------------------------------
    print(f"\n=== Summary (mode={args.mode}, speed={speed:.2f} m/s) ===")
    icp_err = np.linalg.norm(icp_T[:, :3, 3] - gt_T[:, :3, 3], axis=1)
    print(f"  ICP    : mean={icp_err.mean()*100:6.2f} cm  "
          f"max={icp_err.max()*100:6.2f} cm  "
          f"fit_mean={icp_fitness.mean():.3f}")
    if not args.skip_mast3r:
        m_err = np.linalg.norm(mast_T[:, :3, 3] - gt_T[:, :3, 3], axis=1)
        n_lost = sum(1 for m in mast_mode
                     if m in ("tracking_lost", "reloc_pending"))
        print(f"  MASt3R : mean={m_err.mean()*100:6.2f} cm  "
              f"max={m_err.max()*100:6.2f} cm  "
              f"lost={n_lost}/{args.n_frames}")
    print(f"\n.rrd → {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

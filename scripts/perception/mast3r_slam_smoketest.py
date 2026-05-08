"""Phase 1 smoke test: validate MASt3R-SLAM tracks PyBullet-rendered RGB.

Stand-alone — no GateRaceEnv, no policy. Just a scripted oval flight in
the textured room, rendered RGB + depth at 100 Hz, then both localizers
run on the cached frames. Comparison logged to rerun.

Cloud usage (RunPod RTX 3090, Linux + EGL)::

    python -m scripts.perception.mast3r_slam_smoketest \
        --out outputs/perception/mast3r_smoketest.rrd

    # In another terminal: serve the rerun viewer to a browser.
    rerun outputs/perception/mast3r_smoketest.rrd \
        --web-viewer --bind 0.0.0.0 --web-viewer-port 9876

Then port-forward 9876 (RunPod web UI exposes ports automatically)
and open the viewer in your browser.

Why no policy / closed-loop here: Phase 1 is "does MASt3R-SLAM track on
rendered RGB at all?" — a pure perception test. Closing the loop with
the policy is gated on the answer.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

# OMP duplicate guard — avoids libiomp/libomp clash on some Linux conda
# envs. Cheap insurance, harmless when unneeded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import open3d as o3d
import pybullet as pb
import rerun as rr
import rerun.blueprint as rrb

from perception.localization.icp_localizer import (
    CameraIntrinsics, IcpLocalizer, matrix_to_pose,
    sample_mesh_surface, world_pose_to_matrix,
)
from perception.localization.mast3r_localizer import Mast3rLocalizer


APP_ID = "mast3r_slam_smoketest_v1"

# Textured room asset (built by sim/assets/textured_room_v1/build_room.py).
ROOM_DIR = Path("sim/assets/textured_room_v1")
ROOM_OBJ = ROOM_DIR / "room.obj"
ROOM_URDF = ROOM_DIR / "room.urdf"

# Drone camera parameters — match what we did on the closed-loop demo
# so any wins here transfer across.
WIDTH, HEIGHT, VFOV_DEG = 512, 384, 70.0

# Body→cam rotation, verified empirically (see reference_rerun_drone_fpv).
R_BODY_TO_CAM = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0],
])


# ---------------------------------------------------------------------------
# Scripted trajectory
# ---------------------------------------------------------------------------

def oval_trajectory(
    n_frames: int, dt: float,
    radius_x: float = 4.0, radius_y: float = 3.0,
    altitude: float = 1.5, n_laps: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Position + yaw for a closed oval at constant tangential speed.

    Returns (positions: (T,3), yaws: (T,)). Yaw points along the
    instantaneous velocity so the on-board camera looks where the
    drone is going — natural for a racer.
    """
    t_total = n_frames * dt
    omega = 2 * math.pi * n_laps / t_total
    t = np.arange(n_frames) * dt
    theta = omega * t
    pos = np.stack([
        radius_x * np.cos(theta),
        radius_y * np.sin(theta),
        np.full(n_frames, altitude),
    ], axis=1)
    # tangent: derivative of (r_x cos θ, r_y sin θ) wrt time
    vx = -radius_x * np.sin(theta) * omega
    vy =  radius_y * np.cos(theta) * omega
    yaw = np.arctan2(vy, vx)
    return pos, yaw


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def pose_to_matrix(pos: np.ndarray, yaw: float) -> np.ndarray:
    """Build T_world_body for a yaw-only orientation (drone level)."""
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    T[:3, 3] = pos
    return T


# ---------------------------------------------------------------------------
# PyBullet rendering
# ---------------------------------------------------------------------------

def setup_pybullet_scene(use_egl: bool) -> int:
    """Connect to PyBullet (DIRECT) and load the textured room.

    On Linux/headless, ``use_egl=True`` loads the EGL plugin so
    ER_BULLET_HARDWARE_OPENGL renders without an X server. On Windows
    or with a display, ``use_egl=False`` works directly.
    """
    cid = pb.connect(pb.DIRECT)
    if use_egl:
        plugin_id = pb.loadPlugin("eglRendererPlugin", physicsClientId=cid)
        if plugin_id < 0:
            print("[scene] WARNING: EGL plugin failed to load — PyBullet will "
                  "fall back to its software TINY_RENDERER (no hardware GL). "
                  "Texture quality may be reduced. To enable EGL: "
                  "apt-get install libegl1-mesa libnvidia-gl-<driver-major>")
        else:
            print(f"[scene] EGL renderer plugin loaded (id={plugin_id}, "
                  "headless hardware OpenGL)")
    pb.setAdditionalSearchPath(str(ROOM_DIR), physicsClientId=cid)
    pb.loadURDF(str(ROOM_URDF),
                basePosition=[0, 0, 0], useFixedBase=True,
                physicsClientId=cid)
    return cid


def render_rgb_depth(
    cid: int, pos: np.ndarray, yaw: float,
    width: int, height: int, vfov_deg: float,
    near: float = 0.1, far: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render an RGB (uint8 HxWx3) and Z-buffer depth (float32 HxW) from
    the given drone pose, looking along its yaw direction.
    """
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    eye = pos.tolist()
    target = (pos + 5.0 * forward).tolist()
    up = [0.0, 0.0, 1.0]
    view = pb.computeViewMatrix(eye, target, up)
    proj = pb.computeProjectionMatrixFOV(
        fov=vfov_deg, aspect=width / height, nearVal=near, farVal=far,
    )
    _, _, rgba, depth, _ = pb.getCameraImage(
        width, height, viewMatrix=view, projectionMatrix=proj,
        renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        lightAmbientCoeff=0.85, lightDiffuseCoeff=0.6,
        lightSpecularCoeff=0.05, lightDirection=[0, 0, -1],
        shadow=0, physicsClientId=cid,
    )
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    depth = np.asarray(depth, dtype=np.float32).reshape(height, width).copy()
    return rgb, depth


# ---------------------------------------------------------------------------
# Rerun logging
# ---------------------------------------------------------------------------

def log_static_scene(
    map_pcd: o3d.geometry.PointCloud,
    room_dims: tuple[float, float, float],
):
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Map cloud (subsampled).
    map_arr = np.asarray(map_pcd.points)
    sub = np.random.default_rng(0).choice(
        len(map_arr), min(8000, len(map_arr)), replace=False)
    rr.log("/world/map_cloud",
           rr.Points3D(positions=map_arr[sub].tolist(),
                       colors=[(80, 90, 110)] * len(sub),
                       radii=0.025),
           static=True)

    # Room frame outline.
    w, d, h = room_dims
    hw, hd = w / 2, d / 2
    box = [
        [-hw, -hd, 0], [hw, -hd, 0], [hw, hd, 0], [-hw, hd, 0], [-hw, -hd, 0],
        [-hw, -hd, h], [hw, -hd, h], [hw, hd, h], [-hw, hd, h], [-hw, -hd, h],
    ]
    rr.log("/world/room/floor",
           rr.LineStrips3D(strips=[box[:5]],
                           colors=[(120, 120, 130)], radii=0.02),
           static=True)
    rr.log("/world/room/ceiling",
           rr.LineStrips3D(strips=[box[5:]],
                           colors=[(120, 120, 130)], radii=0.02),
           static=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/perception/mast3r_smoketest.rrd"))
    ap.add_argument("--n-frames", type=int, default=300,
                    help="Number of frames in the scripted oval (300 @ 100 Hz = 3 s)")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--n-laps", type=float, default=1.0)
    ap.add_argument("--no-egl", action="store_true",
                    help="Disable EGL plugin (use only with a real display)")
    ap.add_argument("--skip-icp", action="store_true",
                    help="Skip ICP localizer (saves ~30s)")
    ap.add_argument("--skip-mast3r", action="store_true",
                    help="Skip MASt3R-SLAM (smoke-test ICP path only)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Trajectory --------------------------------------------------
    print(f"\n[1/5] Generating scripted oval: {args.n_frames} frames "
          f"@ dt={args.dt}s ({args.n_laps} laps)")
    positions, yaws = oval_trajectory(
        args.n_frames, args.dt, n_laps=args.n_laps,
    )
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1).mean() / args.dt
    print(f"     mean speed: {speed:.2f} m/s")

    # ---- 2. PyBullet render pass ---------------------------------------
    print(f"\n[2/5] Rendering RGB+depth in PyBullet "
          f"(EGL={not args.no_egl})...")
    cid = setup_pybullet_scene(use_egl=not args.no_egl)

    rgb_buf = np.zeros((args.n_frames, HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth_buf = np.zeros((args.n_frames, HEIGHT, WIDTH), dtype=np.float32)
    t0 = time.perf_counter()
    for t in range(args.n_frames):
        rgb, depth = render_rgb_depth(
            cid, positions[t], yaws[t], WIDTH, HEIGHT, VFOV_DEG,
        )
        rgb_buf[t] = rgb
        depth_buf[t] = depth
        if t == 0 or t % 50 == 0 or t == args.n_frames - 1:
            print(f"     t={t:>3}: RGB std={rgb.std():.1f}  "
                  f"depth range=[{depth.min():.3f}, {depth.max():.3f}]")
    pb.disconnect(cid)
    render_dt = time.perf_counter() - t0
    print(f"     done in {render_dt:.1f}s "
          f"({args.n_frames / render_dt:.0f} FPS render)")

    # ---- 3. Build map cloud (for ICP) ----------------------------------
    print(f"\n[3/5] Sampling room mesh for ICP map cloud...")
    map_pcd = sample_mesh_surface(ROOM_OBJ, n_points=30_000)
    print(f"     map cloud: {len(map_pcd.points)} points, has_normals={map_pcd.has_normals()}")

    # Camera intrinsics (used by both localizers, identical).
    intr = CameraIntrinsics.from_vfov(VFOV_DEG, WIDTH, HEIGHT)
    K = np.array([
        [intr.fx, 0, intr.cx],
        [0, intr.fy, intr.cy],
        [0, 0, 1],
    ])

    # Ground-truth pose history.
    gt_T = np.zeros((args.n_frames, 4, 4))
    for t in range(args.n_frames):
        gt_T[t] = pose_to_matrix(positions[t], yaws[t])

    # ---- 4. Localize ----------------------------------------------------
    icp_T = np.zeros_like(gt_T)
    icp_fitness = np.zeros(args.n_frames)
    icp_T[:] = gt_T  # default to GT for skipped frames so plots stay finite

    if not args.skip_icp:
        print(f"\n[4a/5] Running ICP localizer...")
        icp_loc = IcpLocalizer(
            map_pcd=map_pcd, intrinsics=intr, R_body_to_cam=R_BODY_TO_CAM,
            max_iterations=30, max_correspondence_distance_m=0.20,
            min_inlier_fraction=0.20, body_cloud_voxel_size_m=0.05,
            depth_stride=2,
        )
        icp_T[0] = gt_T[0].copy()  # anchor frame 0
        prior = gt_T[0].copy()
        t0 = time.perf_counter()
        for t in range(args.n_frames):
            # Predict-then-correct: use velocity from GT as a stand-in
            # for VIO (Phase 1 isolates the localizer; not testing VIO).
            if t > 0:
                vel = (positions[t] - positions[t - 1]) / args.dt
                prior_pos = prior[:3, 3] + vel * args.dt
                prior = pose_to_matrix(prior_pos,
                                        math.atan2(prior[1, 0], prior[0, 0]))
                # blend in GT yaw (sub-deg via "IMU truth" stand-in)
                prior[:3, :3] = gt_T[t][:3, :3]
            res = icp_loc.localize(depth_buf[t], prior,
                                   is_zbuffer=True, near=0.1, far=100.0)
            icp_T[t] = res.refined_T_world_body
            icp_fitness[t] = res.fitness
            prior = icp_T[t].copy()
            if t == 0 or t % 50 == 0:
                err = np.linalg.norm(icp_T[t][:3, 3] - gt_T[t][:3, 3])
                print(f"     t={t:>3}  fit={res.fitness:.3f}  "
                      f"pos_err={err*100:6.1f} cm")
        icp_dt = time.perf_counter() - t0
        print(f"     done in {icp_dt:.1f}s "
              f"({args.n_frames / icp_dt:.0f} FPS)")

    mast_T = np.zeros_like(gt_T)
    mast_T[:] = gt_T
    mast_mode = ["skipped"] * args.n_frames

    if not args.skip_mast3r:
        print(f"\n[4b/5] Running MASt3R-SLAM localizer...")
        mast_loc = Mast3rLocalizer(
            K=K, R_body_to_cam=R_BODY_TO_CAM,
            img_size_wh=(WIDTH, HEIGHT),
        )
        mast_loc.reset(gt_T[0])
        t0 = time.perf_counter()
        for t in range(args.n_frames):
            res = mast_loc.localize(rgb_buf[t], t=t)
            mast_T[t] = res.refined_T_world_body
            mast_mode[t] = res.mode
            if t == 0 or t % 50 == 0:
                err = np.linalg.norm(mast_T[t][:3, 3] - gt_T[t][:3, 3])
                print(f"     t={t:>3}  mode={res.mode:<14}  "
                      f"pos_err={err*100:6.1f} cm")
        mast_dt = time.perf_counter() - t0
        print(f"     done in {mast_dt:.1f}s "
              f"({args.n_frames / mast_dt:.0f} FPS)")

    # ---- 5. Rerun log + save -------------------------------------------
    print(f"\n[5/5] Logging to rerun → {args.out}")
    rr.init(APP_ID, spawn=False)
    rr.save(str(args.out))
    log_static_scene(map_pcd, (14.0, 10.0, 3.0))

    rr.log("/world/path/gt",
           rr.LineStrips3D(strips=[gt_T[:, :3, 3].tolist()],
                           colors=[(140, 220, 250)], radii=0.03,
                           labels=["GT trajectory"]),
           static=True)
    if not args.skip_icp:
        rr.log("/world/path/icp",
               rr.LineStrips3D(strips=[icp_T[:, :3, 3].tolist()],
                               colors=[(140, 230, 160)], radii=0.025,
                               labels=["ICP estimate"]),
               static=True)
    if not args.skip_mast3r:
        rr.log("/world/path/mast3r",
               rr.LineStrips3D(strips=[mast_T[:, :3, 3].tolist()],
                               colors=[(255, 180, 90)], radii=0.025,
                               labels=["MASt3R-SLAM estimate"]),
               static=True)

    for t in range(args.n_frames):
        rr.set_time("sim_time", duration=t * args.dt)
        rr.log("/world/drone_gt",
               rr.Points3D(positions=[gt_T[t, :3, 3].tolist()],
                           colors=[(140, 220, 250)], radii=0.16))
        if not args.skip_icp:
            err_icp = np.linalg.norm(icp_T[t, :3, 3] - gt_T[t, :3, 3])
            rr.log("/world/drone_icp",
                   rr.Points3D(positions=[icp_T[t, :3, 3].tolist()],
                               colors=[(140, 230, 160)], radii=0.12))
            rr.log("/scalars/icp/pos_err_cm", rr.Scalars(err_icp * 100))
            rr.log("/scalars/icp/fitness", rr.Scalars(float(icp_fitness[t])))
        if not args.skip_mast3r:
            err_mast = np.linalg.norm(mast_T[t, :3, 3] - gt_T[t, :3, 3])
            rr.log("/world/drone_mast3r",
                   rr.Points3D(positions=[mast_T[t, :3, 3].tolist()],
                               colors=[(255, 180, 90)], radii=0.12))
            rr.log("/scalars/mast3r/pos_err_cm", rr.Scalars(err_mast * 100))
        # Camera image (downsampled) every 10 frames so the .rrd stays small.
        if t % 10 == 0:
            rr.log("/camera/rgb", rr.Image(rgb_buf[t]))

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="World — GT(blue) ICP(green) MASt3R(orange)",
                    origin="/world",
                    contents=["/world/**", "/scalars/**:exclude"],
                ),
                rrb.Spatial2DView(name="Camera RGB", origin="/camera"),
                column_shares=[3, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="ICP pos error (cm)",
                                   origin="/scalars/icp/pos_err_cm"),
                rrb.TimeSeriesView(name="MASt3R pos error (cm)",
                                   origin="/scalars/mast3r/pos_err_cm"),
                rrb.TimeSeriesView(name="ICP fitness",
                                   origin="/scalars/icp/fitness"),
            ),
            row_shares=[3, 1],
        ),
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    # ---- Summary ----
    print(f"\n=== Summary ===")
    if not args.skip_icp:
        err = np.linalg.norm(icp_T[:, :3, 3] - gt_T[:, :3, 3], axis=1)
        print(f"  ICP    : mean err = {err.mean()*100:.1f} cm  "
              f"max = {err.max()*100:.1f} cm  "
              f"fitness mean = {icp_fitness.mean():.3f}")
    if not args.skip_mast3r:
        err = np.linalg.norm(mast_T[:, :3, 3] - gt_T[:, :3, 3], axis=1)
        n_lost = sum(1 for m in mast_mode if m in ("tracking_lost", "reloc_pending"))
        print(f"  MASt3R : mean err = {err.mean()*100:.1f} cm  "
              f"max = {err.max()*100:.1f} cm  "
              f"lost frames = {n_lost}/{args.n_frames}")
    print(f"\n.rrd written: {args.out.resolve()}")
    print(f"Open in browser:")
    print(f"  rerun {args.out} --web-viewer --bind 0.0.0.0 --web-viewer-port 9876")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

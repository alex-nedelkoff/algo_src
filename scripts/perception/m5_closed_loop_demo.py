"""M5 — Closed-loop test of G&CNet on the planner's waypoints (COR-106).

Drives the trained MoE generalist (``outputs/expanded_5expert_v3``,
28-dim obs, 4-dim TRPY action) through the M3 ``WaypointTrack`` adapter
fed by the M4 A* planner. Compares two trajectories:

    open-loop:    drone follows the planner's GT waypoints exactly
                  (the demo from week2 — lap time = total path length / speed).
    closed-loop:  drone obeys G&CNet actions; env enforces dynamics.
                  We log policy-driven pose at each step, count gate passes,
                  and timestamp completion.

Why this is M5: the M1 architecture call ("vision is augmentation,
G&CNet stays unmodified") only works if the trained policy actually
follows planner-emitted waypoints — including the off-axis ones with
sharp yaw transitions where the M4 detour bends. M5 is the first time
we verify that empirically.

What this demo deliberately does NOT do:
    - Warehouse rendering / FPV / DA V2 depth — that's the week-2 demo.
      M5 focuses on whether the policy can fly the path; the perception
      stack already shipped in M2.
    - Real VIO — using ground-truth pose for the policy's obs.
    - Closed-loop replanning — the planner output is fixed at start.

Frame of reference: the env hardcodes ``z <= 0 → terminate``, so we
shift all xy/z up by ``ALTITUDE_OFFSET`` so the drone flies at z ≈ 1.8
in the env. Visualization shows the original (warehouse-frame) coords.

Usage::

    python -m scripts.perception.m5_closed_loop_demo
    rerun outputs/perception/m5_closed_loop_demo.rrd
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402  preload before sb3

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as pb  # noqa: E402

import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402

from control.algorithms.ppo import PPO  # noqa: E402
from perception.planning.cost_map_planner import plan as plan_path  # noqa: E402
from perception.planning.occupancy_grid import (  # noqa: E402
    GridExtent, OccupancyGrid2D5,
)
from sim.envs.gate_race_env import GateRaceEnv  # noqa: E402
from sim.tracks.waypoint import waypoints_to_track  # noqa: E402
from scripts.perception.week1_demo import (  # noqa: E402
    WarehouseRenderer, quat_to_rotmat, warehouse_centered_oval,
    build_off_axis_waypoints,
)
from scripts.perception.week2_planner_demo import (  # noqa: E402
    DRONE_RADIUS_M, OBSTACLE_RADIUS_M, SPEED_MPS,
    plan_full_lap, parameterize_constant_speed,
)


APP_ID = "vision_aug_racing_m5_closed_loop_demo_v1"
CHECKPOINT = Path("C:/Users/alexj/Documents/algo_src/outputs/expanded_5expert_v3/model.zip")
ALTITUDE_OFFSET = 6.0  # lift everything so env's z<=0 ground-check doesn't fire


# --------------------------------------------------------------------- policy


class _Dummy28(gym.Env):
    """Minimal env used only to construct the policy with the right shapes."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(28, dtype=np.float32), {}

    def step(self, a):
        return np.zeros(28, dtype=np.float32), 0.0, False, False, {}


def load_moe_policy(checkpoint_path: Path) -> PPO:
    """Build the 5-expert MoE on a dummy 28-dim env, then load weights."""
    policy = PPO(moe=True, n_experts=5, top_k=2, expert_hidden_dim=128)
    policy._model = policy._create_model(_Dummy28())
    policy.load(str(checkpoint_path))
    return policy


# --------------------------------------------------------------------- obs slice


def slice_obs_33_to_28(obs_33: np.ndarray) -> np.ndarray:
    """Drop width / height / arena-extent dims so the env's 33-dim obs
    matches what the 28-dim checkpoint expects.

    33-dim layout: 0–19 (state), 20–25 (gate+1: rel_pos+yaw+w+h),
                   26–31 (gate+2: same), 32 (arena_extent).
    28-dim layout: 0–19 (state), 20–22 (gate+1 rel_pos), 23 (gate+1 yaw),
                   24–26 (gate+2 rel_pos), 27 (gate+2 yaw).
    """
    if obs_33.shape[-1] != 33:
        raise ValueError(f"expected 33-dim obs, got {obs_33.shape[-1]}")
    obs_28 = np.empty(28, dtype=obs_33.dtype)
    obs_28[0:20] = obs_33[0:20]
    obs_28[20:23] = obs_33[20:23]   # gate+1 rel_pos
    obs_28[23] = obs_33[23]         # gate+1 yaw
    obs_28[24:27] = obs_33[26:29]   # gate+2 rel_pos (skipping w/h at 24-25)
    obs_28[27] = obs_33[29]         # gate+2 yaw (skipping w/h at 30-31, arena at 32)
    return obs_28


# --------------------------------------------------------------------- planner


def build_planner_path(
    obstacle_xy: np.ndarray,
    gates_xyz: np.ndarray,
    arena_aabb_min: np.ndarray,
    arena_aabb_max: np.ndarray,
    cell_size_m: float,
    altitude_m: float,
    *,
    smooth_globally: bool = True,
    target_spacing_m: float = 4.0,
) -> np.ndarray:
    """Spawn cylinder, build occupancy, plan A*. Returns lap waypoints.

    When ``smooth_globally=True`` (recommended): per-segment A* outputs
    are concatenated raw, then resampled with a single global cubic
    B-spline arc-length pass. This produces continuous tangents
    (and therefore continuous synthetic yaws) through segment joins,
    not just within segments. Without it, the joint-points produce
    sharp yaw discontinuities that the trained policy struggles with —
    M5 stalled at ~5 gates because of this.
    """
    cid = pb.connect(pb.DIRECT)
    try:
        col = pb.createCollisionShape(
            pb.GEOM_CYLINDER, radius=OBSTACLE_RADIUS_M, height=4.0,
            physicsClientId=cid,
        )
        body = pb.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=col,
            basePosition=[float(obstacle_xy[0]), float(obstacle_xy[1]), altitude_m],
            physicsClientId=cid,
        )

        extent = GridExtent(
            x_min=float(arena_aabb_min[0]) - 1.0, x_max=float(arena_aabb_max[0]) + 1.0,
            y_min=float(arena_aabb_min[1]) - 1.0, y_max=float(arena_aabb_max[1]) + 1.0,
            cell_size_m=cell_size_m,
        )
        occ_raw = OccupancyGrid2D5.from_pybullet_probe_sphere(
            client_id=cid, body_ids=[body],
            extent=extent, altitude_m=altitude_m,
            probe_radius_m=cell_size_m / 2.0,
        )
        occ = occ_raw.dilate(DRONE_RADIUS_M)

        if smooth_globally:
            # Get raw A* output (no per-segment resampling — that's where
            # the per-segment-end tangent discontinuities came from).
            full_path = plan_full_lap(occ, gates_xyz, target_spacing_m=None)
            # plan_full_lap already returns a closed loop (last cell == gate 0
            # == first cell), so DON'T re-close it inside the resampler —
            # that would put two consecutive duplicate points and splprep
            # rejects with "Invalid inputs".
            from sim.tracks.waypoint import resample_waypoints
            full_path = resample_waypoints(
                full_path, target_spacing=target_spacing_m,
                method="cubic", closed_loop=False,
            )
        else:
            full_path = plan_full_lap(occ, gates_xyz, target_spacing_m=target_spacing_m)
        return full_path
    finally:
        pb.disconnect(cid)


# --------------------------------------------------------------------- closed loop


def _training_racing_params() -> "VehicleParams":  # type: ignore  # noqa: F821
    """5" racing quad params from configs/sim/numpy_quad.yaml — what the
    MoE was trained against. Mass / arm-length / k_thrust differ from
    the CrazyFlie defaults by ~30×; running the policy on CF dynamics
    via the TRPY mixer puts it OOD and it stalls before reaching gates.
    """
    from sim.dynamics.params import VehicleParams
    return VehicleParams(
        mass=0.752, arm_length=0.170,
        k_thrust=2.49e-6, k_torque=8.80e-8, tau_motor=0.04,
        inertia=np.diag([0.0025, 0.0025, 0.0045]),
        drag_coeff=np.array([0.01, 0.01, 0.005]),
        max_rpm=31470.0,
    )


def run_closed_loop(
    track,
    policy: PPO,
    *,
    initial_state: np.ndarray | None,
    n_steps: int,
    dt: float,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Step env with policy actions; return logged state per timestep + metrics.

    Training-matched env config: 5" racing quad params, esc_nonlinearity=0.95,
    gate_passage_radius=1.0, random_gate_start (start_vel_std=1.5),
    arena_bounds=15. The vehicle params are essential — the MoE was
    trained against a 5" racing quad, and the TRPY mixer's normalized
    [-1,1] action mapping is calibrated against those params. Running
    on CrazyFlie dynamics (~30× lighter) put the policy OOD and it
    stalled before reaching any gate.
    """
    env = GateRaceEnv(
        track=track,
        params=_training_racing_params(),
        n_envs=1,
        dt=dt,
        max_steps=n_steps,
        n_lookahead_gates=2,
        n_action_history=0,
        gate_passage_radius=1.0,
        gate_collision=False,
        action_mode="trpy",
        esc_nonlinearity=0.95,
        ceiling=20.0,
        random_gate_start=True,
        start_vel_std=1.5,
        start_att_std=0.1,
        arena_bounds=15.0,
    )
    env.reset(seed=seed)
    if initial_state is not None:
        # Optional manual override — but the env's random_gate_start
        # produces an in-distribution start, so prefer letting it pick.
        env._states[0] = initial_state
        env._step_counts[0] = 0
        env._gate_indices[0] = 0
        env._prev_along_normal[0] = 0.0

    states = np.zeros((n_steps, 17), dtype=np.float64)
    actions = np.zeros((n_steps, 4), dtype=np.float64)
    obs_log = np.zeros((n_steps, 28), dtype=np.float32)
    gate_indices = np.zeros(n_steps, dtype=np.int64)
    terminated_at = None
    term_reason = 0
    # Track gates / laps live — env's counter may reset on truncation.
    max_gates_seen = 0
    max_laps_seen = 0

    for t in range(n_steps):
        states[t] = env._states[0]
        gate_indices[t] = int(env._gate_indices[0])
        obs_33 = env._compute_obs_batched()[0]
        obs_28 = slice_obs_33_to_28(obs_33)
        obs_log[t] = obs_28

        action, _ = policy._model.predict(obs_28, deterministic=True)
        actions[t] = action
        _, _, term, trunc, info = env.step(action[None, :])

        max_gates_seen = max(max_gates_seen, int(env._gates_passed[0]))
        max_laps_seen = max(max_laps_seen, int(env._laps_completed[0]))

        if bool(term[0]) or bool(trunc[0]):
            term_reason = int(env._termination_reasons[0])
            terminated_at = t + 1
            break

    metrics = {
        "n_steps_taken": terminated_at if terminated_at is not None else n_steps,
        "gates_passed": max_gates_seen,
        "laps_completed": max_laps_seen,
        "term_reason": term_reason,
        "final_state": env._states[0].copy(),
    }
    return states, actions, obs_log, gate_indices, metrics


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/perception/m5_closed_loop_demo.rrd"))
    ap.add_argument("--dt", type=float, default=0.01,
                    help="env sim step (must match the policy's training dt)")
    ap.add_argument("--max-time-s", type=float, default=120.0)
    ap.add_argument("--cell-size-m", type=float, default=0.25)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- Training-scale track ----------------------------------------------
    # The local 28-dim MoE checkpoint was trained on procedural tracks with
    # 1-6 m gate spacing. The full warehouse oval (gate spacings 7-12 m) is
    # OOD and the policy stalls. Use a clean small oval at origin so the
    # closed-loop pipeline is exercised within the policy's competence.

    print("Building training-scale oval (8 gates, 3m x 2.5m, z=2.0)...")
    n_gates = 8
    theta = np.linspace(0, 2 * np.pi, n_gates, endpoint=False)
    gates_env = np.stack([
        3.0 * np.cos(theta),
        2.5 * np.sin(theta),
        np.full(n_gates, 2.0),
    ], axis=1)
    seg_lens = [
        float(np.linalg.norm(gates_env[(i + 1) % n_gates] - gates_env[i]))
        for i in range(n_gates)
    ]
    print(f"  gate spacings (m): {[round(s, 2) for s in seg_lens]}")

    # Inject a cylinder midway between gates 2 and 3 — forces the planner
    # to detour outward from the racing line.
    obstacle_xy = 0.5 * (gates_env[2, 0:2] + gates_env[3, 0:2])
    print(f"Cylinder obstacle at xy={obstacle_xy.round(2).tolist()}, radius={OBSTACLE_RADIUS_M} m")

    # Define an arena AABB around the gates for the occupancy grid.
    aabb_min_env = np.array([gates_env[:, 0].min() - 1.0,
                             gates_env[:, 1].min() - 1.0,
                             0.0])
    aabb_max_env = np.array([gates_env[:, 0].max() + 1.0,
                             gates_env[:, 1].max() + 1.0,
                             5.0])
    altitude_env = 2.0

    # Best config from a parameter sweep: cubic-spline smoothed, 1 m target
    # spacing, lookahead=1 (matches training's "gate points to next gate"
    # convention), no extra EMA yaw smoothing (the spline is already smooth).
    # This combo took the closed-loop policy from 5 gates → 154 gates / 7 laps
    # on the same scene — comparable to the clean-oval baseline (129 / 16).
    print("Planning A* path around the obstacle (cubic-spline smoothed @ 1m spacing)...")
    full_path_env = build_planner_path(
        obstacle_xy, gates_env, aabb_min_env, aabb_max_env,
        cell_size_m=args.cell_size_m, altitude_m=altitude_env,
        smooth_globally=True, target_spacing_m=1.0,
    )
    print(f"  {full_path_env.shape[0]} waypoints over the lap")

    track = waypoints_to_track(
        full_path_env, yaw_lookahead=1, yaw_smoothing_alpha=1.0, closed_loop=True,
    )

    # Open-loop baseline trajectory at the same speed.
    open_loop_poses = parameterize_constant_speed(
        full_path_env, speed_mps=SPEED_MPS, dt=args.dt,
    )
    open_loop_steps = open_loop_poses.shape[0]
    print(f"  open-loop reference: {open_loop_steps} steps "
          f"({open_loop_steps * args.dt:.1f} s)")

    # ---- Load policy --------------------------------------------------------

    print(f"\nLoading checkpoint: {CHECKPOINT.name}")
    policy = load_moe_policy(CHECKPOINT)

    # ---- Closed-loop run ----------------------------------------------------

    n_steps_max = int(round(args.max_time_s / args.dt))
    print(f"\nClosed-loop run: max {args.max_time_s} s = {n_steps_max} steps")

    t0 = time.perf_counter()
    # Let the env's random_gate_start handle initialization (matches training).
    cl_states, cl_actions, cl_obs, cl_gate_idx, metrics = run_closed_loop(
        track, policy, initial_state=None, n_steps=n_steps_max, dt=args.dt, seed=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"  ran in {elapsed:.1f} s of wall clock")
    print(f"  steps taken: {metrics['n_steps_taken']}")
    print(f"  gates passed: {metrics['gates_passed']}")
    print(f"  laps completed: {metrics['laps_completed']}")
    print(f"  termination reason: {metrics['term_reason']} "
          f"(see GATE_RACE_TERM_NAMES; 0=none, 1=ground, 2=ceiling, 3=quat, "
          f"7=gate_collision, 8=timeout)")

    # ---- Visualisation ------------------------------------------------------

    print(f"\nWriting {args.out}...")
    rr.init(APP_ID, spawn=False)
    rr.save(str(args.out))
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # No frame shifting now — track is at z=2.0 directly.
    def _to_warehouse(arr: np.ndarray, dim_xyz_slice: slice = slice(0, 3)) -> np.ndarray:
        return arr.copy()

    # Cylinder obstacle (visualised as a stack of rings at z=0..4).
    obstacle_z_min = 0.0
    obstacle_z_max = 4.0
    n_ring_pts = 32
    rings = []
    for z in np.linspace(obstacle_z_min, obstacle_z_max, 5):
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
        rr.LineStrips3D(strips=rings, colors=[(255, 80, 80)] * len(rings),
                        radii=0.04),
        static=True,
    )

    # Gates (in warehouse frame).
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
        # No frame shift — track is at native z=2.0.
        gate_pos_warehouse = gate.position.copy()
        frame_world = (rotmat @ frame_local.T).T + gate_pos_warehouse
        rr.log(
            f"/world/gates/gate_{i:02d}",
            rr.LineStrips3D(strips=[frame_world.tolist()], colors=[color], radii=0.025,
                            labels=[f"wp{i}"]),
            static=True,
        )

    # Open-loop and closed-loop trajectories (full polylines, static).
    rr.log(
        "/world/path/open_loop",
        rr.LineStrips3D(
            strips=[_to_warehouse(open_loop_poses[:, 0:3]).tolist()],
            colors=[(140, 220, 250)], radii=0.03,
            labels=["open-loop GT"],
        ),
        static=True,
    )
    n_cl = metrics["n_steps_taken"]
    rr.log(
        "/world/path/closed_loop",
        rr.LineStrips3D(
            strips=[_to_warehouse(cl_states[:n_cl, 0:3]).tolist()],
            colors=[(140, 230, 160)], radii=0.04,
            labels=["closed-loop policy"],
        ),
        static=True,
    )

    # ---- Animated scrub: drone position + actions over time -----------------

    n_frames = max(open_loop_steps, n_cl)
    for t in range(n_frames):
        sim_time_s = t * args.dt
        rr.set_time("sim_time", duration=sim_time_s)

        # Open-loop drone (still flies the GT path).
        if t < open_loop_steps:
            ol_pos = open_loop_poses[t, 0:3].copy()
            rr.log(
                "/world/drone_open_loop",
                rr.Points3D(positions=[ol_pos.tolist()],
                            colors=[(140, 220, 250)], radii=0.16),
            )

        # Closed-loop drone.
        if t < n_cl:
            cl_pos = cl_states[t, 0:3].copy()
            rr.log(
                "/world/drone_closed_loop",
                rr.Points3D(positions=[cl_pos.tolist()],
                            colors=[(140, 230, 160)], radii=0.16),
            )

            # Action scalars: thrust + body rates.
            a = cl_actions[t]
            rr.log("/scalars/action/thrust", rr.Scalars(float(a[0])))
            rr.log("/scalars/action/roll_rate", rr.Scalars(float(a[1])))
            rr.log("/scalars/action/pitch_rate", rr.Scalars(float(a[2])))
            rr.log("/scalars/action/yaw_rate", rr.Scalars(float(a[3])))
            # Distance-from-planned-path: cross-track error.
            if t < open_loop_steps:
                err = float(np.linalg.norm(cl_states[t, 0:3] - open_loop_poses[t, 0:3]))
                rr.log("/scalars/cross_track_error_m", rr.Scalars(err))
            rr.log("/scalars/gate_idx", rr.Scalars(int(cl_gate_idx[t])))

    # Blueprint.
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(
                name="World — gates + obstacle + open-loop (blue) vs closed-loop (green)",
                origin="/world", contents=["/world/**", "/scalars/**:exclude"],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="Cross-track error (m)",
                                   origin="/scalars/cross_track_error_m"),
                rrb.TimeSeriesView(name="Action — TRPY (normalized)",
                                   origin="/scalars/action"),
                rrb.TimeSeriesView(name="Gate index",
                                   origin="/scalars/gate_idx"),
            ),
            row_shares=[3, 1],
        ),
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    print(f"\n  Demo .rrd → {args.out.resolve()}")
    print(f"  Open with: rerun {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Core benchmark runner — evaluate a checkpoint against the golden set."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf

from benchmark.results import (
    AggregateResult,
    BenchmarkResults,
    EnvResult,
    LayoutResult,
    aggregate_env_results,
    aggregate_layout_results,
)
from benchmark.track_loader import GoldenSetLoader

log = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "golden_set"


def compute_env_result(
    layout_name: str,
    variant_index: int,
    episode_info: dict[str, Any],
) -> EnvResult:
    """Convert raw episode info dict to an EnvResult."""
    laps = int(episode_info["laps_completed"])
    ep_len = int(episode_info["l"])
    dt = float(episode_info["effective_dt"])
    termination = str(episode_info["termination"])

    lap_time = (ep_len * dt) / laps if laps > 0 else None

    return EnvResult(
        layout_name=layout_name,
        variant_index=variant_index,
        lap_time=lap_time,
        laps_completed=laps,
        gates_passed=int(episode_info["gates_passed"]),
        crash=termination != "timeout",
        avg_speed=float(episode_info["avg_speed"]),
        termination_reason=termination,
    )


def build_initial_state(
    track_data: dict[str, Any],
    variant_cfg: dict[str, Any],
) -> tuple[NDArray[np.float64], int]:
    """Build the 17-element initial state vector and gate_index from variant config."""
    from sim.tracks import Track

    track: Track = track_data["track"]
    if "gate_fraction" in variant_cfg:
        gate_index = int(track.num_gates * float(variant_cfg["gate_fraction"]))
    else:
        gate_index = int(variant_cfg.get("gate_index", 0))
    gate_index = min(gate_index, track.num_gates - 1)

    state = np.zeros(17, dtype=np.float64)

    if variant_cfg.get("position") is not None:
        state[0:3] = np.array(variant_cfg["position"], dtype=np.float64)
    else:
        gate = track.gates[gate_index]
        q = gate.orientation
        w, x, y, z = q
        normal = np.array([
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
        ])
        behind_dist = 1.0
        state[0:3] = gate.position - behind_dist * normal

        offset = variant_cfg.get("position_offset")
        if offset is not None:
            state[0:3] += np.array(offset, dtype=np.float64)

    vel = variant_cfg.get("velocity", [0.0, 0.0, 0.0])
    state[3:6] = np.array(vel, dtype=np.float64)

    if variant_cfg.get("attitude") is not None:
        state[6:10] = np.array(variant_cfg["attitude"], dtype=np.float64)
    else:
        gate = track.gates[gate_index]
        q = gate.orientation.copy()
        perturb = variant_cfg.get("attitude_perturbation")
        if perturb is not None:
            roll_p, pitch_p = float(perturb[0]), float(perturb[1])
            cr, sr = np.cos(roll_p / 2), np.sin(roll_p / 2)
            cp, sp = np.cos(pitch_p / 2), np.sin(pitch_p / 2)
            q_roll = np.array([cr, sr, 0, 0])
            q_pitch = np.array([cp, 0, sp, 0])
            q = _quat_mul(q, _quat_mul(q_roll, q_pitch))
        state[6:10] = q / np.linalg.norm(q)

    omega = variant_cfg.get("omega", [0.0, 0.0, 0.0])
    state[10:13] = np.array(omega, dtype=np.float64)

    state[13:17] = 500.0

    return state, gate_index


def _quat_mul(q1: NDArray, q2: NDArray) -> NDArray:
    """Hamilton quaternion product [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def run_single_env(
    policy,
    track_data: dict[str, Any],
    layout_name: str,
    variant_index: int,
    variant_cfg: dict[str, Any],
    trajectory_dir: Path | None = None,
    sim_params: dict[str, Any] | None = None,
) -> EnvResult:
    """Run a single benchmark episode and return the result.

    Args:
        trajectory_dir: If provided, save per-step trajectory data as .npz
            (compatible with the existing trajectory recorder / rerun pipeline).
        sim_params: Physics/env params dict (dt, params, esc_nonlinearity, etc.)
            to match the training config. If None, uses GateRaceEnv defaults.
    """
    from sim.envs.gate_race_env import GateRaceEnv
    from training.trajectory_recorder import SCHEMA_VERSION

    initial_state, gate_index = build_initial_state(track_data, variant_cfg)

    # Build env with the same physics params the policy was trained with.
    # sim_params is passed through from run_benchmark() — contains VehicleParams
    # and env settings like esc_nonlinearity from the training config.
    sim_kwargs = sim_params or {}
    env = GateRaceEnv(
        track=track_data["track"],
        n_envs=1,
        dt=sim_kwargs.get("dt", 0.01),
        max_steps=track_data["max_steps"],
        gate_passage_radius=track_data["gate_passage_radius"],
        arena_bounds=track_data["arena_bounds"],
        random_gate_start=False,
        params=sim_kwargs.get("params"),
        esc_nonlinearity=sim_kwargs.get("esc_nonlinearity", 0.5),
        gate_collision=sim_kwargs.get("gate_collision", False),
        action_mode=sim_kwargs.get("action_mode", "motor_rpm"),
        n_lookahead_gates=sim_kwargs.get("n_lookahead_gates", 1),
    )
    try:
        obs, _ = env.reset(options={
            "initial_state": initial_state,
            "gate_index": gate_index,
        })

        # Trajectory recording buffers
        record = trajectory_dir is not None
        positions_list: list[np.ndarray] = []
        quaternions_list: list[np.ndarray] = []
        velocities_list: list[np.ndarray] = []
        body_rates_list: list[np.ndarray] = []
        motor_rpms_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        rewards_list: list[float] = []
        reward_components_list: list[np.ndarray] = []
        gate_events_list: list[tuple[int, int]] = []

        prev_gates_passed = int(env._gates_passed[0])
        timestep = 0

        done = False
        while not done:
            if record:
                # Record state BEFORE action (same as TrajectoryRecorderCallback)
                state = env.get_state(0)
                positions_list.append(state["position"])
                quaternions_list.append(state["quaternion"])
                velocities_list.append(state["velocity"])
                body_rates_list.append(state["body_rates"])
                motor_rpms_list.append(state["motor_rpms"])

            pre_step_gate_idx = int(env._gate_indices[0])

            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated[0] or truncated[0]

            if record:
                actions_list.append(np.asarray(action, dtype=np.float64).flatten()[:4])
                rewards_list.append(float(reward[0]) if hasattr(reward, '__getitem__') else float(reward))
                _, rc_values = env.get_step_reward_components(0)
                reward_components_list.append(rc_values)

                # Detect gate passage
                if done:
                    ep = info.get("episode", {})
                    gp = ep.get("gates_passed", np.array([0]))
                    curr_gp = int(gp[0]) if hasattr(gp, '__getitem__') else int(gp)
                else:
                    curr_gp = int(env._gates_passed[0])
                if curr_gp > prev_gates_passed:
                    gate_events_list.append((timestep, pre_step_gate_idx))
                prev_gates_passed = curr_gp

            timestep += 1

        # Save trajectory if requested
        if record and trajectory_dir is not None:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            npz_path = trajectory_dir / f"{layout_name}_v{variant_index}.npz"

            gate_geom = env.get_gate_geometry()
            rc_names, _ = env.get_step_reward_components(0)

            gate_events = (
                np.array(gate_events_list, dtype=np.int64)
                if gate_events_list
                else np.zeros((0, 2), dtype=np.int64)
            )

            np.savez_compressed(
                npz_path,
                schema_version=np.int64(SCHEMA_VERSION),
                positions=np.array(positions_list, dtype=np.float64),
                quaternions=np.array(quaternions_list, dtype=np.float64),
                velocities=np.array(velocities_list, dtype=np.float64),
                body_rates=np.array(body_rates_list, dtype=np.float64),
                motor_rpms=np.array(motor_rpms_list, dtype=np.float64),
                actions=np.array(actions_list, dtype=np.float64),
                rewards=np.array(rewards_list, dtype=np.float64),
                reward_components=np.array(reward_components_list, dtype=np.float64),
                reward_component_names=np.array(rc_names, dtype=str) if rc_names else np.array([], dtype=str),
                gate_events=gate_events,
                gate_positions=gate_geom["positions"],
                gate_orientations=gate_geom["orientations"],
                gate_half_extents=gate_geom["half_extents"],
                dt=np.float64(env.dt),
            )
            log.info("    Trajectory saved: %s", npz_path)

        # Extract episode info for env 0
        ep = info.get("episode", {})
        episode_info = {
            "l": int(ep["l"][0]),
            "effective_dt": float(ep["effective_dt"]),
            "laps_completed": int(ep["laps_completed"][0]),
            "gates_passed": int(ep["gates_passed"][0]),
            "termination": str(ep["termination"][0]),
            "avg_speed": float(ep["avg_speed"][0]),
        }
        return compute_env_result(layout_name, variant_index, episode_info)

    finally:
        env.close()


def run_benchmark(
    checkpoint_path: str,
    golden_set_version: str = "v1",
    golden_set_mode: str = "holdout",
    local_golden_dir: Path | None = None,
    save_trajectories: bool = True,
    output_dir: Path | None = None,
    sim_config: str | None = None,
) -> BenchmarkResults:
    """Run the full benchmark suite against a checkpoint.

    Args:
        checkpoint_path: Path to the trained model .zip file.
        golden_set_version: Which golden set version to use.
        golden_set_mode: "holdout" or "trained".
        local_golden_dir: Optional local directory with .npz track files.
        save_trajectories: If True, save per-step trajectory .npz files
            (compatible with the rerun pipeline).
        output_dir: Directory for trajectory output. Defaults to
            outputs/benchmark/<timestamp>/.
        sim_config: Path to sim config YAML (e.g., configs/sim/numpy_quad.yaml)
            to use matching physics params. If None, uses defaults.
    """
    from control.algorithms.ppo import PPO

    log.info("Loading checkpoint: %s", checkpoint_path)
    policy = PPO()
    policy.load(checkpoint_path)

    # Load sim params from config if provided
    sim_params = None
    if sim_config:
        sim_cfg = OmegaConf.load(sim_config)
        sim_params_dict: dict[str, Any] = {}
        sim_params_dict["dt"] = float(sim_cfg.get("dt", 0.01))
        sim_params_dict["esc_nonlinearity"] = float(sim_cfg.get("esc_nonlinearity", 0.5))
        sim_params_dict["gate_collision"] = bool(sim_cfg.get("gate_collision", False))
        sim_params_dict["action_mode"] = str(sim_cfg.get("action_mode", "motor_rpm"))
        sim_params_dict["n_lookahead_gates"] = int(sim_cfg.get("n_lookahead_gates", 1))
        # Build VehicleParams if present
        if "params" in sim_cfg:
            from sim.dynamics.params import VehicleParams
            p = sim_cfg.params
            sim_params_dict["params"] = VehicleParams(
                mass=float(p.mass),
                arm_length=float(p.arm_length),
                k_thrust=float(p.k_thrust),
                k_torque=float(p.k_torque),
                tau_motor=float(p.tau_motor),
                prop_radius=float(p.prop_radius),
                inertia=list(p.inertia),
                drag_coeff=list(p.drag_coeff),
                max_rpm=float(p.max_rpm),
            )
        sim_params = sim_params_dict
        log.info("Using sim params from %s", sim_config)

    log.info("Loading golden set %s...", golden_set_version)
    loader = GoldenSetLoader(version=golden_set_version, local_dir=local_golden_dir)
    tracks = loader.load_all()

    sc_path = CONFIGS_DIR / "starting_conditions" / "defaults.yaml"
    sc_cfg = OmegaConf.load(sc_path)
    variants = OmegaConf.to_container(sc_cfg.variants, resolve=True)

    # Setup trajectory output directory
    trajectory_dir = None
    if save_trajectories:
        if output_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = Path("outputs") / "benchmark" / timestamp
        trajectory_dir = output_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        log.info("Trajectories will be saved to: %s", trajectory_dir)

    all_env_results: list[EnvResult] = []
    layout_results: list[LayoutResult] = []

    for layout_name, track_data in sorted(tracks.items()):
        log.info("Benchmarking layout: %s (%d gates)",
                 layout_name, track_data["track"].num_gates)
        layout_envs: list[EnvResult] = []

        for i, variant in enumerate(variants):
            log.info("  Variant %d: %s", i, variant.get("name", f"v{i}"))
            result = run_single_env(
                policy, track_data, layout_name, i, variant,
                trajectory_dir=trajectory_dir,
                sim_params=sim_params,
            )
            layout_envs.append(result)
            all_env_results.append(result)
            log.info("    laps=%d gates=%d crash=%s lap_time=%s",
                     result.laps_completed, result.gates_passed,
                     result.crash, result.lap_time)

        layout_result = aggregate_env_results(layout_name, layout_envs)
        layout_results.append(layout_result)

    aggregate = aggregate_layout_results(layout_results)

    # Convert trajectories to .rrd and upload to R2
    if trajectory_dir is not None:
        rerun_urls = _convert_and_upload_rrd(trajectory_dir)
        # Attach rerun URLs to EnvResults
        for env_r in all_env_results:
            key = f"{env_r.layout_name}_v{env_r.variant_index}"
            env_r.rerun_url = rerun_urls.get(key)

    return BenchmarkResults(
        golden_set_version=golden_set_version,
        checkpoint_path=checkpoint_path,
        golden_set_mode=golden_set_mode,
        per_environment=all_env_results,
        per_layout=layout_results,
        aggregate=aggregate,
    )


def _convert_and_upload_rrd(trajectory_dir: Path) -> dict[str, str]:
    """Convert .npz to .rrd and upload to R2. Returns {name: rerun_viewer_url}."""
    # Step 1: Convert to .rrd
    try:
        from sim.viz.rerun_generator import batch_generate
    except ImportError:
        log.info("rerun-sdk or sim.viz dependencies not available — "
                 "skipping .rrd conversion.")
        return {}

    npz_files = sorted(trajectory_dir.glob("*.npz"))
    if not npz_files:
        return {}

    log.info("Converting %d trajectories to .rrd format (pytorch3d)...", len(npz_files))
    rrd_dir = trajectory_dir.parent / "rrd"
    try:
        rrd_paths = batch_generate(trajectory_dir, output_dir=rrd_dir, renderer="pytorch3d")
        log.info("Generated %d .rrd files in %s", len(rrd_paths), rrd_dir)
    except Exception as e:
        log.warning("Failed to convert trajectories to .rrd: %s", e)
        return {}

    # Step 2: Upload to R2
    rerun_urls: dict[str, str] = {}
    try:
        from artifacts.r2 import make_r2_client_from_env, upload_file, DEFAULT_BUCKET, rerun_viewer_url

        try:
            client = make_r2_client_from_env()
        except EnvironmentError:
            log.warning("R2 credentials not set — .rrd files saved locally only.")
            return {}

        # Use W&B run ID as path prefix if available
        try:
            import wandb
            run_id = wandb.run.id if wandb.run else "local"
        except Exception:
            run_id = "local"

        log.info("Uploading %d .rrd files to R2...", len(rrd_paths))
        for rrd_path in rrd_paths:
            r2_key = f"benchmark/runs/{run_id}/rrd/{rrd_path.name}"
            url = upload_file(
                client, rrd_path, DEFAULT_BUCKET, r2_key,
                content_type="application/octet-stream", skip_existing=False,
            )
            if url:
                viewer_url = rerun_viewer_url(r2_key)
                rerun_urls[rrd_path.stem] = viewer_url

        log.info("Uploaded %d .rrd files to R2.", len(rerun_urls))

    except Exception as e:
        log.warning("Failed to upload .rrd files to R2: %s", e)

    return rerun_urls

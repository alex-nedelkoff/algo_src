"""M1 — Localization drift benchmark via synthetic VIO.

Generates a realistic 30 s racing trajectory (parametric oval), runs the
synthetic-VIO noise model across three profiles
(``orb_slam3_mono_inertial`` / ``orb_slam3_with_map_matching`` /
``imu_dead_reckoning``) over multiple seeds, and reports drift metrics
that decide the architecture call (vision augmentation vs primary).

Why synthetic instead of a real VIO library: every Windows-friendly
candidate (DPVO, DROID-SLAM, ORB-SLAM3 bindings) requires a multi-day
Windows build with CUDA/CMake gymnastics. The architecture decision
this benchmark unblocks needs realistic *order-of-magnitude* drift
numbers, not a measurement of any specific implementation. The
``SyntheticVIO`` noise model is calibrated to literature norms — see
``perception/localization/synthetic_vio.py`` for the references.

Output: ``outputs/perception/localization_drift/``
    summary.json         — aggregated mean / p99 / max across seeds
    drift_curves.png     — drift over time per profile (mean ± std band)
    histograms.png       — final-step drift distribution per profile
    raw/<profile>/seed_<n>.npz  — per-run state for reproducibility

Usage::

    python -m scripts.perception.benchmark_localization_drift \\
        [--profiles orb_slam3_mono_inertial orb_slam3_with_map_matching imu_dead_reckoning] \\
        [--n-seeds 30] \\
        [--duration-s 30] \\
        [--dt 0.01]
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from perception.localization.synthetic_vio import (  # noqa: E402
    PROFILES,
    SyntheticVIO,
    _yaw_to_quat_wxyz,
)


# --------------------------------------------------------------------- traj


def parametric_oval_trajectory(
    duration_s: float = 30.0,
    dt: float = 0.01,
    radius_x: float = 6.0,
    radius_y: float = 4.0,
    altitude: float = 2.0,
    speed_target_mps: float = 5.0,
    altitude_amplitude_m: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Generate a representative racing trajectory: oval at constant target speed.

    The drone follows an elongated oval at race-typical speed; yaw aligns
    with the velocity vector (tangent-to-path); altitude oscillates
    slightly. This stands in for the kind of trajectory a G&CNet would
    produce when racing through a track of moderate density.

    Returns:
        (T, 7) poses [x, y, z, qw, qx, qy, qz], dt
    """
    n_steps = int(round(duration_s / dt))
    poses = np.zeros((n_steps, 7), dtype=np.float64)

    # Estimate angular rate for the target tangential speed: v = ω · sqrt(0.5(rx²+ry²))
    # Approximate radius for angular-velocity estimate.
    eff_radius = math.sqrt(0.5 * (radius_x ** 2 + radius_y ** 2))
    omega = speed_target_mps / eff_radius     # rad/s
    altitude_freq_hz = 0.3                    # gentle altitude oscillation

    for i in range(n_steps):
        t = i * dt
        theta = omega * t
        x = radius_x * math.cos(theta)
        y = radius_y * math.sin(theta)
        z = altitude + altitude_amplitude_m * math.sin(2 * math.pi * altitude_freq_hz * t)

        # Velocity vector (tangent to path).
        vx = -radius_x * omega * math.sin(theta)
        vy = radius_y * omega * math.cos(theta)
        yaw = math.atan2(vy, vx)
        quat = _yaw_to_quat_wxyz(yaw)

        poses[i, 0:3] = [x, y, z]
        poses[i, 3:7] = quat

    return poses, dt


# --------------------------------------------------------------------- runner


def run_profile(
    profile_name: str,
    gt_poses: np.ndarray,
    dt: float,
    n_seeds: int,
    out_dir: Path,
) -> dict:
    """Run synthetic VIO over n_seeds, aggregate per-step drift."""
    raw_dir = out_dir / profile_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    T = gt_poses.shape[0]
    pos_err_per_step = np.zeros((n_seeds, T), dtype=np.float64)
    yaw_err_per_step = np.zeros((n_seeds, T), dtype=np.float64)
    summaries = []

    for s in range(n_seeds):
        vio = SyntheticVIO(profile=profile_name, seed=s)
        est = vio.run(gt_poses, dt=dt)
        drift = vio.compute_drift(est, gt_poses)
        pos_err_per_step[s] = drift["pos_err_m_per_step"]
        yaw_err_per_step[s] = drift["yaw_err_rad_per_step"]
        summaries.append({
            "seed": s,
            "mean_pos_err_m": drift["mean_pos_err_m"],
            "p99_pos_err_m": drift["p99_pos_err_m"],
            "max_pos_err_m": drift["max_pos_err_m"],
            "final_pos_err_m": drift["final_pos_err_m"],
            "mean_yaw_err_rad": drift["mean_yaw_err_rad"],
            "max_yaw_err_rad": drift["max_yaw_err_rad"],
            "final_yaw_err_rad": drift["final_yaw_err_rad"],
        })
        np.savez(
            raw_dir / f"seed_{s:03d}.npz",
            est_poses=est, gt_poses=gt_poses,
            pos_err=drift["pos_err_m_per_step"],
            yaw_err=drift["yaw_err_rad_per_step"],
        )

    # Aggregate across seeds.
    aggregate = {
        "profile": profile_name,
        "n_seeds": n_seeds,
        "duration_s": T * dt,
        "dt": dt,
        # Per-step statistics across seeds — mean and 1σ for each timestep.
        "pos_err_per_step_mean": pos_err_per_step.mean(axis=0).tolist(),
        "pos_err_per_step_std": pos_err_per_step.std(axis=0).tolist(),
        "yaw_err_per_step_mean": yaw_err_per_step.mean(axis=0).tolist(),
        "yaw_err_per_step_std": yaw_err_per_step.std(axis=0).tolist(),
        # Cross-seed summary statistics.
        "across_seeds": {
            "mean_pos_err_m": float(np.mean([s["mean_pos_err_m"] for s in summaries])),
            "p99_pos_err_m_avg": float(np.mean([s["p99_pos_err_m"] for s in summaries])),
            "p99_pos_err_m_max": float(np.max([s["p99_pos_err_m"] for s in summaries])),
            "max_pos_err_m_avg": float(np.mean([s["max_pos_err_m"] for s in summaries])),
            "final_pos_err_m_avg": float(np.mean([s["final_pos_err_m"] for s in summaries])),
            "final_pos_err_m_p99": float(np.percentile(
                [s["final_pos_err_m"] for s in summaries], 99)),
            "mean_yaw_err_deg": math.degrees(
                float(np.mean([s["mean_yaw_err_rad"] for s in summaries]))),
            "max_yaw_err_deg": math.degrees(
                float(np.max([s["max_yaw_err_rad"] for s in summaries]))),
            "final_yaw_err_deg_avg": math.degrees(
                float(np.mean([s["final_yaw_err_rad"] for s in summaries]))),
        },
        "per_seed": summaries,
    }
    return aggregate


# --------------------------------------------------------------------- plots


def make_plots(results: dict[str, dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile_colors = {
        "orb_slam3_with_map_matching": "#7ee787",
        "orb_slam3_mono_inertial": "#ffa657",
        "imu_dead_reckoning": "#ff7b72",
    }
    profile_labels = {
        "orb_slam3_with_map_matching": "ORB-SLAM3-style + map matching",
        "orb_slam3_mono_inertial": "ORB-SLAM3-style, no map matching",
        "imu_dead_reckoning": "IMU dead reckoning (no VO)",
    }

    THEME_BG = "#0d1117"; THEME_SURF = "#161b22"; THEME_TEXT = "#c9d1d9"; THEME_GRID = "#30363d"
    plt.rcParams.update({
        "axes.facecolor": THEME_SURF, "figure.facecolor": THEME_BG,
        "axes.edgecolor": THEME_GRID, "axes.labelcolor": THEME_TEXT,
        "xtick.color": THEME_TEXT, "ytick.color": THEME_TEXT,
        "text.color": THEME_TEXT, "axes.grid": True, "grid.color": THEME_GRID,
        "grid.alpha": 0.3,
    })

    # 1) Drift curves over time (mean ± std band).
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for profile_name, agg in results.items():
        if profile_name not in profile_colors:
            continue
        T = len(agg["pos_err_per_step_mean"])
        t_arr = np.arange(T) * agg["dt"]
        c = profile_colors[profile_name]
        label = profile_labels[profile_name]
        # Position
        m = np.array(agg["pos_err_per_step_mean"])
        s = np.array(agg["pos_err_per_step_std"])
        axes[0].plot(t_arr, m, color=c, linewidth=1.6, label=label)
        axes[0].fill_between(t_arr, m - s, m + s, color=c, alpha=0.2)
        # Yaw (degrees for readability)
        m_y = np.array(agg["yaw_err_per_step_mean"]) * 180 / math.pi
        s_y = np.array(agg["yaw_err_per_step_std"]) * 180 / math.pi
        axes[1].plot(t_arr, m_y, color=c, linewidth=1.6, label=label)
        axes[1].fill_between(t_arr, m_y - s_y, m_y + s_y, color=c, alpha=0.2)
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("position drift (m)")
    axes[0].set_title("Position drift over 30 s race")
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[0].legend(facecolor=THEME_SURF, edgecolor=THEME_GRID, labelcolor=THEME_TEXT, fontsize=9)
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("yaw drift (deg)")
    axes[1].set_title("Yaw drift over 30 s race")
    axes[1].set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "drift_curves.png", dpi=110, bbox_inches="tight",
                facecolor=THEME_BG)
    plt.close(fig)

    # 2) Final-step position error histograms across seeds.
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for profile_name, agg in results.items():
        if profile_name not in profile_colors:
            continue
        finals = [s["final_pos_err_m"] for s in agg["per_seed"]]
        ax.hist(finals, bins=15, alpha=0.6, label=profile_labels[profile_name],
                color=profile_colors[profile_name], edgecolor=THEME_GRID)
    ax.set_xlabel("final-step position drift (m)")
    ax.set_ylabel("# seeds")
    ax.set_title(f"Final drift distribution across {agg['n_seeds']} seeds")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.legend(facecolor=THEME_SURF, edgecolor=THEME_GRID, labelcolor=THEME_TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "histograms.png", dpi=110, bbox_inches="tight",
                facecolor=THEME_BG)
    plt.close(fig)


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES.keys()))
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/perception/localization_drift"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.out.resolve()}")

    print(f"\nGenerating reference trajectory ({args.duration_s} s @ {args.dt} s)...")
    gt_poses, dt = parametric_oval_trajectory(
        duration_s=args.duration_s, dt=args.dt,
    )
    print(f"  trajectory: {gt_poses.shape[0]} steps")

    results = {}
    for prof in args.profiles:
        print(f"\n=== {prof} (n_seeds={args.n_seeds}) ===")
        agg = run_profile(prof, gt_poses, dt, n_seeds=args.n_seeds, out_dir=args.out)
        a = agg["across_seeds"]
        print(f"  mean pos drift: {a['mean_pos_err_m']:.3f} m  "
              f"final-avg: {a['final_pos_err_m_avg']:.3f} m  "
              f"final-p99: {a['final_pos_err_m_p99']:.3f} m")
        print(f"  mean yaw drift: {a['mean_yaw_err_deg']:.2f}°  "
              f"final-avg: {a['final_yaw_err_deg_avg']:.2f}°  "
              f"max: {a['max_yaw_err_deg']:.2f}°")
        results[prof] = agg

    print("\nGenerating plots...")
    make_plots(results, args.out)

    summary_path = args.out / "summary.json"
    # Trim per-step arrays from the JSON for size — keep them in npz only.
    json_safe = {}
    for prof, agg in results.items():
        light = {k: v for k, v in agg.items()
                 if k not in ("pos_err_per_step_mean", "pos_err_per_step_std",
                              "yaw_err_per_step_mean", "yaw_err_per_step_std",
                              "per_seed")}
        light["n_seeds"] = agg["n_seeds"]
        light["across_seeds"] = agg["across_seeds"]
        json_safe[prof] = light
    with summary_path.open("w") as f:
        json.dump({
            "trajectory": {
                "duration_s": args.duration_s,
                "dt": args.dt,
                "n_steps": gt_poses.shape[0],
            },
            "results": json_safe,
        }, f, indent=2, default=str)
    print(f"\nSummary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

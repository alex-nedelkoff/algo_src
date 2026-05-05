"""Synthetic visual-inertial odometry for benchmarking.

This is a noise model, not a real VIO library — it generates a plausible
"localized pose" trajectory by perturbing a ground-truth pose stream with
the kind of error a monocular-inertial VIO would produce. Calibrated to
literature norms (see references below) so the resulting drift numbers
inform the augmentation-vs-primary architecture decision honestly even
without a real VIO integration.

Use it like this::

    gt_poses, dt = ground_truth_trajectory(...)         # (T, 7) [x,y,z,qw,qx,qy,qz]
    vio = SyntheticVIO(profile="orb_slam3_mono_inertial")
    est_poses = vio.run(gt_poses, dt)                   # (T, 7)
    drift = vio.compute_drift(est_poses, gt_poses)

Why we built this instead of integrating real VIO right now: the install
+ wrap of any of the standard libraries (ORB-SLAM3, OpenVINS, DROID-SLAM,
DPVO) is a multi-day Windows-build adventure. The architecture decision
the benchmark unblocks (does vision augment a map-matched VIO, or does it
become primary navigation?) only needs realistic *order-of-magnitude*
drift numbers, not measurements from a specific implementation. We
document the noise model and sources so anyone can replace it later.

Noise model (per-step, in body frame for orientation, world frame for
position):

  Position:
    - Random walk on velocity error: σ_v_per_root_s · sqrt(dt) per step
    - Gaussian noise on position: σ_p (small instantaneous)
    - Periodic map-matching correction (ICP every map_match_interval_s):
      pulls position estimate toward truth by map_match_strength

  Orientation:
    - Random walk on yaw error: σ_yaw_per_root_s · sqrt(dt)
    - Roll/pitch nearly bias-free thanks to gravity-vector observability
      via accelerometer; we model them as bounded Gaussian noise

Profiles cover three regimes:
    "orb_slam3_mono_inertial"
        Conservative state of the art, fully open-loop (no map matching).
        Drift ~5–10 cm/s position, ~0.1°/s yaw.
    "orb_slam3_with_map_matching"
        Same VIO with periodic map-matching. Drift bounded to ~5–15 cm
        position over 30 s. The case we care most about for the qualifier.
    "imu_dead_reckoning"
        IMU integration only, no VO. Drift ~1 m/s² ≈ several m over 30 s.
        Pessimistic baseline for "what if vision dies".

References:
    Campos et al. 2021, ORB-SLAM3 — EuRoC ATE results 0.04–0.10 m over
        ~30 s sequences in mono-inertial mode.
    Geneva et al. 2020, OpenVINS — similar 0.05–0.15 m drift bounds.
    Forster et al. 2017, On-Manifold Preintegration — 1% per-second
        position drift is the typical "without map matching" scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class VIOProfile:
    """Noise hyperparameters for a synthetic VIO regime.

    Attributes:
        sigma_v_per_root_s: random-walk noise on velocity (m/s per √s).
            Translates into position drift growing as σ·√t.
        sigma_p_instant: Gaussian noise on position estimate, m.
            The "high-frequency" part of the noise.
        sigma_yaw_per_root_s: random-walk noise on yaw (rad per √s).
        sigma_pitchroll: bounded Gaussian on pitch/roll, rad.
            Smaller than yaw because the gravity vector is observable.
        map_match_interval_s: how often the map-matcher pulls the
            estimate back toward truth. Set to ``inf`` to disable.
        map_match_strength: fraction of the position error that is
            erased at each map-match update. ``1.0`` = perfect snap to
            truth, ``0.0`` = no correction. Real ICP is somewhere in
            between (0.5–0.9 depending on overlap quality).
        map_match_yaw_strength: same for yaw.
    """
    sigma_v_per_root_s: float
    sigma_p_instant: float
    sigma_yaw_per_root_s: float
    sigma_pitchroll: float
    map_match_interval_s: float
    map_match_strength: float
    map_match_yaw_strength: float


PROFILES: dict[str, VIOProfile] = {
    "orb_slam3_mono_inertial": VIOProfile(
        sigma_v_per_root_s=0.015,         # ≈ 1 % of typical race speeds (1.5 m/s) per √s
        sigma_p_instant=0.005,            # 5 mm per-step jitter
        sigma_yaw_per_root_s=0.002,       # ≈ 0.1°/s drift
        sigma_pitchroll=0.005,            # 0.3° tilt error
        map_match_interval_s=float("inf"),
        map_match_strength=0.0,
        map_match_yaw_strength=0.0,
    ),
    "orb_slam3_with_map_matching": VIOProfile(
        sigma_v_per_root_s=0.015,
        sigma_p_instant=0.005,
        sigma_yaw_per_root_s=0.002,
        sigma_pitchroll=0.005,
        map_match_interval_s=1.0,         # ICP every 1 s
        map_match_strength=0.6,           # 60 % of position error corrected per match
        map_match_yaw_strength=0.4,
    ),
    "imu_dead_reckoning": VIOProfile(
        # Accelerometer-bias-driven double-integration drift dominates.
        # With ~0.01 m/s² typical bias, 30 s of pure integration produces
        # ~ 4 m error; we calibrate σ to give that ballpark final drift.
        sigma_v_per_root_s=0.5,
        sigma_p_instant=0.01,
        sigma_yaw_per_root_s=0.05,        # gyro bias ~0.05 rad/√s
        sigma_pitchroll=0.02,
        map_match_interval_s=float("inf"),
        map_match_strength=0.0,
        map_match_yaw_strength=0.0,
    ),
}


def _yaw_to_quat_wxyz(yaw: float) -> NDArray[np.float64]:
    """Yaw → [w, x, y, z] quaternion (Z-axis rotation only)."""
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _quat_yaw(q: NDArray[np.float64]) -> float:
    """Extract yaw from [w, x, y, z]."""
    qw, qx, qy, qz = q
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SyntheticVIO:
    """Generate a plausible noisy-pose stream from a ground-truth trajectory.

    State carried forward between steps (cumulative random walks):
        - position error vector (3,)
        - yaw error scalar (rad)
    """

    def __init__(
        self,
        profile: Literal[
            "orb_slam3_mono_inertial",
            "orb_slam3_with_map_matching",
            "imu_dead_reckoning",
        ] | VIOProfile = "orb_slam3_with_map_matching",
        seed: int = 0,
    ) -> None:
        if isinstance(profile, str):
            if profile not in PROFILES:
                raise ValueError(f"unknown profile {profile!r}; choices: {list(PROFILES)}")
            self.profile_name = profile
            self.profile = PROFILES[profile]
        else:
            self.profile_name = "custom"
            self.profile = profile
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        gt_poses: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Simulate a VIO run from a ground-truth trajectory.

        Args:
            gt_poses: (T, 7) ground-truth poses [x, y, z, qw, qx, qy, qz].
            dt: timestep in seconds.

        Returns:
            (T, 7) estimated poses.
        """
        if gt_poses.ndim != 2 or gt_poses.shape[1] != 7:
            raise ValueError(f"gt_poses must be (T, 7), got {gt_poses.shape}")

        T = gt_poses.shape[0]
        p = self.profile
        out = np.zeros_like(gt_poses)

        # Cumulative state — error vectors that get corrected at map match.
        pos_err = np.zeros(3, dtype=np.float64)
        yaw_err = 0.0

        # Convert "every X seconds" to "every N steps". `inf` (no map matching)
        # → 0 below, which the loop guard treats as "never trigger".
        if math.isfinite(p.map_match_interval_s) and p.map_match_interval_s > 0:
            steps_per_match = int(round(p.map_match_interval_s / dt))
        else:
            steps_per_match = 0
        sqrt_dt = math.sqrt(dt)

        for t in range(T):
            # Brownian motion on position / yaw. Per-step increment is
            # N(0, σ·√dt) so cumulative variance after T steps is σ²·T·dt =
            # σ²·t and std grows as σ·√t — the standard scaling for
            # bias-corrupted IMU integration before any sensor fusion.
            pos_err += self._rng.normal(0, p.sigma_v_per_root_s * sqrt_dt, size=3)
            yaw_err += float(self._rng.normal(0, p.sigma_yaw_per_root_s * sqrt_dt))

            # Periodic map-match correction — ICP pulls back toward truth
            # by a fraction of the current error. steps_per_match == 0
            # disables matching entirely.
            if steps_per_match > 0 and t > 0 and t % steps_per_match == 0:
                pos_err *= 1.0 - p.map_match_strength
                yaw_err *= 1.0 - p.map_match_yaw_strength

            # Per-step "instantaneous" Gaussian noise (high-frequency jitter).
            pos_jitter = self._rng.normal(0, p.sigma_p_instant, size=3)
            pitch_jitter = float(self._rng.normal(0, p.sigma_pitchroll))
            roll_jitter = float(self._rng.normal(0, p.sigma_pitchroll))

            est_pos = gt_poses[t, 0:3] + pos_err + pos_jitter
            true_yaw = _quat_yaw(gt_poses[t, 3:7])
            est_yaw = _wrap_pi(true_yaw + yaw_err)
            # Build a quaternion that mostly captures the yaw error;
            # pitch/roll perturbations applied via small-angle approximation.
            yaw_q = _yaw_to_quat_wxyz(est_yaw)
            # Compose with small pitch/roll perturbation quaternion.
            p_quat = np.array([
                math.cos(pitch_jitter / 2.0),
                0.0, math.sin(pitch_jitter / 2.0), 0.0,
            ])
            r_quat = np.array([
                math.cos(roll_jitter / 2.0),
                math.sin(roll_jitter / 2.0), 0.0, 0.0,
            ])
            est_q = _quat_mul(_quat_mul(yaw_q, p_quat), r_quat)
            est_q /= np.linalg.norm(est_q)

            out[t, 0:3] = est_pos
            out[t, 3:7] = est_q

        return out

    @staticmethod
    def compute_drift(
        est_poses: NDArray[np.float64],
        gt_poses: NDArray[np.float64],
    ) -> dict:
        """Per-step + summary drift metrics.

        Returns:
            dict with keys: pos_err_m_per_step (T,), yaw_err_rad_per_step (T,),
            mean_pos_err_m, p99_pos_err_m, max_pos_err_m,
            mean_yaw_err_rad, max_yaw_err_rad, final_pos_err_m,
            final_yaw_err_rad.
        """
        pos_err = np.linalg.norm(est_poses[:, 0:3] - gt_poses[:, 0:3], axis=1)
        yaw_err = np.array([
            abs(_wrap_pi(_quat_yaw(est_poses[t, 3:7]) - _quat_yaw(gt_poses[t, 3:7])))
            for t in range(est_poses.shape[0])
        ])
        return {
            "pos_err_m_per_step": pos_err,
            "yaw_err_rad_per_step": yaw_err,
            "mean_pos_err_m": float(pos_err.mean()),
            "p99_pos_err_m": float(np.percentile(pos_err, 99)),
            "max_pos_err_m": float(pos_err.max()),
            "final_pos_err_m": float(pos_err[-1]),
            "mean_yaw_err_rad": float(yaw_err.mean()),
            "max_yaw_err_rad": float(yaw_err.max()),
            "final_yaw_err_rad": float(yaw_err[-1]),
        }


def _quat_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product of two [w, x, y, z] quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])

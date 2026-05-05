"""Waypoint → Track adapter.

Lets the existing G&CNet (which expects a sequence of gate poses) consume
an arbitrary list of 3-D waypoints, e.g. from a vision-aware planner. Each
waypoint becomes a synthetic ``GateState`` whose orientation is a yaw-only
quaternion derived from the path geometry — so the policy's gate-relative
observation is well-defined even though there's no physical gate.

Why this works without retraining: the G&CNet's input layer is
gate-relative (position-in-current-gate-frame, yaw-error-vs-current-gate,
lookahead-gate poses-in-current-gate-frame). All it needs is *some* gate
pose at each waypoint — the policy doesn't know whether that pose comes
from a real gate or a synthesised one, so substituting waypoints leaves
the obs distribution intact.

Yaw heuristic — "anticipatory tangent":

  yaw_k = atan2(wp_{k + lookahead}.y - wp_k.y, wp_{k + lookahead}.x - wp_k.x)

then EMA-smoothed in the angle domain (with proper wrap-to-±π) so
sharp transitions don't translate to instantaneous yaw demands. Looking
``lookahead`` waypoints ahead (default 2) makes the drone "already turning
toward the *next-next* waypoint" by the time it crosses the current one,
matching how a racing pilot anticipates corners.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from sim.tracks import Track, _yaw_to_quat
from sim.types import GateState


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to [-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_synthetic_yaws(
    waypoints: np.ndarray,
    lookahead: int = 2,
    smoothing_alpha: float = 0.5,
    closed_loop: bool = False,
) -> np.ndarray:
    """Compute one yaw per waypoint via anticipatory lookahead + EMA smoothing.

    Args:
        waypoints: (N, 3) array of waypoint positions.
        lookahead: For waypoint k, yaw points toward wp[k + lookahead].
            Default 2 (look two waypoints ahead). lookahead=1 reduces to
            the convention used by ``build_figure8_track``.
        smoothing_alpha: EMA factor for sequential smoothing in the angle
            domain. ``alpha=1.0`` disables smoothing. Default 0.5 — half
            of each new measurement, half of the previous smoothed value.
        closed_loop: If True, treat the waypoint list as a closed loop
            (last waypoint wraps to first for lookahead). If False, the
            tail of the sequence falls back to "look backward" once
            forward lookahead is exhausted.

    Returns:
        (N,) array of yaws in radians.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError(f"waypoints must be (N, 3), got {waypoints.shape}")
    if lookahead < 1:
        raise ValueError(f"lookahead must be >= 1, got {lookahead}")
    if not 0.0 < smoothing_alpha <= 1.0:
        raise ValueError(f"smoothing_alpha must be in (0, 1], got {smoothing_alpha}")

    n = len(waypoints)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.zeros(1)  # cannot infer direction from a single waypoint

    raw_yaws = np.zeros(n)
    for k in range(n):
        if closed_loop:
            target_idx = (k + lookahead) % n
        else:
            target_idx = min(k + lookahead, n - 1)
            # End-of-sequence: if forward lookahead saturates at k itself,
            # look backward instead so we still have a direction.
            if target_idx == k:
                # Direction of travel arriving at k from k-1 (assumes n >= 2).
                src_idx = max(k - 1, 0)
                dx = waypoints[k, 0] - waypoints[src_idx, 0]
                dy = waypoints[k, 1] - waypoints[src_idx, 1]
                raw_yaws[k] = math.atan2(dy, dx) if (abs(dx) > 1e-9 or abs(dy) > 1e-9) else (raw_yaws[k - 1] if k > 0 else 0.0)
                continue

        dx = float(waypoints[target_idx, 0] - waypoints[k, 0])
        dy = float(waypoints[target_idx, 1] - waypoints[k, 1])
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            # Degenerate (target == current). Inherit the previous yaw if
            # available, otherwise default to 0.
            raw_yaws[k] = raw_yaws[k - 1] if k > 0 else 0.0
        else:
            raw_yaws[k] = math.atan2(dy, dx)

    if smoothing_alpha >= 1.0:
        return raw_yaws

    # EMA smoothing in the angle domain — wrap the *difference*, not the
    # absolute angle, so transitions across ±π don't flip 360°.
    smoothed = np.zeros(n)
    smoothed[0] = raw_yaws[0]
    for k in range(1, n):
        diff = _wrap_pi(raw_yaws[k] - smoothed[k - 1])
        smoothed[k] = _wrap_pi(smoothed[k - 1] + smoothing_alpha * diff)
    return smoothed


def waypoints_to_track(
    waypoints: np.ndarray,
    *,
    yaws: np.ndarray | None = None,
    yaw_lookahead: int = 2,
    yaw_smoothing_alpha: float = 0.5,
    closed_loop: bool = False,
    name: str = "waypoints",
    metadata: dict[str, Any] | None = None,
) -> Track:
    """Build a ``Track`` from a sequence of 3-D waypoints.

    Each waypoint becomes a synthetic ``GateState``. Orientations are
    yaw-only quaternions derived from the path tangent (anticipatory
    lookahead + EMA smoothing) unless an explicit per-waypoint yaw array
    is provided.

    Args:
        waypoints: (N, 3) array of waypoint positions in world frame.
        yaws: Optional (N,) array of explicit yaws. If None, yaws are
            computed via ``compute_synthetic_yaws``.
        yaw_lookahead: Lookahead used by the synthetic yaw heuristic.
        yaw_smoothing_alpha: EMA factor for the synthetic yaw heuristic.
        closed_loop: Whether the waypoint sequence wraps (closed track).
        name: Track name (passed through to ``Track``).
        metadata: Optional metadata dict (passed through to ``Track``).

    Returns:
        A ``Track`` of N ``GateState`` objects compatible with all
        existing eval / policy code.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError(f"waypoints must be (N, 3), got {waypoints.shape}")
    n = len(waypoints)
    if n == 0:
        raise ValueError("waypoints must have at least 1 entry")

    if yaws is None:
        yaws = compute_synthetic_yaws(
            waypoints,
            lookahead=yaw_lookahead,
            smoothing_alpha=yaw_smoothing_alpha,
            closed_loop=closed_loop,
        )
    else:
        yaws = np.asarray(yaws, dtype=np.float64)
        if yaws.shape != (n,):
            raise ValueError(f"yaws must be ({n},), got {yaws.shape}")

    gates = [
        GateState(position=waypoints[k], orientation=_yaw_to_quat(float(yaws[k])))
        for k in range(n)
    ]
    return Track(gates, name=name, metadata=metadata)


def resample_waypoints(
    waypoints: np.ndarray,
    target_spacing: float = 4.0,
    *,
    closed_loop: bool = False,
    method: Literal["linear", "cubic"] = "cubic",
    dense_n: int = 1000,
) -> np.ndarray:
    """Resample a waypoint sequence to approximately uniform arc-length spacing.

    Reason: the G&CNet was trained with mean inter-gate spacing in the
    3–6 m range. Plans from a vision-aware planner can be denser (lots
    of small steps to navigate clutter) or sparser (long unobstructed
    segments). Resampling to ~4 m keeps the lookahead tokens (current
    gate, gate+1, gate+2) at distances the policy expects.

    Args:
        waypoints: (N, 3) input waypoints.
        target_spacing: Desired arc-length distance between consecutive
            output waypoints. The actual spacing rounds slightly up so
            the path length divides evenly. Default 4.0 m.
        closed_loop: If True, treat the waypoint sequence as a closed
            loop (last segment wraps to first). The first waypoint is
            duplicated as the last to make this explicit in the output.
        method: ``"linear"`` walks segments linearly; ``"cubic"`` fits a
            cubic B-spline through all input waypoints (parametric, in 3-D)
            and resamples along that. Cubic is smoother but introduces
            slight overshoot near tight bends. Default ``"cubic"``.
        dense_n: Number of densely-sampled points used to integrate arc
            length on the cubic spline. Higher → more accurate spacing.
            Default 1000. Ignored for ``"linear"``.

    Returns:
        (M, 3) resampled waypoints. ``M = ceil(total_length / target_spacing) + 1``
        for open paths; ``M = ceil(total_length / target_spacing) + 1`` with
        the last point matching the first for closed loops.

    Raises:
        ValueError: if waypoints isn't (N, 3) with N >= 2, or if method
            is unrecognised, or if target_spacing <= 0.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError(f"waypoints must be (N, 3), got {waypoints.shape}")
    n_in = len(waypoints)
    if n_in < 2:
        raise ValueError(f"need at least 2 waypoints to resample, got {n_in}")
    if target_spacing <= 0:
        raise ValueError(f"target_spacing must be > 0, got {target_spacing}")
    if method not in ("linear", "cubic"):
        raise ValueError(f"method must be 'linear' or 'cubic', got {method!r}")

    # Build the source path (points to interpolate along).
    if closed_loop:
        path_points = np.vstack([waypoints, waypoints[0:1]])
    else:
        path_points = waypoints

    if method == "linear" or n_in < 4:
        # splprep needs at least 4 points by default (k=3); fall back to
        # linear if we don't have enough.
        return _resample_linear(path_points, target_spacing)
    return _resample_cubic(path_points, target_spacing, dense_n)


def _resample_linear(path: np.ndarray, target_spacing: float) -> np.ndarray:
    """Walk segments linearly and place waypoints at uniform arc length."""
    seg_vec = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = float(cum_len[-1])
    if total_len < 1e-9:
        # Degenerate: all waypoints at the same point.
        return path[:1].copy()

    n_out = max(2, int(math.ceil(total_len / target_spacing)) + 1)
    target_s = np.linspace(0.0, total_len, n_out)
    # For each target arc-length, locate the segment and lerp.
    out = np.empty((n_out, 3), dtype=np.float64)
    seg_idx = np.searchsorted(cum_len, target_s, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_len) - 1)
    for i, (s, k) in enumerate(zip(target_s, seg_idx)):
        seg_start = cum_len[k]
        seg_length = seg_len[k]
        if seg_length < 1e-12:
            out[i] = path[k]
        else:
            t = (s - seg_start) / seg_length
            out[i] = path[k] + t * seg_vec[k]
    return out


def _resample_cubic(path: np.ndarray, target_spacing: float, dense_n: int) -> np.ndarray:
    """Fit a parametric cubic B-spline and resample at uniform arc length."""
    from scipy.interpolate import splev, splprep

    # Parameter u runs from 0 to 1 along the spline. s=0 → interpolating
    # (no smoothing). per=0 → open, per=1 would close — but our caller
    # already duplicated the first point for closed loops, so per=0 is
    # always correct here.
    tck, _ = splprep(path.T, s=0.0, per=0)

    # Densely sample the spline to integrate arc length.
    u_dense = np.linspace(0.0, 1.0, dense_n)
    pts_dense = np.asarray(splev(u_dense, tck)).T  # (dense_n, 3)
    seg_dense = np.diff(pts_dense, axis=0)
    seg_len_dense = np.linalg.norm(seg_dense, axis=1)
    cum_len_dense = np.concatenate([[0.0], np.cumsum(seg_len_dense)])
    total_len = float(cum_len_dense[-1])
    if total_len < 1e-9:
        return path[:1].copy()

    n_out = max(2, int(math.ceil(total_len / target_spacing)) + 1)
    target_s = np.linspace(0.0, total_len, n_out)
    # Map each target arc-length to a u via inverse interpolation, then
    # evaluate the spline at those u values.
    target_u = np.interp(target_s, cum_len_dense, u_dense)
    out = np.asarray(splev(target_u, tck)).T  # (n_out, 3)
    return out


def track_to_waypoints(track: Track) -> tuple[np.ndarray, np.ndarray]:
    """Inverse: extract (positions, yaws) from a Track.

    Used for the identity round-trip test — feeding a Track's positions
    + yaws back through ``waypoints_to_track`` should produce a Track
    with identical observations.

    Returns:
        (positions (N, 3), yaws (N,)) tuple.
    """
    n = track.num_gates
    positions = np.zeros((n, 3), dtype=np.float64)
    yaws = np.zeros(n, dtype=np.float64)
    for k, gate in enumerate(track.gates):
        positions[k] = gate.position
        # Yaw extraction from quaternion [w, x, y, z]:
        # yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))
        qw, qx, qy, qz = gate.orientation
        yaws[k] = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return positions, yaws

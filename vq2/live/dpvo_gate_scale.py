"""Observe-only metric scale fitting from GateNet and monocular DPVO poses."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Iterable


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class GateScalePair:
    frame_ns: int
    gate_id: int
    raw_p: Vector3
    metric_p: Vector3
    bracket_before_ns: int
    bracket_after_ns: int


@dataclass(frozen=True)
class GateScaleEstimate:
    gate_id: int
    frame_ns: int
    scale: float | None
    relative_mad: float | None
    baseline_m: float
    pair_count: int
    sample_count: int
    ready: bool


def _vector3(values: Iterable[float], name: str) -> Vector3:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((right - left) ** 2 for left, right in zip(a, b)))


class GateScaleCalibrator:
    """Fit scale from same-gate relative motion at camera-frame timestamps.

    GateNet inference completes well after its source image was received.  The
    calibrator therefore interpolates the DPVO translation at ``frame_ns``
    between bracketing DPVO outputs instead of pairing against wall-clock time.
    Only translation norms enter the fit, so the unknown similarity rotation
    between the two coordinate systems does not bias scale.
    """

    def __init__(
        self,
        *,
        min_samples: int = 8,
        min_baseline_m: float = 1.0,
        min_pair_baseline_m: float = 0.4,
        max_relative_mad: float = 0.10,
        max_bracket_ns: int = 250_000_000,
        max_history: int = 600,
    ) -> None:
        if min_samples < 2:
            raise ValueError("min_samples must be at least two")
        if min_baseline_m <= 0.0 or min_pair_baseline_m <= 0.0:
            raise ValueError("baseline thresholds must be positive")
        if max_relative_mad <= 0.0:
            raise ValueError("max_relative_mad must be positive")
        if max_bracket_ns < 0 or max_history < 2:
            raise ValueError("history limits are invalid")
        self.min_samples = int(min_samples)
        self.min_baseline_m = float(min_baseline_m)
        self.min_pair_baseline_m = float(min_pair_baseline_m)
        self.max_relative_mad = float(max_relative_mad)
        self.max_bracket_ns = int(max_bracket_ns)
        self.max_history = int(max_history)
        self._poses: list[tuple[int, Vector3]] = []
        self._pending: list[tuple[int, int, Vector3]] = []
        self._seen_dpvo_ns: set[int] = set()
        self._seen_gatenet_ns: set[int] = set()
        self.pairs: list[GateScalePair] = []

    def add_dpvo(
        self, frame_ns: int, raw_p: Iterable[float]
    ) -> list[GateScaleEstimate]:
        frame_ns = int(frame_ns)
        if frame_ns in self._seen_dpvo_ns:
            return []
        raw = _vector3(raw_p, "raw_p")
        self._seen_dpvo_ns.add(frame_ns)
        self._poses.append((frame_ns, raw))
        self._poses.sort(key=lambda item: item[0])
        if len(self._poses) > self.max_history:
            self._poses = self._poses[-self.max_history :]
        return self._drain_pending()

    def add_gatenet(
        self, frame_ns: int, gate_id: int, metric_p: Iterable[float]
    ) -> list[GateScaleEstimate]:
        frame_ns = int(frame_ns)
        if frame_ns in self._seen_gatenet_ns:
            return []
        metric = _vector3(metric_p, "metric_p")
        self._seen_gatenet_ns.add(frame_ns)
        observation = (frame_ns, int(gate_id), metric)
        pair = self._make_pair(observation)
        if pair is None:
            self._pending.append(observation)
            self._trim_pending()
            return []
        self.pairs.append(pair)
        self._trim_pairs()
        return [self.estimate(pair.gate_id, pair.frame_ns)]

    def _interpolate(
        self, frame_ns: int
    ) -> tuple[Vector3, int, int] | None:
        if not self._poses:
            return None
        for pose_ns, pose in self._poses:
            if pose_ns == frame_ns:
                return pose, pose_ns, pose_ns
        before = None
        after = None
        for sample in self._poses:
            if sample[0] < frame_ns:
                before = sample
                continue
            if sample[0] > frame_ns:
                after = sample
                break
        if before is None or after is None:
            return None
        before_gap = frame_ns - before[0]
        after_gap = after[0] - frame_ns
        if before_gap > self.max_bracket_ns or after_gap > self.max_bracket_ns:
            return None
        span = after[0] - before[0]
        alpha = before_gap / span
        interpolated = tuple(
            left + alpha * (right - left)
            for left, right in zip(before[1], after[1])
        )
        return interpolated, before[0], after[0]

    def _make_pair(
        self, observation: tuple[int, int, Vector3]
    ) -> GateScalePair | None:
        frame_ns, gate_id, metric = observation
        interpolated = self._interpolate(frame_ns)
        if interpolated is None:
            return None
        raw, before_ns, after_ns = interpolated
        return GateScalePair(
            frame_ns=frame_ns,
            gate_id=gate_id,
            raw_p=raw,
            metric_p=metric,
            bracket_before_ns=before_ns,
            bracket_after_ns=after_ns,
        )

    def _drain_pending(self) -> list[GateScaleEstimate]:
        retained = []
        estimates = []
        newest_pose_ns = self._poses[-1][0]
        for observation in self._pending:
            pair = self._make_pair(observation)
            if pair is not None:
                self.pairs.append(pair)
                estimates.append(self.estimate(pair.gate_id, pair.frame_ns))
            elif observation[0] >= newest_pose_ns:
                retained.append(observation)
            # Once a later pose exists but the bracket is too wide, a tighter
            # sample can no longer arrive in this forward-only live stream.
        self._pending = retained
        self._trim_pairs()
        return estimates

    def _trim_pending(self) -> None:
        if len(self._pending) > self.max_history:
            self._pending = self._pending[-self.max_history :]

    def _trim_pairs(self) -> None:
        if len(self.pairs) > self.max_history:
            self.pairs = self.pairs[-self.max_history :]

    def estimate(self, gate_id: int, frame_ns: int = 0) -> GateScaleEstimate:
        samples = [pair for pair in self.pairs if pair.gate_id == int(gate_id)]
        ratio_records = []
        observed_baseline_m = 0.0
        for left_index, left in enumerate(samples):
            for right_index, right in enumerate(
                samples[left_index + 1 :], start=left_index + 1
            ):
                metric_distance = _distance(left.metric_p, right.metric_p)
                observed_baseline_m = max(observed_baseline_m, metric_distance)
                if metric_distance < self.min_pair_baseline_m:
                    continue
                raw_distance = _distance(left.raw_p, right.raw_p)
                if raw_distance <= 1e-9 or not math.isfinite(raw_distance):
                    continue
                ratio = metric_distance / raw_distance
                if math.isfinite(ratio) and ratio > 0.0:
                    ratio_records.append(
                        (ratio, left_index, right_index, metric_distance)
                    )
        if not ratio_records:
            return GateScaleEstimate(
                gate_id=int(gate_id),
                frame_ns=int(frame_ns),
                scale=None,
                relative_mad=None,
                baseline_m=0.0,
                pair_count=0,
                sample_count=len(samples),
                ready=False,
            )
        ratios = [record[0] for record in ratio_records]
        scale = statistics.median(ratios)
        mad = statistics.median(abs(value - scale) for value in ratios)
        relative_mad = mad / scale

        # A single bad GateNet endpoint creates many mutually dependent pair
        # ratios.  It can yield a low MAD and provide the only apparent metric
        # baseline.  Readiness therefore requires an inlier graph in which at
        # least min_samples distinct observations each agree with two others.
        degrees = [0] * len(samples)
        inlier_edges = []
        for ratio, left_index, right_index, metric_distance in ratio_records:
            if abs(ratio - scale) / scale <= self.max_relative_mad:
                degrees[left_index] += 1
                degrees[right_index] += 1
                inlier_edges.append((left_index, right_index, metric_distance))
        supported = {
            index for index, degree in enumerate(degrees) if degree >= 2
        }
        baseline_m = max(
            (
                distance
                for left_index, right_index, distance in inlier_edges
                if left_index in supported and right_index in supported
            ),
            default=0.0,
        )
        ready = (
            len(supported) >= self.min_samples
            and baseline_m >= self.min_baseline_m
            and relative_mad <= self.max_relative_mad
        )
        return GateScaleEstimate(
            gate_id=int(gate_id),
            frame_ns=int(frame_ns),
            scale=scale,
            relative_mad=relative_mad,
            baseline_m=baseline_m,
            pair_count=len(ratios),
            sample_count=len(samples),
            ready=ready,
        )


__all__ = [
    "GateScaleCalibrator",
    "GateScaleEstimate",
    "GateScalePair",
]

"""Independent route-frame helpers for low-memory DPVO guidance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable


Intrinsics = tuple[float, float, float, float]


@dataclass(frozen=True)
class DpvoSessionConfig:
    patches: int = 32
    width: int = 640
    height: int = 360
    stride: int = 2
    intrinsics: Intrinsics = (226.0, 226.0, 319.5, 179.5)
    cuda_fraction: float = 0.48
    # DPVO keyframe-graph bounds. Defaults match dpvo config/default.yaml so an
    # unset config reproduces stock behaviour; smaller windows cap per-frame BA
    # cost so latency stays flat as keyframes accumulate over a flight. Both
    # values affect the optimized trajectory (hence metric scale), so they are
    # part of the calibration identity: change them -> recalibrate scale.
    removal_window: int = 22
    optimization_window: int = 10

    def __post_init__(self) -> None:
        if not 8 <= self.patches <= 96:
            raise ValueError("patches must be in [8, 96]")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if len(self.intrinsics) != 4 or not all(
            math.isfinite(float(value)) and float(value) > 0
            for value in self.intrinsics
        ):
            raise ValueError("intrinsics must contain four positive finite values")
        if not 0.0 < self.cuda_fraction <= 1.0:
            raise ValueError("cuda_fraction must be in (0, 1]")
        if self.optimization_window < 2:
            raise ValueError("optimization_window must be >= 2")
        if self.removal_window < self.optimization_window:
            raise ValueError("removal_window must be >= optimization_window")


def scale_intrinsics(
    intrinsics: Iterable[float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> Intrinsics:
    """Scale fx, fy, cx, cy from source (width, height) to target size."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image dimensions must be positive")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    sx = target_width / source_width
    sy = target_height / source_height
    return fx * sx, fy * sy, cx * sx, cy * sy


def calibration_identity(config: DpvoSessionConfig, model_sha256: str) -> str:
    """Return a stable identity for every setting that affects metric scale."""
    payload = {"config": asdict(config), "model_sha256": str(model_sha256)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fit_tick_scale(
    dpvo_g1: Iterable[float],
    dpvo_g2: Iterable[float],
    world_g1: Iterable[float],
    world_g2: Iterable[float],
) -> float:
    """Fit monocular scale from the known gate-1 to gate-2 displacement."""
    dpvo_delta = [float(b) - float(a) for a, b in zip(dpvo_g1, dpvo_g2)]
    world_delta = [float(b) - float(a) for a, b in zip(world_g1, world_g2)]
    if len(dpvo_delta) != 3 or len(world_delta) != 3:
        raise ValueError("positions must be three-dimensional")
    dpvo_distance = math.sqrt(sum(value * value for value in dpvo_delta))
    world_distance = math.sqrt(sum(value * value for value in world_delta))
    if dpvo_distance <= 1e-9 or not math.isfinite(dpvo_distance):
        raise ValueError("DPVO gate displacement is too small")
    if world_distance <= 0.0 or not math.isfinite(world_distance):
        raise ValueError("world gate displacement must be positive")
    return world_distance / dpvo_distance


@dataclass(frozen=True)
class RouteObservation:
    p: tuple[float, float, float] | None
    healthy: bool
    reason: str
    frame_ns: int


class TickRebasedRoute:
    """Turn monocular DPVO deltas into a judge-anchored route position."""

    def __init__(
        self,
        scale: float,
        gate1_world: Iterable[float],
        max_speed: float,
        camera_alignment: Iterable[Iterable[float]] | None = None,
    ) -> None:
        self.scale = float(scale)
        self.gate1_world = self._vector3(gate1_world, "gate1_world")
        self.max_speed = float(max_speed)
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be positive and finite")
        if not math.isfinite(self.max_speed) or self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive and finite")

        if camera_alignment is None:
            camera_alignment = (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        rows = tuple(tuple(float(value) for value in row) for row in camera_alignment)
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError("camera_alignment must be 3x3")
        if not all(math.isfinite(value) for row in rows for value in row):
            raise ValueError("camera_alignment must be finite")
        self.camera_alignment = rows

        self.raw_origin: tuple[float, float, float] | None = None
        self.yaw_at_tick: float | None = None
        self._last_frame_ns: int | None = None
        self._last_p: tuple[float, float, float] | None = None
        self._failed_reason: str | None = None

    @staticmethod
    def _vector3(values: Iterable[float], name: str) -> tuple[float, float, float]:
        vector = tuple(float(value) for value in values)
        if len(vector) != 3:
            raise ValueError(f"{name} must be three-dimensional")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"{name} must be finite")
        return vector

    def _world_position(
        self, raw_p: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        assert self.raw_origin is not None
        assert self.yaw_at_tick is not None
        raw_delta = tuple(
            self.scale * (value - origin)
            for value, origin in zip(raw_p, self.raw_origin)
        )
        aligned = tuple(
            sum(row[index] * raw_delta[index] for index in range(3))
            for row in self.camera_alignment
        )
        cosine = math.cos(self.yaw_at_tick)
        sine = math.sin(self.yaw_at_tick)
        rotated = (
            cosine * aligned[0] - sine * aligned[1],
            sine * aligned[0] + cosine * aligned[1],
            aligned[2],
        )
        return tuple(
            origin + delta for origin, delta in zip(self.gate1_world, rotated)
        )

    def observe(
        self,
        raw_p: Iterable[float],
        frame_ns: int,
        gate_idx: int,
        yaw: float,
    ) -> RouteObservation:
        try:
            raw = self._vector3(raw_p, "raw_p")
        except (TypeError, ValueError):
            self._failed_reason = "nonfinite"
            return RouteObservation(None, False, "nonfinite", int(frame_ns))
        frame_ns = int(frame_ns)
        if self._failed_reason is not None:
            return RouteObservation(None, False, self._failed_reason, frame_ns)
        if self.raw_origin is None:
            if int(gate_idx) < 1:
                return RouteObservation(None, False, "await_tick", frame_ns)
            if not math.isfinite(float(yaw)):
                self._failed_reason = "nonfinite"
                return RouteObservation(None, False, "nonfinite", frame_ns)
            self.raw_origin = raw
            self.yaw_at_tick = float(yaw)
            self._last_frame_ns = frame_ns
            self._last_p = self.gate1_world
            return RouteObservation(self.gate1_world, True, "rebase", frame_ns)

        assert self._last_frame_ns is not None
        assert self._last_p is not None
        if frame_ns <= self._last_frame_ns:
            self._failed_reason = "timestamp"
            return RouteObservation(None, False, "timestamp", frame_ns)
        p = self._world_position(raw)
        dt = (frame_ns - self._last_frame_ns) * 1e-9
        distance = math.sqrt(sum((b - a) ** 2 for a, b in zip(self._last_p, p)))
        if distance / dt > self.max_speed:
            self._failed_reason = "speed"
            return RouteObservation(None, False, "speed", frame_ns)
        self._last_frame_ns = frame_ns
        self._last_p = p
        return RouteObservation(p, True, "ok", frame_ns)


def select_route_position(
    state: dict,
    ticks: int,
    now: float,
    observe_only: bool,
    max_age: float = 0.35,
) -> tuple[tuple[float, float, float] | None, str]:
    """Select DPVO for control only when its independent route pose is usable."""
    if int(ticks) < 1:
        return None, "await_tick"
    if observe_only:
        return None, "observe_only"
    if not state.get("dpvo_route_healthy", False):
        return None, str(state.get("dpvo_route_reason") or "unhealthy")
    try:
        wall = float(state["dpvo_route_wall"])
    except (KeyError, TypeError, ValueError):
        return None, "missing"
    age = float(now) - wall
    if not math.isfinite(age) or age < 0.0:
        return None, "timestamp"
    if age > float(max_age):
        return None, "stale"
    try:
        p = tuple(float(value) for value in state["dpvo_route_p"])
    except (KeyError, TypeError, ValueError):
        return None, "missing"
    if len(p) != 3 or not all(math.isfinite(value) for value in p):
        return None, "nonfinite"
    return p, "ok"



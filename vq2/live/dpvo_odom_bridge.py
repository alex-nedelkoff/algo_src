"""Windows client for independent, tick-rebased WSL2 DPVO route odometry."""

from __future__ import annotations

from collections import deque
import hashlib
import json
import os
import socket
import subprocess
import threading
import time

import numpy as np

try:
    from .dpvo_gate_scale import GateScaleCalibrator
    from .dpvo_bridge_protocol import (
        encode_frame_packet,
        encode_packet,
        parse_json_message,
        recv_packet,
        session_message,
    )
    from .dpvo_route import (
        DpvoSessionConfig,
        RouteObservation,
        TickRebasedRoute,
        calibration_identity,
        scale_intrinsics,
    )
except ImportError:
    from dpvo_gate_scale import GateScaleCalibrator
    from dpvo_bridge_protocol import (
        encode_frame_packet,
        encode_packet,
        parse_json_message,
        recv_packet,
        session_message,
    )
    from dpvo_route import (
        DpvoSessionConfig,
        RouteObservation,
        TickRebasedRoute,
        calibration_identity,
        scale_intrinsics,
    )


PORT = int(os.environ.get("DPVO_BRIDGE_PORT", "9099"))
MODEL = os.environ.get("DPVO_MODEL_WIN", r"C:\Users\alexj\DPVO\dpvo.pth")
FULL_SIZE = (640, 360)
FULL_INTRINSICS = (226.0, 226.0, 319.5, 179.5)


def prewarm_abort_reason(state: dict) -> str | None:
    if state.get("stop"):
        return "stop"
    if state.get("go_passed"):
        return "prewarm_late"
    return None


def wait_for_gpu_handoff(
    state: dict, sleep=time.sleep, poll_s: float = 0.02
) -> str:
    while not state.get("gatenet_unloaded"):
        if state.get("stop"):
            return "stop"
        sleep(poll_s)
    return "handoff"


def config_from_env(environment=None) -> DpvoSessionConfig:
    environment = os.environ if environment is None else environment
    width = int(environment.get("DPVO_WIDTH", "640"))
    height = int(environment.get("DPVO_HEIGHT", "360"))
    intrinsics = scale_intrinsics(FULL_INTRINSICS, FULL_SIZE, (width, height))
    return DpvoSessionConfig(
        patches=int(environment.get("DPVO_PATCHES", "32")),
        width=width,
        height=height,
        stride=int(environment.get("DPVO_STRIDE", "2")),
        intrinsics=intrinsics,
        cuda_fraction=float(environment.get("DPVO_CUDA_FRACTION", "0.48")),
        removal_window=int(environment.get("DPVO_REMOVAL_WINDOW", "22")),
        optimization_window=int(environment.get("DPVO_OPT_WINDOW", "10")),
    )


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ready(reply: dict, config: DpvoSessionConfig, model_sha256: str) -> None:
    if reply.get("type") != "ready":
        raise ValueError(f"DPVO service did not become ready: {reply!r}")
    expected = calibration_identity(config, model_sha256)
    if reply.get("identity") != expected:
        raise ValueError("DPVO ready identity does not match requested session")


def publish_route_observation(state: dict, observation, wall: float) -> None:
    state["dpvo_route_p"] = observation.p
    state["dpvo_route_wall"] = float(wall)
    state["dpvo_route_healthy"] = bool(observation.healthy)
    state["dpvo_route_reason"] = str(observation.reason)
    state["dpvo_route_frame_ns"] = int(observation.frame_ns)


def track_during_gatenet(environment=None) -> bool:
    """Permit GPU overlap only for the explicit telemetry-only experiment."""
    environment = os.environ if environment is None else environment
    return (
        environment.get("GNSCALE") == "1"
        and environment.get("DPVO_OBSERVE") == "1"
    )


def publish_gate_scale(state: dict, estimate, wall: float) -> None:
    """Publish diagnostics without changing route scale or control locks."""
    state["dpvo_gate_scale"] = estimate.scale
    state["dpvo_gate_scale_relative_mad"] = estimate.relative_mad
    state["dpvo_gate_scale_baseline_m"] = float(estimate.baseline_m)
    state["dpvo_gate_scale_pair_count"] = int(estimate.pair_count)
    state["dpvo_gate_scale_sample_count"] = int(estimate.sample_count)
    state["dpvo_gate_scale_gate"] = int(estimate.gate_id)
    state["dpvo_gate_scale_frame_ns"] = int(estimate.frame_ns)
    state["dpvo_gate_scale_ready"] = bool(estimate.ready)
    state["dpvo_gate_scale_wall"] = float(wall)


def read_dpvo_frame(state: dict, last_frame_ns: int):
    """Return one coherent camera timestamp/JPEG snapshot, if it is new."""
    sample = state.get("dpvo_frame")
    if not isinstance(sample, tuple) or len(sample) != 2:
        return None
    try:
        frame_ns = int(sample[0])
    except (TypeError, ValueError):
        return None
    jpeg = sample[1]
    if (
        frame_ns <= 0
        or frame_ns == int(last_frame_ns)
        or not isinstance(jpeg, (bytes, bytearray))
    ):
        return None
    return frame_ns, jpeg


def update_gate_scale(
    calibrator,
    state: dict,
    frame_ns: int,
    raw_p,
    last_gatenet_ns: int,
):
    """Ingest one DPVO pose and a coherent new GateNet state snapshot."""
    pair_count_before = len(calibrator.pairs)
    estimates = list(calibrator.add_dpvo(frame_ns, raw_p))
    sample = state.get("gatenet_sample")
    try:
        gatenet_ns = int(sample[0])
        gate_id = int(sample[1])
        metric_p = tuple(float(value) for value in sample[2])
    except (IndexError, KeyError, TypeError, ValueError):
        gatenet_ns = 0
    if gatenet_ns > 0 and gatenet_ns != int(last_gatenet_ns):
        try:
            new_estimates = calibrator.add_gatenet(
                gatenet_ns, gate_id, metric_p
            )
        except (TypeError, ValueError):
            pass
        else:
            last_gatenet_ns = gatenet_ns
            estimates.extend(new_estimates)
    pairs = tuple(calibrator.pairs[pair_count_before:])
    return int(last_gatenet_ns), pairs, tuple(estimates)


def observe_route_pose(
    route,
    pose_history,
    raw_p,
    frame_ns: int,
    gate_idx: int,
    yaw: float,
    tick_ns: int,
    tolerance_ns: int = 150_000_000,
):
    """Rebase from the pose nearest the judge tick, then process current pose."""
    if gate_idx < 1 or route.raw_origin is not None:
        return route.observe(raw_p, frame_ns, gate_idx, yaw)
    if tick_ns <= 0:
        return RouteObservation(None, False, "tick_timestamp", frame_ns)
    sample = min(pose_history, key=lambda entry: abs(entry[0] - tick_ns))
    if abs(sample[0] - tick_ns) > tolerance_ns:
        return RouteObservation(None, False, "tick_alignment", frame_ns)
    rebased = route.observe(sample[1], sample[0], gate_idx, sample[2])
    if not rebased.healthy or sample[0] == frame_ns:
        return rebased
    return route.observe(raw_p, frame_ns, gate_idx, yaw)


def load_calibration(path: str, config: DpvoSessionConfig, model_sha256: str) -> dict:
    with open(path, "r", encoding="utf-8") as source:
        calibration = json.load(source)
    expected = calibration_identity(config, model_sha256)
    if calibration.get("identity") != expected:
        raise ValueError("calibration identity does not match live DPVO session")
    scale = float(calibration["scale"])
    gate1_world = tuple(float(value) for value in calibration["gate1_world"])
    if scale <= 0.0 or len(gate1_world) != 3:
        raise ValueError("invalid DPVO calibration")
    return {**calibration, "scale": scale, "gate1_world": gate1_world}


def ensure_control_approved(calibration: dict, observe_only: bool) -> None:
    if not observe_only and not calibration.get("control_approved", False):
        raise ValueError(
            "DPVO calibration is observe-only; offline stability validation failed"
        )


def _wsl_ip() -> str:
    # DPVO_BRIDGE_HOST (07-17): the bridge service may run on a REMOTE Linux
    # host (Vagon has no WSL -- e.g. the laptop's WSL2 build over Tailscale).
    # Explicit host wins; local WSL discovery is the laptop-era fallback.
    host = os.environ.get("DPVO_BRIDGE_HOST", "").strip()
    if host:
        return host
    output = subprocess.check_output(
        ["wsl", "-d", "Ubuntu", "hostname", "-I"],
        stderr=subprocess.DEVNULL,
    ).decode()
    addresses = output.split()
    if not addresses:
        raise RuntimeError("WSL2 returned no network address")
    return addresses[0]


class DpvoOdom(threading.Thread):
    """Publish an independent route pose; never read from or update the KF."""

    def __init__(self, state, kf, kf_lock, m_body_cam, jlog=None):
        super().__init__(daemon=True)
        self.state = state
        self.m_body_cam = np.asarray(m_body_cam, dtype=float)
        self.jlog = jlog or (lambda *args, **kwargs: None)
        self.config = None
        self.model_sha256 = None
        self.route = None
        self.sock = None
        self.ready_wall = None
        self.prewarm_start_wall = None

    def _latch_failure(self, reason: str) -> None:
        self.state["dpvo_route_healthy"] = False
        self.state["dpvo_route_reason"] = str(reason)

    def _prepare(self) -> None:
        self.config = config_from_env()
        self.model_sha256 = hash_file(MODEL)
        calibration_path = os.environ.get("DPVO_CAL")
        if not calibration_path:
            raise ValueError("DPVO_CAL is required for tick-rebased route odometry")
        calibration = load_calibration(
            calibration_path, self.config, self.model_sha256
        )
        ensure_control_approved(
            calibration, os.environ.get("DPVO_OBSERVE") == "1"
        )
        self.route = TickRebasedRoute(
            scale=calibration["scale"],
            gate1_world=calibration["gate1_world"],
            max_speed=float(os.environ.get("DPVO_MAX_SPEED", "8.0")),
            camera_alignment=self.m_body_cam,
        )

    def _prewarm(self) -> bool:
        started = time.time()
        self.prewarm_start_wall = started
        self.jlog(
            "dpvo_prewarm_start",
            race_ms=int(self.state.get("race_ms", 0)),
            go_passed=bool(self.state.get("go_passed")),
        )
        reason = prewarm_abort_reason(self.state)
        if reason is not None:
            self._latch_failure(reason)
            self.jlog("dpvo_prewarm_failed", reason=reason)
            return False
        if not self._connect(stop_at_go=True):
            return False
        self.jlog(
            "dpvo_prewarm_ready",
            ready_wall=self.ready_wall,
            init_ms=round((self.ready_wall - started) * 1000.0, 1),
        )
        return True

    def _connect(self, stop_at_go: bool = False) -> bool:
        host = _wsl_ip()
        error = None
        for _ in range(30):
            if stop_at_go:
                reason = prewarm_abort_reason(self.state)
                if reason is not None:
                    self._latch_failure(reason)
                    self.jlog("dpvo_prewarm_failed", reason=reason)
                    return False
            sock = None
            try:
                sock = socket.socket()
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(1.0)
                sock.connect((host, PORT))
                sock.sendall(
                    encode_packet(session_message(self.config, self.model_sha256))
                )
                sock.settimeout(0.25 if stop_at_go else 15.0)
                while True:
                    try:
                        payload = recv_packet(sock)
                        break
                    except socket.timeout:
                        reason = prewarm_abort_reason(self.state)
                        if stop_at_go and reason is not None:
                            sock.close()
                            self._latch_failure(reason)
                            self.jlog("dpvo_prewarm_failed", reason=reason)
                            return False
                if payload is None:
                    raise ConnectionError("DPVO service closed during handshake")
                reply = parse_json_message(payload)
                validate_ready(reply, self.config, self.model_sha256)
                reason = prewarm_abort_reason(self.state)
                if stop_at_go and reason is not None:
                    sock.close()
                    self._latch_failure(reason)
                    self.jlog("dpvo_prewarm_failed", reason=reason)
                    return False
                ready_wall = time.time()
                self.ready_wall = ready_wall
                self.state["dpvo_prewarm_ready"] = True
                self.state["dpvo_prewarm_ready_wall"] = ready_wall
                sock.settimeout(15.0)
                self.sock = sock
                self.jlog(
                    "dpvo_ready",
                    identity=reply["identity"],
                    patches=self.config.patches,
                    intrinsics=list(self.config.intrinsics),
                    ready_wall=ready_wall,
                    init_ms=None if self.prewarm_start_wall is None else round(
                        (ready_wall - self.prewarm_start_wall) * 1000.0, 1
                    ),
                    vram_alloc_mb=reply.get("vram_alloc_mb"),
                    vram_reserved_mb=reply.get("vram_reserved_mb"),
                    cuda_free_mb=reply.get("cuda_free_mb"),
                )
                print(
                    f"DPVO bridge ready {host}:{PORT}, "
                    f"patches={self.config.patches}",
                    flush=True,
                )
                return True
            except (OSError, ValueError) as caught:
                error = caught
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                time.sleep(0.1 if stop_at_go else 1.0)
        raise ConnectionError(f"DPVO bridge unavailable: {error!r}")
    def _step(self, frame_ns: int, jpeg: bytes) -> dict:
        self.sock.sendall(encode_packet(encode_frame_packet(frame_ns, jpeg)))
        payload = recv_packet(self.sock)
        if payload is None:
            raise ConnectionError("DPVO service closed")
        reply = parse_json_message(payload)
        if reply.get("type") == "error":
            raise RuntimeError(reply.get("error", "DPVO service error"))
        if reply.get("type") != "pose" or int(reply["frame_ns"]) != frame_ns:
            raise ValueError("DPVO pose reply does not match submitted frame")
        return reply

    def run(self) -> None:
        try:
            self._run()
        except Exception as error:
            import traceback

            self._latch_failure("service")
            print(f"DPVO bridge thread died: {error!r}", flush=True)
            traceback.print_exc()
            self.jlog("dpvo_died", err=repr(error))
        finally:
            try:
                if self.sock is not None:
                    self.sock.sendall(encode_packet(b""))
                    self.sock.close()
            except OSError:
                pass

    def _track(self) -> None:
        state = self.state
        last_ns = 0
        submitted = 0
        first_pose_logged = False
        alignment_logged = False
        pose_history = deque(maxlen=300)
        gate_scale = (
            GateScaleCalibrator() if track_during_gatenet() else None
        )
        last_gatenet_ns = 0
        if gate_scale is not None:
            state["dpvo_gate_scale_mode"] = "observe"
        while not state.get("stop"):
            frame = read_dpvo_frame(state, last_ns)
            if frame is None:
                time.sleep(0.004)
                continue
            frame_ns, jpeg = frame
            last_ns = frame_ns
            submitted += 1
            if submitted % self.config.stride:
                continue
            reply = self._step(frame_ns, jpeg)
            if not reply.get("ok"):
                continue
            raw_p = tuple(float(value) for value in reply["p"])
            yaw = float(state.get("yaw", 0.0))
            pose_history.append((frame_ns, raw_p, yaw))
            if not first_pose_logged:
                first_pose_logged = True
                wall = time.time()
                self.jlog(
                    "dpvo_first_pose",
                    frame_ns=frame_ns,
                    ready_to_pose_ms=None if self.ready_wall is None else round(
                        (wall - self.ready_wall) * 1000.0, 1
                    ),
                )

            if gate_scale is not None:
                last_gatenet_ns, new_pairs, estimates = update_gate_scale(
                    gate_scale, state, frame_ns, raw_p, last_gatenet_ns
                )
                for pair in new_pairs:
                    self.jlog(
                        "dpvo_gate_scale_pair",
                        frame_ns=pair.frame_ns,
                        gate=pair.gate_id,
                        raw=list(pair.raw_p),
                        metric=list(pair.metric_p),
                        before_ms=round(
                            (pair.frame_ns - pair.bracket_before_ns) / 1e6, 3
                        ),
                        after_ms=round(
                            (pair.bracket_after_ns - pair.frame_ns) / 1e6, 3
                        ),
                    )
                for estimate in estimates:
                    publish_gate_scale(state, estimate, time.time())
                    self.jlog(
                        "dpvo_gate_scale_fit",
                        frame_ns=estimate.frame_ns,
                        gate=estimate.gate_id,
                        scale=None if estimate.scale is None else round(
                            estimate.scale, 6
                        ),
                        relative_mad=None
                        if estimate.relative_mad is None
                        else round(estimate.relative_mad, 6),
                        baseline_m=round(estimate.baseline_m, 3),
                        pair_count=estimate.pair_count,
                        sample_count=estimate.sample_count,
                        ready=estimate.ready,
                    )
                    if (
                        estimate.ready
                        and not state.get("dpvo_gate_scale_announced")
                    ):
                        state["dpvo_gate_scale_announced"] = True
                        print(
                            "DPVO GateNet scale candidate "
                            f"{estimate.scale:.3f} m/unit (observe only)",
                            flush=True,
                        )
                    # GNSCALE APPLY (07-17): feed the GateNet-fitted scale INTO
                    # the route, replacing the per-session-wrong fixed cal.
                    # test5 proved the fixed fg62 scale (9.38) inflates this
                    # session's speeds ~1.5x -> DPVO_MAX_SPEED rejects 50/64
                    # route poses. Guarded: ready + tight MAD + GNSCALE_APPLY.
                    # Still observe-safe -- DPVO_OBSERVE keeps _dpvo_control
                    # False, so the corrected route is LOGGED, not steered;
                    # this measures whether the fit survives the transit.
                    if (
                        estimate.ready
                        and estimate.scale is not None
                        and estimate.relative_mad is not None
                        and estimate.relative_mad <= float(
                            os.environ.get("GNSCALE_APPLY_MAD", "0.15"))
                        and os.environ.get("GNSCALE_APPLY") == "1"
                        and not state.get("dpvo_scale_locked")
                    ):
                        _old = self.route.scale
                        self.route.scale = float(estimate.scale)
                        state["dpvo_scale_locked"] = True
                        self.jlog(
                            "dpvo_scale_applied",
                            scale=round(float(estimate.scale), 6),
                            was=round(float(_old), 6),
                            mad=round(float(estimate.relative_mad), 6),
                            gate=estimate.gate_id,
                            pairs=estimate.pair_count,
                        )
                        print(
                            f"DPVO scale LOCKED from GateNet: "
                            f"{estimate.scale:.3f} m/unit (was {_old:.3f} cal, "
                            f"mad {estimate.relative_mad:.3f})",
                            flush=True,
                        )

            gate_idx = int(state.get("gate_idx", 0))
            tick_ns = int(state.get("gate_tick_ns", 0))
            if gate_idx >= 1 and self.route.raw_origin is None and not alignment_logged:
                alignment_logged = True
                if tick_ns > 0:
                    sample = min(pose_history, key=lambda entry: abs(entry[0] - tick_ns))
                    offset_ms = (sample[0] - tick_ns) / 1e6
                    decision = "aligned" if abs(offset_ms) <= 150.0 else "tick_alignment"
                    pose_ns = sample[0]
                else:
                    offset_ms = None
                    decision = "tick_timestamp"
                    pose_ns = None
                self.jlog(
                    "dpvo_tick_align",
                    tick_ns=tick_ns,
                    pose_ns=pose_ns,
                    offset_ms=offset_ms,
                    decision=decision,
                )

            observation = observe_route_pose(
                self.route,
                pose_history,
                raw_p,
                frame_ns,
                gate_idx,
                yaw,
                tick_ns,
            )
            publish_route_observation(state, observation, time.time())
            self.jlog(
                "dpvo_route",
                ok=observation.healthy,
                reason=observation.reason,
                frame_ns=frame_ns,
                p=None if observation.p is None else list(observation.p),
                raw=list(raw_p),
                dt_ms=reply.get("dt_ms"),
                vram_alloc_mb=reply.get("vram_alloc_mb"),
                vram_reserved_mb=reply.get("vram_reserved_mb"),
                cuda_free_mb=reply.get("cuda_free_mb"),
            )

    def _run(self) -> None:
        self._prepare()
        if not self._prewarm():
            return
        if track_during_gatenet():
            print(
                "DPVO observe-only scale tracking starts before race GO",
                flush=True,
            )
            tracking_wall = time.time()
            self.jlog(
                "dpvo_tracking_start",
                overlap=True,
                pre_go=True,
                race_ms=int(self.state.get("race_ms", 0)),
            )
            self._track()
            return
        print("DPVO route thread waiting for GateNet GPU handoff", flush=True)
        outcome = wait_for_gpu_handoff(self.state)
        if outcome != "handoff":
            return
        tracking_wall = time.time()
        handoff_wall = float(
            self.state.get("gatenet_unloaded_wall", tracking_wall)
        )
        self.jlog(
            "dpvo_tracking_start",
            handoff_delay_ms=round((tracking_wall - handoff_wall) * 1000.0, 1),
        )
        self._track()

__all__ = [
    "DpvoOdom",
    "config_from_env",
    "ensure_control_approved",
    "load_calibration",
    "observe_route_pose",
    "prewarm_abort_reason",
    "publish_route_observation",
    "read_dpvo_frame",
    "publish_gate_scale",
    "track_during_gatenet",
    "update_gate_scale",
    "validate_ready",
    "wait_for_gpu_handoff",
]

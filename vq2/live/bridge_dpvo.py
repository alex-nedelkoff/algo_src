#!/usr/bin/env python3
"""Persistent low-memory WSL2 DPVO service for the VQ2 flight process."""

from __future__ import annotations

import hashlib
import os
import socket
import sys
import time

try:
    from .dpvo_bridge_protocol import (
        decode_frame_packet,
        encode_packet,
        json_message,
        parse_session_message,
        recv_packet,
    )
except ImportError:
    from dpvo_bridge_protocol import (
        decode_frame_packet,
        encode_packet,
        json_message,
        parse_session_message,
        recv_packet,
    )


PORT = int(os.environ.get("DPVO_BRIDGE_PORT", "9099"))
DPVO_REPO = os.environ.get("DPVO_REPO", "/home/alexj/DPVO_wsl")
DPVO_CONFIG = os.environ.get(
    "DPVO_CONFIG", "/home/alexj/DPVO_wsl/config/default.yaml"
)
DPVO_MODEL = os.environ.get("DPVO_MODEL", "/mnt/c/Users/alexj/DPVO/dpvo.pth")


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_identity(request, actual_model_sha256: str) -> str:
    if request.model_sha256 != actual_model_sha256:
        raise ValueError("client and service DPVO model hashes differ")
    return request.identity


def _load_runtime(config):
    if DPVO_REPO not in sys.path:
        sys.path.insert(0, DPVO_REPO)
    import cv2
    import numpy as np
    import torch

    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(config.cuda_fraction)

    from dpvo.config import cfg

    cfg.merge_from_file(DPVO_CONFIG)
    cfg.PATCHES_PER_FRAME = config.patches
    # Bound the keyframe graph so per-frame BA cost stays flat over a long
    # flight (offline-measured: 300ms -> 171ms plateau at 12/6 + half-res + p24).
    cfg.REMOVAL_WINDOW = config.removal_window
    cfg.OPTIMIZATION_WINDOW = config.optimization_window
    cfg.MIXED_PRECISION = True
    cfg.LOOP_CLOSURE = False
    cfg.CLASSIC_LOOP_CLOSURE = False

    from dpvo.dpvo import DPVO
    from dpvo.lietorch import SE3

    intrinsics = torch.tensor(config.intrinsics, dtype=torch.float, device="cuda")
    slam = DPVO(
        cfg,
        DPVO_MODEL,
        ht=config.height,
        wd=config.width,
        viz=False,
    )
    return cv2, np, torch, SE3, slam, intrinsics


def _pose_reply(runtime, config, frame_ns: int, jpeg: bytes, frame_index: int):
    cv2, np, torch, SE3, slam, intrinsics = runtime
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG decode failed")
    if image.shape[1] != config.width or image.shape[0] != config.height:
        image = cv2.resize(
            image, (config.width, config.height), interpolation=cv2.INTER_AREA
        )
    started = time.perf_counter()
    tensor = torch.from_numpy(np.ascontiguousarray(image[..., ::-1]))
    tensor = tensor.permute(2, 0, 1).cuda()
    slam(frame_index, tensor, intrinsics)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    reply = {
        "type": "pose",
        "frame_ns": frame_ns,
        "i": frame_index + 1,
        "n": int(slam.n),
        "ok": slam.n >= 9,
        "dt_ms": round(elapsed_ms, 1),
        "vram_alloc_mb": round(torch.cuda.memory_allocated() / 1048576.0, 1),
        "vram_reserved_mb": round(torch.cuda.memory_reserved() / 1048576.0, 1),
        "cuda_free_mb": round(cuda_free / 1048576.0, 1),
        "cuda_total_mb": round(cuda_total / 1048576.0, 1),
    }
    if reply["ok"]:
        pose = SE3(slam.pg.poses_[slam.n - 1]).inv()
        values = pose.data.detach().cpu().numpy().reshape(-1)
        reply["p"] = [round(float(value), 6) for value in values[:3]]
        reply["q"] = [round(float(value), 6) for value in values[3:7]]
    return reply


def serve_connection(connection: socket.socket, actual_model_sha256: str) -> None:
    request_payload = recv_packet(connection)
    if not request_payload:
        raise ValueError("first packet must be a session message")
    request = parse_session_message(request_payload)
    validate_model_identity(request, actual_model_sha256)
    runtime = _load_runtime(request.config)
    _, _, torch, _, slam, _ = runtime
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    connection.sendall(
        encode_packet(
            json_message(
                "ready",
                identity=request.identity,
                patches=request.config.patches,
                width=request.config.width,
                height=request.config.height,
                intrinsics=request.config.intrinsics,
                vram_alloc_mb=round(torch.cuda.memory_allocated() / 1048576.0, 1),
                vram_reserved_mb=round(torch.cuda.memory_reserved() / 1048576.0, 1),
                cuda_free_mb=round(cuda_free / 1048576.0, 1),
                cuda_total_mb=round(cuda_total / 1048576.0, 1),
            )
        )
    )
    frame_index = 0
    try:
        while True:
            payload = recv_packet(connection)
            if payload is None or payload == b"":
                break
            frame_ns, jpeg = decode_frame_packet(payload)
            reply = _pose_reply(
                runtime, request.config, frame_ns, jpeg, frame_index
            )
            frame_index += 1
            message_type = reply.pop("type")
            connection.sendall(
                encode_packet(json_message(message_type, **reply))
            )
    finally:
        del slam
        torch.cuda.empty_cache()


def main() -> None:
    actual_model_sha256 = hash_file(DPVO_MODEL)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen(1)
    print(f"DPVO bridge listening on 0.0.0.0:{PORT}", flush=True)
    while True:
        connection, address = server.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(15.0)
        print(f"client connected {address}", flush=True)
        try:
            serve_connection(connection, actual_model_sha256)
        except Exception as error:
            print(f"connection error: {error!r}", flush=True)
            try:
                connection.sendall(
                    encode_packet(json_message("error", error=str(error)))
                )
            except OSError:
                pass
        finally:
            connection.close()
            print("client disconnected; DPVO reset", flush=True)


if __name__ == "__main__":
    main()

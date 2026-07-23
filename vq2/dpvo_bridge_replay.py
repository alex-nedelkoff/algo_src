"""Read-only corpus replay client for the WSL2 DPVO bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys

from .live.dpvo_bridge_protocol import (
    encode_frame_packet,
    encode_packet,
    parse_json_message,
    recv_packet,
    session_message,
)
from .live.dpvo_route import (
    FULL_INTRINSICS,
    DpvoSessionConfig,
    calibration_identity,
    scale_intrinsics,
)


FULL_SIZE = (640, 360)


def select_frame_rows(rows, stride: int, block_gap_s: float = 5.0) -> list[dict]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    rows = list(rows)
    if not rows:
        return []
    block_start = 0
    for index in range(1, len(rows)):
        if float(rows[index]["rx_wall"]) - float(rows[index - 1]["rx_wall"]) > block_gap_s:
            block_start = index
    return rows[block_start::stride]


def pose_record(reply: dict, identity: str) -> dict:
    if not reply.get("ok") or "p" not in reply:
        raise ValueError("service reply has no usable DPVO pose")
    record = {
        "identity": identity,
        "frame_ns": int(reply["frame_ns"]),
        "p": [float(value) for value in reply["p"]],
    }
    for key in (
        "i",
        "n",
        "dt_ms",
        "vram_alloc_mb",
        "vram_reserved_mb",
        "cuda_free_mb",
        "cuda_total_mb",
    ):
        if key in reply:
            record[key] = reply[key]
    return record


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wsl_ip() -> str:
    output = subprocess.check_output(
        ["wsl", "-d", "Ubuntu", "hostname", "-I"],
        stderr=subprocess.DEVNULL,
    ).decode()
    return output.split()[0]


def replay(
    corpus: Path,
    output: Path,
    config: DpvoSessionConfig,
    model: Path,
    host: str | None = None,
    port: int = 9099,
) -> dict:
    metadata_path = corpus / "frames_dedup.jsonl"
    frames_path = corpus / "frames"
    with metadata_path.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    selected = select_frame_rows(rows, config.stride)
    if not selected:
        raise ValueError("corpus has no replay frames")

    model_sha256 = _hash_file(model)
    identity = calibration_identity(config, model_sha256)
    sock = socket.socket()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(30.0)
    sock.connect((host or _wsl_ip(), port))
    sock.sendall(encode_packet(session_message(config, model_sha256)))
    payload = recv_packet(sock)
    if payload is None:
        raise ConnectionError("service closed during replay handshake")
    ready = parse_json_message(payload)
    if ready.get("type") != "ready" or ready.get("identity") != identity:
        raise ValueError(f"DPVO service rejected replay session: {ready!r}")

    count = 0
    latencies = []
    peak_allocated = float(ready.get("vram_alloc_mb", 0.0))
    peak_reserved = float(ready.get("vram_reserved_mb", 0.0))
    min_cuda_free = float(ready.get("cuda_free_mb", "inf"))
    try:
        with output.open("w", encoding="utf-8", newline="\n") as destination:
            for row in selected:
                frame_ns = int(row["sim_ns"])
                jpeg_path = frames_path / f"{frame_ns}.jpg"
                jpeg = jpeg_path.read_bytes()
                sock.sendall(
                    encode_packet(encode_frame_packet(frame_ns, jpeg))
                )
                payload = recv_packet(sock)
                if payload is None:
                    raise ConnectionError("service closed during replay")
                reply = parse_json_message(payload, "pose")
                peak_allocated = max(
                    peak_allocated, float(reply.get("vram_alloc_mb", 0.0))
                )
                peak_reserved = max(
                    peak_reserved, float(reply.get("vram_reserved_mb", 0.0))
                )
                min_cuda_free = min(
                    min_cuda_free, float(reply.get("cuda_free_mb", "inf"))
                )
                if "dt_ms" in reply:
                    latencies.append(float(reply["dt_ms"]))
                if not reply.get("ok"):
                    continue
                destination.write(
                    json.dumps(pose_record(reply, identity), separators=(",", ":"))
                    + "\n"
                )
                count += 1
    finally:
        try:
            sock.sendall(encode_packet(b""))
        except OSError:
            pass
        sock.close()
    return {
        "identity": identity,
        "frames_submitted": len(selected),
        "poses_written": count,
        "mean_dt_ms": None if not latencies else sum(latencies) / len(latencies),
        "peak_vram_alloc_mb": peak_allocated,
        "peak_vram_reserved_mb": peak_reserved,
        "min_cuda_free_mb": None if math.isinf(min_cuda_free)
        else min_cuda_free,
        "output": str(output),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path(r"C:\Users\alexj\DPVO\dpvo.pth")
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--patches", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--cuda-fraction", type=float, default=0.48)
    parser.add_argument("--removal-window", type=int, default=22)
    parser.add_argument("--optimization-window", type=int, default=10)
    arguments = parser.parse_args(argv)
    intrinsics = scale_intrinsics(
        FULL_INTRINSICS, FULL_SIZE, (arguments.width, arguments.height)
    )
    config = DpvoSessionConfig(
        patches=arguments.patches,
        width=arguments.width,
        height=arguments.height,
        stride=arguments.stride,
        intrinsics=intrinsics,
        cuda_fraction=arguments.cuda_fraction,
        removal_window=arguments.removal_window,
        optimization_window=arguments.optimization_window,
    )
    summary = replay(
        arguments.corpus,
        arguments.output,
        config,
        arguments.model,
        arguments.host,
        arguments.port,
    )
    json.dump(summary, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

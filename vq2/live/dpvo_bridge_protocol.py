"""CUDA-free wire protocol shared by the Windows and WSL DPVO processes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import socket
import struct

try:
    from .dpvo_route import DpvoSessionConfig, calibration_identity
except ImportError:
    from dpvo_route import DpvoSessionConfig, calibration_identity


MAX_PACKET_BYTES = 16 * 1024 * 1024
FRAME_MAGIC = b"FRM1"


@dataclass(frozen=True)
class SessionRequest:
    config: DpvoSessionConfig
    model_sha256: str
    identity: str


def encode_packet(payload: bytes) -> bytes:
    payload = bytes(payload)
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError("packet is too large")
    return struct.pack(">I", len(payload)) + payload


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        data = sock.recv(size - len(chunks))
        if not data:
            return None
        chunks.extend(data)
    return bytes(chunks)


def recv_packet(sock: socket.socket) -> bytes | None:
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    size = struct.unpack(">I", header)[0]
    if size > MAX_PACKET_BYTES:
        raise ValueError("packet is too large")
    if size == 0:
        return b""
    return _recv_exact(sock, size)


def session_message(config: DpvoSessionConfig, model_sha256: str) -> bytes:
    identity = calibration_identity(config, model_sha256)
    payload = {
        "type": "session",
        "config": asdict(config),
        "model_sha256": str(model_sha256),
        "identity": identity,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def parse_session_message(payload: bytes) -> SessionRequest:
    try:
        message = json.loads(payload.decode("utf-8"))
        if message.get("type") != "session":
            raise ValueError
        values = dict(message["config"])
        values["intrinsics"] = tuple(values["intrinsics"])
        config = DpvoSessionConfig(**values)
        model_sha256 = str(message["model_sha256"])
        identity = str(message["identity"])
    except (AttributeError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("first packet must be a valid session message") from None
    except ValueError as error:
        if str(error):
            raise
        raise ValueError("first packet must be a valid session message") from None
    expected = calibration_identity(config, model_sha256)
    if identity != expected:
        raise ValueError("session identity does not match configuration")
    return SessionRequest(config, model_sha256, identity)


def json_message(message_type: str, **values) -> bytes:
    payload = {"type": message_type, **values}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def parse_json_message(payload: bytes, expected_type: str | None = None) -> dict:
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON message") from error
    if not isinstance(message, dict):
        raise ValueError("JSON message must be an object")
    if expected_type is not None and message.get("type") != expected_type:
        raise ValueError(f"expected {expected_type!r} message")
    return message


def encode_frame_packet(frame_ns: int, jpeg: bytes) -> bytes:
    frame_ns = int(frame_ns)
    if frame_ns < 0:
        raise ValueError("frame timestamp must be non-negative")
    return FRAME_MAGIC + struct.pack(">Q", frame_ns) + bytes(jpeg)


def decode_frame_packet(payload: bytes) -> tuple[int, bytes]:
    if len(payload) < 12 or payload[:4] != FRAME_MAGIC:
        raise ValueError("invalid frame packet")
    frame_ns = struct.unpack(">Q", payload[4:12])[0]
    jpeg = payload[12:]
    if not jpeg:
        raise ValueError("frame packet has no JPEG payload")
    return frame_ns, jpeg

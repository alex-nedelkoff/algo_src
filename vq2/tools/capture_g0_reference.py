"""Capture a fresh-reset G0 reference without arming or sending controls.

Run only after the simulator event is live and before any flight command.  The
output is a compact frame/telemetry corpus whose G0 identity follows from the
fresh-reset context, not visual gate appearance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store


def capture(out: Path, seconds: float = 8.0) -> int:
    out.mkdir(parents=True, exist_ok=False)
    frames = out / "frames"
    frames.mkdir()
    store = Store()
    mav = MavlinkIO(store)
    vision = VisionIO(store)
    if not mav.wait_heartbeat(10):
        raise RuntimeError("no simulator heartbeat")
    mav.start()
    vision.start()
    deadline = time.time() + 5.0
    race = None
    while time.time() < deadline:
        race = store.get_race()
        if race is not None:
            break
        time.sleep(0.02)
    if not race or not race.get("race_live") or race.get("active_gate_index") != 0:
        raise RuntimeError(f"require fresh live G0 event, got {race!r}")
    started = time.time()
    last_seq = -1
    seen_ns: set[int] = set()
    rows = []
    while time.time() - started < seconds:
        frame, seq = store.get_frame()
        if frame is None or seq == last_seq:
            time.sleep(0.01)
            continue
        last_seq = seq
        image, frame_ns = frame
        if int(frame_ns) in seen_ns:
            continue
        seen_ns.add(int(frame_ns))
        path = frames / f"{frame_ns}.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write {path}")
        drone = store.get_drone()
        rows.append({
            "frame_ns": int(frame_ns),
            "wall_s": time.time(),
            "race": store.get_race(),
            "drone": None if drone is None else {
                "pos_ned": drone.pos_ned.tolist(), "vel_ned": drone.vel_ned.tolist(),
                "quat_wxyz": drone.quat_wxyz.tolist(), "t_us": int(drone.t_us),
            },
        })
    mav.stop()
    vision.stop()
    (out / "telemetry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "meta.json").write_text(json.dumps({
        "kind": "fresh-reset-g0-reference", "seconds": seconds,
        "n_frames": len(rows), "no_arm": True,
        "association": "G0 from fresh reset / active_gate_index=0",
    }, indent=2))
    print(f"captured {len(rows)} receive-only G0 frames -> {out}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    return 0 if capture(args.out, args.seconds) else 1


if __name__ == "__main__":
    raise SystemExit(main())

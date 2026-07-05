"""Corpus loader for recorded VQ2 raw-sensor runs.

A corpus directory (produced by the laptop recorder scripts) contains:
  mavlink.jsonl       — every MAVLink message in arrival order (+ _rx_wall)
  frames_dedup.jsonl  — one row per unique camera frame: sim_ns, fid, rx_wall
  frames/<sim_ns>.jpg — the frames themselves (motion corpus) or <fid>.jpg (at-rest)
  cmds.jsonl          — commands sent during the run (flight corpora only)

Hard rules encoded here (each one burned us in VQ1/VQ2 bring-up):
  * time_usec RESETS on sim reset — streams are split into segments on stamp
    drops; cross-segment integration is meaningless.
  * duplicate stamps are dropped (recorder can poll faster than telemetry).
  * camera sim_ns is epoch-based, IMU time_usec is boot-based — different
    clocks; rx_wall is the only cross-stream bridge until a proper sync
    estimate exists.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

# A backward stamp jump larger than this starts a new segment (sim reset).
SEGMENT_DROP_US = 1_000_000


@dataclass
class ImuSample:
    t_us: int
    acc: tuple  # (xacc, yacc, zacc) m/s^2, body frame, specific force
    gyr: tuple  # (xgyro, ygyro, zgyro) rad/s, body frame, AS-RECEIVED (no wfix)
    rx_wall: float


@dataclass
class ActuatorSample:
    t_us: int
    rx_wall: float
    raw: dict


@dataclass
class FrameRef:
    sim_ns: int
    rx_wall: float
    path: str


@dataclass
class Segment:
    """One contiguous stamp epoch of the MAVLink streams."""

    imu: list = field(default_factory=list)
    act: list = field(default_factory=list)
    enc: list = field(default_factory=list)  # raw ENCAPSULATED_DATA dicts

    @property
    def duration_s(self) -> float:
        if len(self.imu) < 2:
            return 0.0
        return (self.imu[-1].t_us - self.imu[0].t_us) / 1e6

    def imu_gaps(self, nominal_hz: float = 144.0) -> dict:
        """Stamp-gap stats: how many nominal periods were dropped."""
        if len(self.imu) < 2:
            return {"n": 0}
        period_us = 1e6 / nominal_hz
        gaps = [b.t_us - a.t_us for a, b in zip(self.imu, self.imu[1:])]
        dropped = sum(round(g / period_us) - 1 for g in gaps if g > 1.5 * period_us)
        return {
            "n": len(gaps),
            "median_us": sorted(gaps)[len(gaps) // 2],
            "max_us": max(gaps),
            "dropped_periods": int(dropped),
        }


@dataclass
class DetectionFrame:
    """GateNet output for one camera frame (raw dicts from detections.jsonl)."""

    sim_ns: int
    insts: list


@dataclass
class Corpus:
    root: str
    segments: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    detections: list = field(default_factory=list)

    @property
    def flight_segment(self) -> Optional[Segment]:
        """The last segment is the post-reset run in every corpus so far."""
        return self.segments[-1] if self.segments else None


def load(root: str) -> Corpus:
    corpus = Corpus(root=root)

    seg = Segment()
    last_us = None
    seen_us: set = set()
    with open(os.path.join(root, "mavlink.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            typ = d.get("mavpackettype")
            t_us = d.get("time_usec")
            if t_us is not None:
                if last_us is not None and t_us < last_us - SEGMENT_DROP_US:
                    if seg.imu or seg.act:
                        corpus.segments.append(seg)
                    seg = Segment()
                    seen_us = set()
                last_us = t_us
            if typ == "HIGHRES_IMU":
                key = ("imu", t_us)
                if key in seen_us:
                    continue
                seen_us.add(key)
                seg.imu.append(
                    ImuSample(
                        t_us=t_us,
                        acc=(d["xacc"], d["yacc"], d["zacc"]),
                        gyr=(d["xgyro"], d["ygyro"], d["zgyro"]),
                        rx_wall=d.get("_rx_wall", 0.0),
                    )
                )
            elif typ == "ACTUATOR_OUTPUT_STATUS":
                key = ("act", t_us)
                if key in seen_us:
                    continue
                seen_us.add(key)
                seg.act.append(ActuatorSample(t_us=t_us, rx_wall=d.get("_rx_wall", 0.0), raw=d))
            elif typ == "ENCAPSULATED_DATA":
                seg.enc.append(d)
    if seg.imu or seg.act:
        corpus.segments.append(seg)

    frames_meta = os.path.join(root, "frames_dedup.jsonl")
    fdir = os.path.join(root, "frames")
    seen_paths: set = set()
    if os.path.exists(frames_meta):
        with open(frames_meta) as f:
            for line in f:
                d = json.loads(line)
                # motion corpus names frames <sim_ns>.jpg, at-rest corpus <fid>.jpg
                for candidate in (f"{d['sim_ns']}.jpg", f"{d.get('fid', 0):08d}.jpg"):
                    p = os.path.join(fdir, candidate)
                    if os.path.exists(p):
                        corpus.frames.append(
                            FrameRef(sim_ns=d["sim_ns"], rx_wall=d.get("rx_wall", 0.0), path=p)
                        )
                        seen_paths.add(p)
                        break
    # meta-less frames (the recorder's frames/ dir accumulates across runs while
    # the meta file is truncated per run): index by <sim_ns>.jpg filename,
    # rx_wall unknown (0.0) — usable for detection joins, not for clock bridging
    if os.path.isdir(fdir):
        for fn in os.listdir(fdir):
            p = os.path.join(fdir, fn)
            if p in seen_paths or not fn.endswith(".jpg"):
                continue
            stem = fn[:-4]
            if stem.isdigit() and len(stem) > 12:  # epoch-ns stems only
                corpus.frames.append(FrameRef(sim_ns=int(stem), rx_wall=0.0, path=p))
    corpus.frames.sort(key=lambda fr: fr.sim_ns)

    det_meta = os.path.join(root, "detections.jsonl")
    if os.path.exists(det_meta):
        with open(det_meta) as f:
            for line in f:
                d = json.loads(line)
                corpus.detections.append(DetectionFrame(sim_ns=d["sim_ns"], insts=d["insts"]))
        corpus.detections.sort(key=lambda df: df.sim_ns)
    return corpus

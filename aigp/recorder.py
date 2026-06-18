"""Durable VQ-sim data recorder + loader.

Companion hard rule to the dashboard: every sim run also DURABLY records its
(command, state, IMU, actuator, collision) time-series so we can fit a drone model
and train RL on it later. Data lands under VQ_DATA_ROOT (on disk, OUTSIDE any git
worktree so it survives worktree churn) and is indexed so downstream tools point
at it with ONE path.

Layout:
  $VQ_DATA_ROOT/                  (default: ~/Documents/vq_data; override via env VQ_DATA_ROOT)
    index.jsonl                   one JSON line per run (run_id, path, script, mode, n, t_start, notes)
    <stamp>_<script>/
      data.npz                    compressed aligned arrays (one row per log() call)
      meta.json                   run metadata

data.npz columns (all aligned):
  t_wall (s, our clock from run start), t_us (sim ODOMETRY time_usec),
  cmd (n,K)  = the command we SENT (K=4: motors, or [roll,pitch,yaw rate, thrust], per `mode`),
  pos (n,3), vel (n,3), quat (n,4 wxyz), omega (n,3)   = ODOMETRY,
  acc (n,3), gyro (n,3)                                = HIGHRES_IMU,
  act (n,4)                                            = ACTUATOR_OUTPUT_STATUS,
  coll_seq (n,), gate_idx (n,), live (n,)

Record:   rec = Recorder(store, script="collect_vq", mode="motor", notes="...")
          ... each iter, AFTER sending `cmd`:  rec.log(cmd)
          rec.close()                          # always call (use try/finally)
Load:     from aigp.recorder import load_index, load_run
          runs = load_index();  d = load_run(runs[-1]["path"])   # dict of arrays + d["meta"]
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np


def data_root() -> Path:
    env = os.environ.get("VQ_DATA_ROOT")
    return Path(env) if env else Path(os.path.expanduser("~")) / "Documents" / "vq_data"


VQ_DATA_ROOT = data_root()
_COLS = ["t_wall", "t_us", "cmd", "pos", "vel", "quat", "omega", "acc", "gyro", "act",
         "coll_seq", "gate_idx", "live"]


class Recorder:
    def __init__(self, store, script="run", mode="motor", notes="", root=None,
                 extra_meta=None, flush_every=2000):
        self.store = store
        self.root = Path(root) if root else data_root()
        self.stamp = time.strftime("%Y%m%dT%H%M%S")
        self.run_id = f"{self.stamp}_{script}"
        self.dir = self.root / self.run_id
        self.script, self.mode, self.notes = script, mode, notes
        self.extra_meta = extra_meta or {}
        self.flush_every = int(flush_every)
        self.t0 = time.time()
        self.rows = []
        self.ok = True
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[recorder] disabled ({e})", flush=True)
            self.ok = False

    def log(self, cmd, t_wall=None):
        if not self.ok:
            return
        ds = self.store.get_drone(); im = self.store.get_imu(); ac = self.store.get_actuators()
        _, seq = self.store.get_collision(); race = self.store.get_race()
        self.rows.append({
            "t_wall": (t_wall if t_wall is not None else time.time()) - self.t0,
            "t_us": int(ds.t_us) if ds else -1,
            "cmd": np.asarray(cmd, float).ravel(),
            "pos": np.asarray(ds.pos_ned, float) if ds else np.full(3, np.nan),
            "vel": np.asarray(ds.vel_ned, float) if ds else np.full(3, np.nan),
            "quat": np.asarray(ds.quat_wxyz, float) if ds else np.full(4, np.nan),
            "omega": np.asarray(ds.omega, float) if ds else np.full(3, np.nan),
            "acc": np.asarray(im[0], float) if im else np.full(3, np.nan),
            "gyro": np.asarray(im[1], float) if im else np.full(3, np.nan),
            "act": np.asarray(ac[0][:4], float) if ac else np.full(4, np.nan),
            "coll_seq": int(seq),
            "gate_idx": int(race["active_gate_index"]) if race and "active_gate_index" in race else -1,
            "live": 1 if (race and race.get("race_live")) else 0,
        })
        if len(self.rows) % self.flush_every == 0:
            self._dump()                       # periodic durability checkpoint

    def _dump(self):
        cols = {k: np.array([r[k] for r in self.rows]) for k in _COLS}
        np.savez_compressed(self.dir / "data.npz", **cols)

    def close(self):
        if not self.ok:
            return
        n = len(self.rows)
        if n:
            self._dump()
        dt = float(np.median(np.diff([r["t_wall"] for r in self.rows]))) if n > 1 else None
        meta = {"run_id": self.run_id, "path": str(self.dir), "script": self.script,
                "mode": self.mode, "n_samples": n, "t_start": self.stamp, "dt_median_s": dt,
                "notes": self.notes, "columns": _COLS, "quat_order": "wxyz", "frame": "NED",
                **self.extra_meta}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        with open(self.root / "index.jsonl", "a") as f:
            f.write(json.dumps({k: meta[k] for k in
                                ("run_id", "path", "script", "mode", "n_samples", "t_start", "notes")}) + "\n")
        print(f"[recorder] wrote {n} rows -> {self.dir}", flush=True)
        self.ok = False


def load_index(root=None):
    idx = (Path(root) if root else data_root()) / "index.jsonl"
    return [json.loads(x) for x in idx.read_text().splitlines() if x.strip()] if idx.exists() else []


def load_run(path):
    path = Path(path)
    d = {k: v for k, v in np.load(path / "data.npz", allow_pickle=False).items()}
    d["meta"] = json.loads((path / "meta.json").read_text())
    return d

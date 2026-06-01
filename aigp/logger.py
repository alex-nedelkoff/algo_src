"""Write captured frames + per-frame labels to disk."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from .labels import compute_label


class DataLogger:
    def __init__(self, out_dir, run_id: str, meta: dict):
        self.root = Path(out_dir) / run_id
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        self._labels = open(self.root / "labels.jsonl", "w")
        self._idx = 0

    def log(self, img, t_sim_ns: int, drone, gate) -> None:
        lab = compute_label(self._idx, t_sim_ns, drone, gate)
        cv2.imwrite(str(self.frames_dir / f"{self._idx:06d}.jpg"), img)
        self._labels.write(json.dumps(lab) + "\n")
        self._labels.flush()
        self._idx += 1

    def close(self) -> None:
        self._labels.close()

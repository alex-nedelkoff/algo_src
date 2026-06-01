import json
import numpy as np
from aigp.protocol import Gate
from aigp.state import DroneState
from aigp.logger import DataLogger

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def test_logger_writes_frame_and_label(tmp_path):
    log = DataLogger(out_dir=tmp_path, run_id="run0", meta={"k": "v"})
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    ds = DroneState(np.zeros(3), np.zeros(3), IDENT, np.zeros(3), 0)
    gate = Gate(id=0, pos_ned=np.array([10.0, 0.0, 0.0]),
                quat_ned_wxyz=IDENT, width=2.0, height=2.0)
    log.log(img, t_sim_ns=123, drone=ds, gate=gate)
    log.close()

    frames = list((tmp_path / "run0" / "frames").glob("*.jpg"))
    assert len(frames) == 1
    lines = (tmp_path / "run0" / "labels.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    lab = json.loads(lines[0])
    assert lab["frame"] == 0 and lab["gate_id"] == 0
    meta = json.loads((tmp_path / "run0" / "meta.json").read_text())
    assert meta["k"] == "v"


def test_logger_frame_index_increments(tmp_path):
    log = DataLogger(out_dir=tmp_path, run_id="r", meta={})
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    ds = DroneState(np.zeros(3), np.zeros(3), IDENT, np.zeros(3), 0)
    gate = Gate(id=0, pos_ned=np.array([10.0, 0.0, 0.0]),
                quat_ned_wxyz=IDENT, width=2.0, height=2.0)
    log.log(img, 1, ds, gate)
    log.log(img, 2, ds, gate)
    log.close()
    lines = (tmp_path / "r" / "labels.jsonl").read_text().strip().splitlines()
    assert [json.loads(l)["frame"] for l in lines] == [0, 1]

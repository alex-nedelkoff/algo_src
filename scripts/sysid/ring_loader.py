"""Mine vq_data recordings into a (cmd, omega, tilt) time series for rate-loop ID.
Pure numpy. The npz key mapping is passed explicitly (keymap) so the real recorder
schema is supplied, not guessed; DEFAULT_KEYMAP is the confirmed current schema."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# BEST-GUESS schema -- NOT yet confirmed against a real vq_data file (Step 0 found none). Pass an explicit keymap, or confirm these keys against a real recording before relying on the defaults.
DEFAULT_KEYMAP = {"t": "t", "cmd": "wcmd", "omega": "omega", "quat": "quat"}


@dataclass
class RunSeries:
    dt: float
    cmd: np.ndarray     # (T,3) omega_cmd rad/s
    omega: np.ndarray   # (T,3) measured rad/s
    tilt_deg: np.ndarray  # (T,)


def _tilt_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    # tilt = angle of body-z from world-up = acos(R[2,2]); R[2,2] = 1 - 2(x^2+y^2)
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(r22, -1.0, 1.0)))


def load_npz(path: str, keymap: dict = None) -> RunSeries:
    km = keymap or DEFAULT_KEYMAP
    d = np.load(path, allow_pickle=True)
    t = np.asarray(d[km["t"]], float).ravel()
    cmd = np.asarray(d[km["cmd"]], float).reshape(len(t), 3)
    omega = np.asarray(d[km["omega"]], float).reshape(len(t), 3)
    quat = np.asarray(d[km["quat"]], float).reshape(len(t), 4)
    dt = float(np.median(np.diff(t)))
    return RunSeries(dt=dt, cmd=cmd, omega=omega, tilt_deg=_tilt_deg(quat))


def bin_coverage(runs, axis, tilt_edges, amp_edges) -> np.ndarray:
    """Compute 2-D count matrix of samples binned by (tilt, |cmd|) for one axis.

    Args:
        runs: list of RunSeries
        axis: 0=roll, 1=pitch, 2=yaw
        tilt_edges: bin edges for tilt_deg (array-like)
        amp_edges: bin edges for |cmd[:, axis]| (array-like)

    Returns:
        cov: (n_tilt_bins, n_amp_bins) integer array of sample counts
    """
    tilt_edges = np.asarray(tilt_edges, float)
    amp_edges = np.asarray(amp_edges, float)
    cov = np.zeros((len(tilt_edges) - 1, len(amp_edges) - 1), dtype=int)
    for rs in runs:
        ti = np.digitize(rs.tilt_deg, tilt_edges) - 1
        ai = np.digitize(np.abs(rs.cmd[:, axis]), amp_edges) - 1
        ok = (ti >= 0) & (ti < cov.shape[0]) & (ai >= 0) & (ai < cov.shape[1])
        np.add.at(cov, (ti[ok], ai[ok]), 1)
    return cov

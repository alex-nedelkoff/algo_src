"""Acquire gate poses: trigger a reset, then wait for the track broadcast."""
from __future__ import annotations

import time


def acquire_gates(store, commander, timeout=15.0, gate_cli=None):
    """Returns a list[Gate]. Strategy:
    1) if gates already cached, return them;
    2) send sim_reset and wait up to `timeout` for the broadcast;
    3) if still none and a CLI fallback gate is given, synthesize a 1-gate list.
    """
    existing = store.get_gates()
    if existing:
        return existing

    commander.sim_reset()
    deadline = time.time() + timeout
    while time.time() < deadline:
        gates = store.get_gates()
        if gates:
            return gates
        time.sleep(0.1)

    if gate_cli is not None:
        import numpy as np
        from .protocol import Gate
        return [Gate(id=0, pos_ned=np.asarray(gate_cli, float),
                     quat_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                     width=2.0, height=2.0)]

    raise RuntimeError(
        "No track/gate broadcast received after reset. Restart the flight while "
        "connected, or pass --gate N,E,D."
    )

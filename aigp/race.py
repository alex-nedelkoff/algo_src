"""Race-start gating: never actuate until the sim race is live.

The AI-GP sim DQs and closes if any actuation (arm / attitude / velocity) is
sent during the ~3 s pre-race countdown. The race is live once
boot_ms >= start_ms (see protocol.is_race_live), surfaced by the IO layer as
Store.get_race_live().
"""
from __future__ import annotations

import time


def wait_for_race_live(store, timeout: float = 30.0, poll: float = 0.05) -> bool:
    """Block until the race is live (safe to actuate), or timeout. Returns success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if store.get_race_live():
            return True
        time.sleep(poll)
    return False


def wait_for_fresh_race_live(store, commander, timeout: float = 25.0, poll: float = 0.05) -> bool:
    """SIM_RESET, then wait for the FRESH countdown to complete before returning.

    Without the freshness check, the Store still holds race_live=True from the
    prior race right after reset, so a naive wait returns instantly and any
    subsequent actuation lands during the new countdown (ignored / held)."""
    prev = store.get_race()
    prev_boot = prev["boot_ms"] if prev else None
    commander.sim_reset()
    deadline = time.time() + timeout
    # phase 1: detect the reset — boot_ms drops (new race) or race goes not-live
    while time.time() < deadline:
        r = store.get_race()
        if r is not None and (
            (prev_boot is not None and r["boot_ms"] < prev_boot) or not r.get("race_live")
        ):
            break
        time.sleep(poll)
    # phase 2: wait for the fresh race to go live
    while time.time() < deadline:
        if store.get_race_live():
            return True
        time.sleep(poll)
    return False

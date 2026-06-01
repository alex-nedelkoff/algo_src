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

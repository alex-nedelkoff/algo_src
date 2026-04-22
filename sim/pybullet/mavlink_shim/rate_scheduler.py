"""Per-message rate scheduler for MAVLink outbound messages."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RateScheduler:
    """Returns which messages should fire on each tick.

    rates_hz: {message_name: target_rate_hz}.  Rate of 0 = never fires.
    tick_hz: the highest-frequency tick (sets the scheduling resolution).

    Each message has a "next_due_us" timestamp. On each tick, messages
    whose next_due_us is <= the current tick time fire and have their
    next_due_us advanced by their period. Drift-free.
    """
    rates_hz: dict[str, float]
    tick_hz: float
    _next_due_us: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        # Initialize all enabled messages to fire immediately at t=0.
        self._next_due_us = {
            name: 0 for name, hz in self.rates_hz.items() if hz > 0
        }

    def due(self, t_us: int) -> list[str]:
        """Return names of messages due to fire at or before t_us."""
        due_now = []
        for name, next_due in list(self._next_due_us.items()):
            if next_due <= t_us:
                due_now.append(name)
                period_us = int(1_000_000 / self.rates_hz[name])
                self._next_due_us[name] = next_due + period_us
        return due_now

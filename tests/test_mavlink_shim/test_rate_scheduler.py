"""Tests for RateScheduler — per-message tick due-time accumulator."""
import pytest

from sim.pybullet.mavlink_shim.rate_scheduler import RateScheduler


def test_messages_due_at_their_configured_period():
    sched = RateScheduler(rates_hz={"a": 200.0, "b": 100.0}, tick_hz=200.0)
    # Tick 0: both due at t=0 (initial state).
    assert set(sched.due(t_us=0)) == {"a", "b"}  # both due at t=0 (initial)
    # Next tick at 5 ms (one period of "a" at 200 Hz) — "a" due, "b" not due yet.
    assert set(sched.due(t_us=5_000)) == {"a"}
    # At 10 ms — "a" due (second period), "b" due (first period at 100 Hz).
    assert set(sched.due(t_us=10_000)) == {"a", "b"}


def test_message_count_matches_rate_over_n_seconds():
    sched = RateScheduler(rates_hz={"x": 100.0, "y": 1.0}, tick_hz=200.0)
    counts = {"x": 0, "y": 0}
    # Simulate 5 seconds at 200 Hz tick rate.
    n_ticks = 5 * 200
    for tick in range(n_ticks):
        t_us = int(tick * 1_000_000 / 200)
        for name in sched.due(t_us):
            counts[name] += 1
    # Expected: ~500 "x" (100 Hz × 5 s), ~5 "y" (1 Hz × 5 s).
    assert 495 <= counts["x"] <= 505
    assert 4 <= counts["y"] <= 6


def test_rate_zero_never_fires():
    sched = RateScheduler(rates_hz={"silent": 0.0, "loud": 1000.0}, tick_hz=1000.0)
    counts = {"silent": 0, "loud": 0}
    for tick in range(1000):
        t_us = int(tick * 1_000)
        for name in sched.due(t_us):
            counts[name] += 1
    assert counts["silent"] == 0
    assert counts["loud"] >= 990

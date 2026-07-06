import math
import os

import numpy as np
import pytest

from vq2.corpus import ImuSample
from vq2.fusion import AttitudeTracker, FusionConfig, run_fusion

ACC2 = os.path.expanduser(
    "~/Documents/drone-ai-grand-prix/vq2_data/vq2_accept2_wp"
)
G = 9.81


def _quiet_sample(roll, pitch, t_us):
    """IMU sample whose accel-implied attitude equals (roll, pitch), zero
    rates, |f| = g (quasi-static). Inverts estimators.accel_implied_attitude:
    roll = atan2(ay, -az), pitch = atan2(ax, sqrt(ay^2+az^2))."""
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    acc = (G * sp, G * cp * sr, -G * cp * cr)
    return ImuSample(t_us=t_us, acc=acc, gyr=(0.0, 0.0, 0.0), rx_wall=0.0)


def test_cf_pulls_attitude_error_back_during_quiet_flight():
    att = AttitudeTracker(roll=0.10, pitch=-0.08, cf_gain=0.05)
    for i in range(1000):  # ~7 s at 144 Hz, true attitude level
        att.update(_quiet_sample(0.0, 0.0, t_us=i * 6944))
    assert abs(att.roll) < math.radians(1.0)
    assert abs(att.pitch) < math.radians(1.0)


def test_cf_frozen_during_dynamic_accel():
    """|f| far from g (thrust transient) => accel is NOT gravity => no
    correction, and zero rates integrate nothing."""
    att = AttitudeTracker(roll=0.10, pitch=0.0, cf_gain=0.05)
    for i in range(50):
        att.update(ImuSample(t_us=i * 6944, acc=(3.0, 1.0, -14.0),
                             gyr=(0.0, 0.0, 0.0), rx_wall=0.0))
    assert abs(att.roll - 0.10) < 1e-9


def test_cf_frozen_during_fast_rotation():
    """High gyro => not quasi-static => no correction; rotation still
    integrates (that part is real motion)."""
    att = AttitudeTracker(roll=0.0, pitch=0.0, cf_gain=0.05)
    dt_us = 6944
    n = 20
    for i in range(n):
        att.update(ImuSample(t_us=i * dt_us, acc=(0.0, 0.0, -G),
                             gyr=(1.0, 0.0, 0.0), rx_wall=0.0))
    expected = 1.0 * dt_us / 1e6 * (n - 1)  # pure integration, no CF pull
    assert abs(att.roll - expected) < 1e-9


def test_cf_zero_gain_is_pure_gyro():
    att = AttitudeTracker(roll=0.10, pitch=-0.08, cf_gain=0.0)
    for i in range(500):
        att.update(_quiet_sample(0.0, 0.0, t_us=i * 6944))
    assert abs(att.roll - 0.10) < 1e-9
    assert abs(att.pitch + 0.08) < 1e-9


# NOTE: a fifth test (CF shrinks post-transient gravity leak on the attempt-2
# corpus) was written and REFUTED the CF hypothesis: the quiet-window a_w is
# dominated by REAL cruise dynamics (2.8 m/s^2), and under thrust |f|~g holds
# even in steady tilted cruise, so the quasi-static gate cannot distinguish
# gravity from the thrust axis -- the CF pulls real tilt toward zero. Accel
# tilt correction stays INVALID under thrust (estimators.py docstring rule).
# AttitudeTracker keeps cf_gain=0 in flight; the knob exists for bench use.

"""Unit tests for the VQ2 replay harness (synthetic data — no corpus needed)."""
import json
import math
import os

import pytest

from vq2 import corpus as corpus_mod
from vq2.estimators import BaselineImuEstimator, accel_implied_attitude, is_at_rest
from vq2.corpus import ImuSample
from vq2.replay import find_rest_windows, replay_corpus

G = 9.81


def imu(t_us, acc=(0.0, 0.0, -G), gyr=(0.0, 0.0, 0.0)):
    return ImuSample(t_us=t_us, acc=acc, gyr=gyr, rx_wall=t_us / 1e6)


def test_rest_detector():
    assert is_at_rest(imu(0))
    assert not is_at_rest(imu(0, gyr=(0.1, 0, 0)))
    assert not is_at_rest(imu(0, acc=(0, 0, -5.0)))


def test_accel_implied_attitude_matches_known_spawn_tilt():
    # measured VQ2 spawn: xacc -3.00, zacc -9.34 -> pitch ~ -17.8 deg
    r, p = accel_implied_attitude(imu(0, acc=(-3.0, 0.0, -9.34)))
    assert abs(math.degrees(p) - (-17.8)) < 0.2
    assert abs(math.degrees(r)) < 0.1


def test_wfix_pitch_integration():
    """Message pitch-rate is mirrored: constant +msg gyro_y must DECREASE
    the estimated (physical) pitch."""
    est = BaselineImuEstimator()
    est.seed_from_rest(imu(0))
    for i in range(1, 101):
        est.predict(imu(i * 10_000, gyr=(0.0, 0.5, 0.0)))
    assert est.state.pitch < -0.4  # ~ -0.5 rad after 1 s of +0.5 msg-rate


def test_roll_integration_not_mirrored():
    est = BaselineImuEstimator()
    est.seed_from_rest(imu(0))
    for i in range(1, 101):
        est.predict(imu(i * 10_000, gyr=(0.5, 0.0, 0.0)))
    assert est.state.roll > 0.4


def test_velocity_zero_at_level_rest():
    est = BaselineImuEstimator()
    est.seed_from_rest(imu(0))
    for i in range(1, 201):
        est.predict(imu(i * 10_000))
    assert abs(est.state.vx_b) < 1e-9
    assert abs(est.state.vz_up) < 1e-9


def test_velocity_integrates_thrust():
    """Level hover + 1 m/s^2 net upward specific force for 1 s -> vz_up ~ 1."""
    est = BaselineImuEstimator()
    est.seed_from_rest(imu(0))
    for i in range(1, 101):
        est.predict(imu(i * 10_000, acc=(0.0, 0.0, -(G + 1.0))))
    assert abs(est.state.vz_up - 1.0) < 0.02


def test_duplicate_stamps_ignored():
    est = BaselineImuEstimator()
    est.seed_from_rest(imu(0))
    est.predict(imu(10_000, gyr=(1.0, 0, 0)))
    r1 = est.state.roll
    est.predict(imu(10_000, gyr=(1.0, 0, 0)))  # same stamp: must be a no-op
    assert est.state.roll == r1


def test_find_rest_windows():
    rows = [imu(i * 10_000) for i in range(50)]
    rows += [imu((50 + i) * 10_000, gyr=(0.3, 0, 0)) for i in range(50)]
    rows += [imu((100 + i) * 10_000) for i in range(50)]
    w = find_rest_windows(rows, min_len=30)
    assert len(w) == 2
    assert w[0][0] == 0 and w[1][1] == 149


@pytest.fixture
def tiny_corpus(tmp_path):
    """Two stamp epochs (reset in between); second epoch = rest, motion, rest."""
    rows = []
    # epoch 1 (pre-reset junk)
    for i in range(60):
        rows.append({"mavpackettype": "HIGHRES_IMU", "time_usec": 5_000_000 + i * 7000,
                     "xacc": -3.0, "yacc": 0.0, "zacc": -9.34,
                     "xgyro": 0.0, "ygyro": 0.0, "zgyro": 0.0, "_rx_wall": 100.0 + i * 0.007})
    # epoch 2: rest(60) -> pitch motion(60, msg gyro_y +0.5) -> rest(60)
    t0 = 1_000_000
    w = 200.0
    for i in range(60):
        rows.append({"mavpackettype": "HIGHRES_IMU", "time_usec": t0 + i * 7000,
                     "xacc": 0.0, "yacc": 0.0, "zacc": -G,
                     "xgyro": 0.0, "ygyro": 0.0, "zgyro": 0.0, "_rx_wall": w + i * 0.007})
    # motion: msg gyro_y +0.5 (physical pitch rate -0.5 via wfix); accel follows
    # the static-gravity split for the CURRENT physical pitch so the synthetic
    # data is self-consistent (ax = G*sin(p), az = -G*cos(p))
    p = 0.0
    for i in range(60):
        p += -0.5 * 0.007
        rows.append({"mavpackettype": "HIGHRES_IMU", "time_usec": t0 + (60 + i) * 7000,
                     "xacc": G * math.sin(p), "yacc": 0.0, "zacc": -G * math.cos(p),
                     "xgyro": 0.0, "ygyro": 0.5, "zgyro": 0.0, "_rx_wall": w + (60 + i) * 0.007})
    # mirror back to level
    for i in range(60):
        p += 0.5 * 0.007
        rows.append({"mavpackettype": "HIGHRES_IMU", "time_usec": t0 + (120 + i) * 7000,
                     "xacc": G * math.sin(p), "yacc": 0.0, "zacc": -G * math.cos(p),
                     "xgyro": 0.0, "ygyro": -0.5, "zgyro": 0.0, "_rx_wall": w + (120 + i) * 0.007})
    for i in range(60):
        rows.append({"mavpackettype": "HIGHRES_IMU", "time_usec": t0 + (180 + i) * 7000,
                     "xacc": 0.0, "yacc": 0.0, "zacc": -G,
                     "xgyro": 0.0, "ygyro": 0.0, "zgyro": 0.0, "_rx_wall": w + (180 + i) * 0.007})
    with open(tmp_path / "mavlink.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(tmp_path)


def test_segmentation_on_stamp_reset(tiny_corpus):
    c = corpus_mod.load(tiny_corpus)
    assert len(c.segments) == 2
    assert len(c.flight_segment.imu) == 240


def test_replay_end_to_end(tiny_corpus):
    report = replay_corpus(tiny_corpus)
    assert report["segments"] == 2
    assert len(report["rest_windows"]) >= 2
    err = report["final_rest"]["att_err_deg"]
    assert abs(err["roll"]) < 0.01
    assert abs(err["pitch"]) < 0.01
    assert abs(report["final_rest"]["vel_drift"]["vz_up"]) < 0.01

import cv2
import numpy as np

from vq2.safe_gate_servo import GateObservation, SafeGateServo, detect_aperture_gate


def obs(u=320, w=60, v=180):
    return GateObservation(float(u), float(v), float(w), float(w))


def test_requires_stable_centered_track_before_forward_motion():
    servo = SafeGateServo(stable_required=3)
    assert servo.step(obs()).forward_m_s == 0.0
    assert servo.step(obs()).forward_m_s == 0.0
    assert servo.step(obs()).forward_m_s == 0.08


def test_lost_or_clipped_gate_fails_closed_to_hold():
    servo = SafeGateServo(stable_required=1)
    assert servo.step(obs()).forward_m_s == 0.08
    assert servo.step(None).forward_m_s == 0.0
    assert not servo.step(obs(v=260)).accepted


def test_rejects_track_hop_and_never_forwards_on_it():
    servo = SafeGateServo(stable_required=1)
    servo.step(obs(u=300, w=60))
    cmd = servo.step(obs(u=390, w=60))
    assert not cmd.accepted
    assert cmd.forward_m_s == 0.0
    assert servo.step(obs(u=320, w=60)).reason == "track was lost; reset required"


def test_ignores_small_red_component_before_initial_acquisition():
    servo = SafeGateServo(stable_required=1)
    assert not servo.step(obs(u=420, w=32)).accepted
    assert servo.step(obs(u=320, w=82)).forward_m_s == 0.08


def test_yaw_direction_matches_empirical_legacy_sign():
    cmd = SafeGateServo(stable_required=99).step(obs(u=352))
    assert cmd.yaw_rate_rad_s < 0.0


def test_aperture_detector_rejects_solid_red_sign_and_returns_hole_center():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (110, 110), (0, 0, 255), -1)  # solid sign
    cv2.rectangle(image, (250, 90), (370, 210), (0, 0, 255), -1)
    cv2.rectangle(image, (275, 115), (345, 185), (0, 0, 0), -1)
    got = detect_aperture_gate(image)
    assert got is not None
    assert (got.u, got.v) == (310.5, 150.5)

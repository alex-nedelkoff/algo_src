from __future__ import annotations

import pytest

from vq2.live.dpvo_odom_bridge import ensure_control_approved


def test_unapproved_calibration_is_allowed_only_for_observation():
    calibration = {"control_approved": False}
    ensure_control_approved(calibration, observe_only=True)
    with pytest.raises(ValueError, match="observe-only"):
        ensure_control_approved(calibration, observe_only=False)


def test_approved_calibration_can_enable_control():
    ensure_control_approved({"control_approved": True}, observe_only=False)

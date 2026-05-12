"""SET_POSITION_TARGET_LOCAL_NED parsing — type_mask honouring + NED→ENU."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sim.pybullet.mavlink_shim.position_target import (
    PositionTarget, parse_set_position_target_local_ned,
)


def _msg(**kw) -> SimpleNamespace:
    """Build a minimal pymavlink-like message."""
    base = dict(
        x=0.0, y=0.0, z=0.0,
        vx=0.0, vy=0.0, vz=0.0,
        afx=0.0, afy=0.0, afz=0.0,
        yaw=0.0, yaw_rate=0.0,
        type_mask=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_full_target_parses_with_ned_to_enu_conversion():
    # NED input: x=5 (north), y=2 (east), z=-3 (down → 3m above ground).
    # ENU expected: x=east=2, y=north=5, z=up=3.
    tgt = parse_set_position_target_local_ned(_msg(x=5.0, y=2.0, z=-3.0, yaw=0.5))
    assert isinstance(tgt, PositionTarget)
    np.testing.assert_allclose(tgt.pos_enu, [2.0, 5.0, 3.0])
    assert tgt.use_position is True
    assert tgt.yaw_enu == -0.5  # yaw flips sign across NED↔ENU


def test_velocity_only_target():
    # Bits set: pos x/y/z (0x07), accel x/y/z (0x1C0), yaw (0x400) = 0x5C7
    tgt = parse_set_position_target_local_ned(
        _msg(vx=1.0, vy=2.0, vz=-0.5, type_mask=0x5C7)
    )
    assert tgt.use_position is False
    assert tgt.use_velocity is True
    np.testing.assert_allclose(tgt.vel_enu, [2.0, 1.0, 0.5])


def test_yaw_ignore_bit_blanks_yaw():
    tgt = parse_set_position_target_local_ned(_msg(yaw=1.5, type_mask=0x400))
    assert tgt.use_yaw is False


def test_accel_fields_pass_through():
    tgt = parse_set_position_target_local_ned(_msg(afx=0.5, afy=0.0, afz=-0.2))
    np.testing.assert_allclose(tgt.accel_enu, [0.0, 0.5, 0.2])
    assert tgt.use_accel is True

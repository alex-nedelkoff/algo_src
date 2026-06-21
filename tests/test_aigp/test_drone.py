import numpy as np
from aigp.drone import Result, Status, FlightConfig
from aigp.navigator import NavGains


def test_flightconfig_safe_defaults_within_envelope():
    c = FlightConfig()
    assert c.amax <= 4.5 and c.tilt_deg <= 20 and c.default_speed <= 6


def test_flightconfig_to_navgains_maps_fields():
    c = FlightConfig(vmax=7.0, amax=4.0, tilt_deg=20.0, zff=-2.4, capture=2.5, kiz=0.8, c_max=18.0)
    g = c.to_navgains()
    assert isinstance(g, NavGains)
    assert g.MAX_SPEED == 7.0 and g.VLAT_MAX == 7.0 and g.FWD_AMAX == 4.0
    assert g.DECEL_MAX >= 4.0 and g.TILT_MAX_DEG == 20.0 and g.Z_FF == -2.4
    assert g.CAPTURE == 2.5 and g.KI_Z == 0.8 and g.C_MAX == 18.0


def test_status_defaults_running():
    s = Status(result=Result.RUNNING, phase="x", pos_ned=np.zeros(3), vel=0.0,
               tilt_deg=0.0, mode="hold", target=None, progress=0.0, error=None)
    assert s.result is Result.RUNNING and s.error is None

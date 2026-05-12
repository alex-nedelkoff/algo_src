"""Default rates_hz follow VADR-TS-002 §4.4 (heartbeat ≥2 Hz, no ODOMETRY)."""
from sim.pybullet.mavlink_shim.shim import MavlinkShim


def test_default_rates_match_spec():
    # Read the default factory directly — we don't construct a shim here,
    # so no backend is needed.
    fields = {f.name: f for f in MavlinkShim.__dataclass_fields__.values()}
    defaults = fields["rates_hz"].default_factory()

    assert defaults.get("heartbeat") == 2.0, "heartbeat must be ≥2 Hz per spec §4.4"
    assert "timesync" in defaults and defaults["timesync"] == 10.0
    assert "odometry" not in defaults, "ODOMETRY is not in the spec; drop from defaults"
    # Existing rates retained.
    assert defaults["attitude"] == 100.0
    assert defaults["highres_imu"] == 200.0

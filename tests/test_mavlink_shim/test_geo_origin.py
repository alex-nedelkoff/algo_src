"""Flat-earth ENU->lat/lon offset helper."""
from __future__ import annotations

import math

from sim.pybullet.mavlink_shim.geo_origin import (
    ANDURIL_HQ_LAT_LON, enu_to_lat_lon,
)


def test_default_origin_is_anduril_hq():
    lat, lon = ANDURIL_HQ_LAT_LON
    assert abs(lat - 33.6595) < 1e-6
    assert abs(lon - (-117.9988)) < 1e-6


def test_origin_offset_is_origin():
    lat, lon = enu_to_lat_lon(0.0, 0.0, origin=ANDURIL_HQ_LAT_LON)
    assert lat == ANDURIL_HQ_LAT_LON[0]
    assert lon == ANDURIL_HQ_LAT_LON[1]


def test_north_offset_increases_lat():
    """100 m north -> lat increases by ~100/111320 deg."""
    lat0, lon0 = ANDURIL_HQ_LAT_LON
    lat, lon = enu_to_lat_lon(east=0.0, north=100.0, origin=ANDURIL_HQ_LAT_LON)
    expected_dlat = 100.0 / 111_320.0
    assert abs((lat - lat0) - expected_dlat) < 1e-6
    assert abs(lon - lon0) < 1e-9


def test_east_offset_scales_with_cos_lat():
    """100 m east at lat=33.66 -> dlon = 100/(111320*cos(33.66 deg))."""
    lat0, lon0 = ANDURIL_HQ_LAT_LON
    lat, lon = enu_to_lat_lon(east=100.0, north=0.0, origin=ANDURIL_HQ_LAT_LON)
    expected_dlon = 100.0 / (111_320.0 * math.cos(math.radians(lat0)))
    assert abs((lon - lon0) - expected_dlon) < 1e-6
    assert abs(lat - lat0) < 1e-9

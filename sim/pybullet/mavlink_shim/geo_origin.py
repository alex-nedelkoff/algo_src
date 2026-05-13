"""Flat-earth ENU->lat/lon helper for synthetic GPS in the QGC shim.

Used only to give QGroundControl a map location to render. The simulator
has no real geographic position (VADR-TS-002 says 'GPS simulation is not
available'); this is a presentation-layer fiction.

Approximation: 1 deg lat ~= 111_320 m everywhere. 1 deg lon scales by
cos(lat). Good to < 1 m within a 1 km radius — far beyond the warehouse
operating volume.
"""
from __future__ import annotations

import math


# Anduril HQ, Costa Mesa, CA. Chosen because (a) recognisable, (b) clearly
# fictional for sim purposes, (c) makes the QGC map look populated.
ANDURIL_HQ_LAT_LON: tuple[float, float] = (33.6595, -117.9988)

_M_PER_DEG_LAT = 111_320.0


def enu_to_lat_lon(
    east: float, north: float, origin: tuple[float, float],
) -> tuple[float, float]:
    """Project an ENU (east, north) offset onto lat/lon, flat-earth."""
    origin_lat, origin_lon = origin
    dlat = north / _M_PER_DEG_LAT
    dlon = east / (_M_PER_DEG_LAT * math.cos(math.radians(origin_lat)))
    return origin_lat + dlat, origin_lon + dlon

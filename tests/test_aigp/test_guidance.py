import numpy as np
from aigp.guidance import Setpoint, yaw_to_target, OrbitPattern


def test_yaw_to_target_north():
    # gate north of drone -> yaw 0
    assert np.isclose(yaw_to_target(np.zeros(3), np.array([5.0, 0.0, 0.0])), 0.0)


def test_yaw_to_target_east():
    # gate east of drone -> yaw +pi/2
    assert np.isclose(yaw_to_target(np.zeros(3), np.array([0.0, 5.0, 0.0])), np.pi / 2)


def test_orbit_on_radius_is_tangential():
    gate = np.array([0.0, 0.0, -2.0])
    radius, speed = 4.0, 2.0
    # drone 4 m north of gate, at gate height
    drone = np.array([4.0, 0.0, -2.0])
    sp = OrbitPattern(radius=radius, speed=speed, target_z=-2.0).update(
        drone, np.zeros(3), gate
    )
    # tangential (CCW) at north point is +east; radius error 0 -> no radial component
    assert np.isclose(sp.vx, 0.0, atol=1e-6)
    assert np.isclose(sp.vy, speed, atol=1e-6)
    assert np.isclose(sp.vz, 0.0, atol=1e-6)
    # yaw points back at gate (south) -> pi
    assert np.isclose(abs(sp.yaw), np.pi, atol=1e-6)


def test_orbit_too_close_pushes_outward():
    gate = np.array([0.0, 0.0, 0.0])
    drone = np.array([2.0, 0.0, 0.0])  # 2 m from gate, radius 4 -> too close
    sp = OrbitPattern(radius=4.0, speed=2.0, target_z=0.0, kp_radius=1.0).update(
        drone, np.zeros(3), gate
    )
    # radial direction is +north; being too close -> velocity has +north component
    assert sp.vx > 0.0


def test_orbit_holds_altitude():
    gate = np.array([0.0, 0.0, -2.0])
    # drone at z=0 is BELOW target altitude z=-2 (NED: more-negative z = higher)
    drone = np.array([4.0, 0.0, 0.0])
    sp = OrbitPattern(radius=4.0, speed=2.0, target_z=-2.0, kp_z=1.0).update(
        drone, np.zeros(3), gate
    )
    # must climb to reach target_z=-2 -> decrease z -> vz < 0
    assert sp.vz < 0.0


from aigp.guidance import ApproachPattern


def test_approach_moves_toward_first_waypoint():
    gate = np.array([0.0, 0.0, -2.0])
    # one waypoint: 6 m north of gate
    pat = ApproachPattern(offsets=[(6.0, 0.0, 0.0)], speed=2.0, switch_dist=1.0)
    drone = np.array([0.0, 0.0, -2.0])  # 6 m south of the waypoint
    sp = pat.update(drone, np.zeros(3), gate)
    # waypoint is north -> velocity points north at ~speed
    assert sp.vx > 0.0
    assert np.isclose(np.linalg.norm([sp.vx, sp.vy, sp.vz]), 2.0, atol=1e-6)


def test_approach_advances_waypoint_when_close():
    gate = np.array([0.0, 0.0, 0.0])
    pat = ApproachPattern(
        offsets=[(1.0, 0.0, 0.0), (0.0, 5.0, 0.0)], speed=2.0, switch_dist=1.5
    )
    drone = np.array([1.0, 0.0, 0.0])  # already at waypoint 0 (within switch_dist)
    sp = pat.update(drone, np.zeros(3), gate)
    # should have advanced to waypoint 1 (east) -> velocity points east
    assert sp.vy > 0.0
    assert pat.idx == 1

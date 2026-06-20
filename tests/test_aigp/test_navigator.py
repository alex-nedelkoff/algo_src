import json
import numpy as np
from aigp.navigator import (
    NavGains, load_plant, attitude_command,
    cruise_accel, settle_accel, WaypointNavigator,
)
from aigp.state import DroneState

G = 9.81


def test_navgains_defaults_match_goto():
    g = NavGains()
    assert g.MAX_SPEED == 1.2
    assert g.ARRIVE == 1.5
    assert g.TILT_MAX_DEG == 15.0
    assert g.WMAX == 4.0
    np.testing.assert_allclose(g.KP_ATT, [0.5, 1.6, 1.0])


def test_load_plant_reads_sim_response_schema(tmp_path):
    p = tmp_path / "sim_response.json"
    p.write_text(json.dumps({
        "hover_thrust": 0.42, "k_a": 13.0,
        "rate_gain_axes": {"roll": 9.0, "pitch": 9.5, "yaw": 4.0},
    }))
    hover, k_a, rg = load_plant(str(p))
    assert hover == 0.42 and k_a == 13.0
    np.testing.assert_allclose(rg, [9.0, 9.5, 4.0])


def _mkstate(pos, vel, quat=(1, 0, 0, 0), omega=(0, 0, 0)):
    return DroneState(np.array(pos, float), np.array(vel, float),
                      np.array(quat, float), np.array(omega, float), 0)


def test_cruise_accel_drives_along_leg():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0])
    a2, tv, spd = cruise_accel(st, leg_start=np.array([0, 0, -2.0]),
                               target=np.array([10, 0, -2.0]), gains=g)
    np.testing.assert_allclose(tv, [1.0, 0.0], atol=1e-9)
    assert spd == g.MAX_SPEED          # 0.6 * 10 clipped to 1.2
    assert a2[0] > 0 and abs(a2[1]) < 1e-9   # accel toward target, no cross term on the line


def test_cruise_accel_corrects_cross_track():
    g = NavGains()
    st = _mkstate([0, 1, -2], [0, 0, 0])   # 1 m to +E of the N-running line
    a2, tv, spd = cruise_accel(st, leg_start=np.array([0, 0, -2.0]),
                               target=np.array([10, 0, -2.0]), gains=g)
    assert a2[1] < 0                    # accel pushes back toward the line (-E)


def test_settle_accel_damps_and_pulls():
    g = NavGains()
    st = _mkstate([1, 0, -2], [0.5, 0, 0])  # 1 m past target in +N, moving +N
    a2 = settle_accel(st, target=np.array([0, 0, -2.0]), gains=g)
    assert a2[0] < 0                    # pulls back to target AND damps +N velocity


_PLANT = (0.5, 13.0, np.array([1.0, 1.0, 1.0]))   # hover, k_a, rg (rg=1 -> rate_cmd == w_des clipped)


def test_attitude_command_level_hover():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0])           # level, at rest, on altitude
    rate, thr, tilt, dbg = attitude_command(st, a2=np.zeros(2), z_sp=-2.0,
                                             yaw_ref=0.0, plant=_PLANT, gains=g)
    assert abs(thr - 0.5) < 1e-6                    # collective == hover when level + no accel
    np.testing.assert_allclose(rate, [0, 0, 0], atol=1e-9)
    assert abs(tilt) < 1e-6


def test_attitude_command_clamps_tilt():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0])
    _, _, _, dbg = attitude_command(st, a2=np.array([100.0, 0.0]), z_sp=-2.0,
                                    yaw_ref=0.0, plant=_PLANT, gains=g)
    tilt_max_acc = np.tan(np.radians(g.TILT_MAX_DEG)) * G
    assert abs(np.linalg.norm(dbg["a"][:2]) - tilt_max_acc) < 1e-6


def test_attitude_command_clips_rate_to_wmax():
    g = NavGains(WMAX=0.1)
    st = _mkstate([0, 0, -2], [0, 0, 0])
    rate, _, _, _ = attitude_command(st, a2=np.array([5.0, 0.0]), z_sp=-2.0,
                                     yaw_ref=0.0, plant=_PLANT, gains=g)
    assert np.all(np.abs(rate) <= 0.1 + 1e-9)


def test_attitude_command_yaw_sign():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0])            # yaw_cur = 0
    rate_pos, *_ = attitude_command(st, np.zeros(2), -2.0, yaw_ref=+0.3, plant=_PLANT, gains=g)
    rate_neg, *_ = attitude_command(st, np.zeros(2), -2.0, yaw_ref=-0.3, plant=_PLANT, gains=g)
    assert rate_pos[2] > 0 and rate_neg[2] < 0


class _FakeStore:
    def __init__(self, ds):
        self._ds = ds
    def get_drone(self):
        return self._ds
    def get_race_live(self):
        return True


def _nav(ds):
    return WaypointNavigator(_FakeStore(ds), commander=None, plant=_PLANT)


def test_set_origin_captures_current_state():
    nav = _nav(_mkstate([1, 2, -3], [0, 0, 0]))   # level quat -> yaw 0
    nav.set_origin()
    np.testing.assert_allclose(nav._origin_pos, [1, 2, -3])
    assert abs(nav._origin_yaw) < 1e-9


def test_resolve_world_is_absolute():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0]))
    nav.set_origin(pos_ned=np.array([0, 0, -2.0]), yaw=0.0)
    np.testing.assert_allclose(nav._resolve(np.array([3, 4, -2.0]), "world"), [3, 4, -2])


def test_resolve_body_offsets_from_origin():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0]))
    nav.set_origin(pos_ned=np.array([0, 0, -2.0]), yaw=0.0)   # heading 0 -> fwd=-N, right=+E
    np.testing.assert_allclose(nav._resolve(np.array([6, 0, 0.0]), "body"), [-6, 0, -2])
    np.testing.assert_allclose(nav._resolve(np.array([0, 4, 0.0]), "body"), [0, 4, -2])


def test_resolve_rejects_bad_frame():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0]))
    nav.set_origin(pos_ned=np.array([0, 0, -2.0]), yaw=0.0)
    try:
        nav._resolve(np.array([1, 0, 0.0]), "polar")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_yaw_ref_face_points_at_target():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0]))
    nav.set_origin(pos_ned=np.array([0, 0, -2.0]), yaw=0.0)
    yr = nav._yaw_ref("face", np.array([0, 5, -2.0]))   # target due +E
    assert abs(yr - np.pi / 2) < 1e-9
    assert nav._yaw_ref("hold", np.array([0, 5, -2.0])) == 0.0


def test_cruise_accel_zero_length_segment_steers_by_rel():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0])
    # leg_start == target (degenerate); rel points to the actual target 5 m north
    a2, tv, spd = cruise_accel(st, leg_start=np.array([5, 0, -2.0]),
                               target=np.array([5, 0, -2.0]), gains=g)
    np.testing.assert_allclose(tv, [1.0, 0.0], atol=1e-6)   # tangent falls back to rel direction (+N)
    assert spd >= 0


from aigp.navigator import _qfix, WFIX


def test_qfix_shuffles_wxyz_to_xyzw():
    np.testing.assert_allclose(_qfix([0.7, 0.1, 0.2, 0.3]), [0.1, 0.2, 0.3, 0.7])


def test_wfix_is_pitch_mirror():
    np.testing.assert_allclose(WFIX, [1.0, -1.0, 1.0])


def test_set_origin_computes_scam_and_yaw0t():
    # spawn quat in sim-wire convention: identity true attitude is wire [0,0,0,1]
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0], quat=(0, 0, 0, 1)))
    nav.set_origin()
    assert nav._s_cam in (1.0, -1.0)
    assert np.isfinite(nav._yaw0_t)


from aigp.navigator import ZVDShaper, attitude_command_tf, _spline_accel

# sim-wire quat whose TRUE attitude (qfix) is level + nose along +N: true wxyz [1,0,0,0] -> wire [0,1,0,0]
_WIRE_LEVEL = (0.0, 1.0, 0.0, 0.0)


def test_tf_level_hover_zero_rate():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)
    rate, thr, tilt, dbg = attitude_command_tf(st, np.zeros(2), -2.0, yaw_ref=0.0,
                                               plant=_PLANT, gains=g, s_cam=1.0)
    assert abs(thr - 0.5) < 1e-6 and abs(tilt) < 1e-6
    np.testing.assert_allclose(rate, [0, 0, 0], atol=1e-6)


def test_tf_z_ff_adds_lift():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)   # ze = 0 (on altitude)
    _, _, _, d0 = attitude_command_tf(st, np.zeros(2), -2.0, 0.0, _PLANT, g, 1.0, z_ff=0.0)
    _, _, _, d1 = attitude_command_tf(st, np.zeros(2), -2.0, 0.0, _PLANT, g, 1.0, z_ff=-2.0)
    assert d1["a"][2] < d0["a"][2]              # negative z_ff -> more upward accel command
    np.testing.assert_allclose(d1["a"][2] - d0["a"][2], -2.0, atol=1e-9)


def test_tf_collective_anticipates_commanded_tilt():
    g = NavGains(); g.Z_FF = 0.0
    st = _mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)   # ACTUAL attitude is level
    _, thr0, _, _ = attitude_command_tf(st, np.zeros(2), -2.0, 0.0, _PLANT, g, 1.0)
    _, thr1, _, _ = attitude_command_tf(st, np.array([5.0, 0.0]), -2.0, 0.0, _PLANT, g, 1.0)
    # anticipatory: a horizontal (tilting) accel command raises the collective even though the
    # actual tilt is still level -> no altitude sag while leaning. (actual-tilt comp would not.)
    assert thr1 > thr0


def test_tf_clamps_tilt():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)
    _, _, _, dbg = attitude_command_tf(st, np.array([100.0, 0.0]), -2.0, 0.0, _PLANT, g, 1.0)
    assert abs(np.linalg.norm(dbg["a"][:2]) - np.tan(np.radians(g.TILT_MAX_DEG)) * 9.81) < 1e-6


def test_tf_yaw_sign_uses_truecam():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)   # true-cam yaw_cur = 0
    rp, *_ = attitude_command_tf(st, np.zeros(2), -2.0, yaw_ref=+0.3, plant=_PLANT, gains=g, s_cam=1.0)
    rn, *_ = attitude_command_tf(st, np.zeros(2), -2.0, yaw_ref=-0.3, plant=_PLANT, gains=g, s_cam=1.0)
    # WFIX mirrors the yaw axis sign in the wire frame; assert the two are opposite + nonzero
    assert rp[2] * rn[2] < 0 and abs(rp[2]) > 1e-6


def test_zvd_impulse_response_three_taps():
    z = ZVDShaper(delay=(3, 3, 3))
    out0 = z.shape([1.0, 0.0, 0.0])          # impulse on roll
    assert abs(out0[0] - 0.371) < 1e-9        # tap A0 now
    for _ in range(3):
        out = z.shape([0.0, 0.0, 0.0])        # advance to frame 3 (buf[3] holds impulse)
    assert abs(out[0] - 0.476) < 1e-9         # tap A1 at delay 3
    for _ in range(3):
        out = z.shape([0.0, 0.0, 0.0])        # advance to frame 6 (buf[6] holds impulse)
    assert abs(out[0] - 0.153) < 1e-9         # tap A2 at 2*delay


def test_zvd_amplitudes_sum_to_one():
    np.testing.assert_allclose(ZVDShaper.AMP.sum(), 1.0, atol=1e-9)


def test_spline_accel_corrects_cross_track_and_holds_speed():
    g = NavGains()
    # reference: on the +N line at the origin, tangent +N, scheduled speed 2.0
    ref = {"pos": np.array([0.0, 0.0, -2.0]), "tang": np.array([1.0, 0.0, 0.0]), "v": 2.0}
    st = _mkstate([0.0, 1.0, -2.0], [0.0, 0.0, 0.0])      # 1 m to +E of the line, at rest
    a2, travel = _spline_accel(st, ref, g)
    np.testing.assert_allclose(travel, [1.0, 0.0], atol=1e-9)
    assert a2[1] < 0      # cross-track accel pushes back toward the line (-E)
    assert a2[0] > 0      # along-track accel builds toward scheduled speed


# --- true-frame strafe (camera-decoupled legs) ---
from aigp.navigator import _strafe_recompose, _course_guidance


def test_strafe_recompose_emits_scalars_along_heading():
    # a_al along the nose, a_lat along the nose-perp; s_lat=+1
    fwd = np.array([1.0, 0.0])
    a_h = _strafe_recompose(a_al=0.3, a_lat=0.5, fwd=fwd, s_lat=1.0)
    np.testing.assert_allclose(a_h, [0.3, 0.5], atol=1e-9)   # lat = [-fwd_y, fwd_x] = [0,1]


def test_strafe_recompose_s_lat_reflects_lateral():
    fwd = np.array([1.0, 0.0])
    a_h = _strafe_recompose(a_al=0.3, a_lat=0.5, fwd=fwd, s_lat=-1.0)
    np.testing.assert_allclose(a_h, [0.3, -0.5], atol=1e-9)   # only the lateral sign flips


def test_strafe_recompose_rotates_into_heading():
    # nose rotated +90deg: forward intent emits along the new nose (+E)
    fwd = np.array([0.0, 1.0])
    a_h = _strafe_recompose(a_al=1.0, a_lat=0.0, fwd=fwd, s_lat=1.0)
    np.testing.assert_allclose(a_h, [0.0, 1.0], atol=1e-9)


def test_course_guidance_forward_target_is_pure_along():
    # target straight ahead along +cam_live: drive forward, no lateral (FIXED course frame)
    g = NavGains()
    cam_live = np.array([1.0, 0.0]); lat_course = np.array([0.0, 1.0])
    a_al, a_lat = _course_guidance(pos=np.zeros(3), vel=np.zeros(3),
                                   target=np.array([12.0, 0.0, -2.0]),
                                   cam_live=cam_live, lat_course=lat_course, gains=g)
    assert a_al > 0 and abs(a_lat) < 1e-9


def test_course_guidance_lateral_target_is_pure_lateral():
    # target purely to +lat_course: drive lateral, ~no forward -- SAME frame as the forward case
    g = NavGains()
    cam_live = np.array([1.0, 0.0]); lat_course = np.array([0.0, 1.0])
    a_al, a_lat = _course_guidance(pos=np.zeros(3), vel=np.zeros(3),
                                   target=np.array([0.0, 12.0, -2.0]),
                                   cam_live=cam_live, lat_course=lat_course, gains=g)
    assert a_lat > 0 and abs(a_al) < 1e-9


def test_advance_wp_capture_midcourse_arrive_at_last():
    g = NavGains()
    # mid-course waypoints advance at the loose CAPTURE radius (flow through corners, no stop)
    assert WaypointNavigator._advance_wp(g.CAPTURE - 0.1, is_last=False, gains=g) is True
    assert WaypointNavigator._advance_wp(g.CAPTURE + 0.1, is_last=False, gains=g) is False
    # the FINAL waypoint only completes at the tight ARRIVE radius
    assert WaypointNavigator._advance_wp(g.ARRIVE - 0.1, is_last=True, gains=g) is True
    assert WaypointNavigator._advance_wp(g.CAPTURE, is_last=True, gains=g) is False
    assert g.CAPTURE > g.ARRIVE   # flow radius must exceed the arrival radius


def test_dr_vel_filters_position_derivative():
    from aigp.navigator import _dr_vel
    # raw lateral velocity = (1-0)/0.1 = 10; one EMA step at alpha 0.85 -> 0.15*10 = 1.5
    vw = _dr_vel(np.zeros(2), np.array([0.0, 1.0]), np.array([0.0, 0.0]), dt=0.1, alpha=0.85)
    np.testing.assert_allclose(vw, [0.0, 1.5], atol=1e-9)


from aigp.navigator import _line_guidance, _z_int_step


def test_z_int_accumulates_within_gate():
    # |ze| < gate -> integrate KI*ze*dt
    zi = _z_int_step(0.0, ze=1.0, dt=0.5, ki=0.8, gate=1.5, clip=3.0)
    np.testing.assert_allclose(zi, 0.4, atol=1e-9)


def test_z_int_no_windup_outside_gate():
    # big altitude error -> hold (no windup during the initial climb transient)
    assert _z_int_step(0.5, ze=2.0, dt=0.5, ki=0.8, gate=1.5, clip=3.0) == 0.5


def test_z_int_clamps():
    assert _z_int_step(2.9, ze=1.0, dt=1.0, ki=0.8, gate=1.5, clip=3.0) == 3.0
    assert _z_int_step(-2.9, ze=-1.0, dt=1.0, ki=0.8, gate=1.5, clip=3.0) == -3.0


def test_z_int_unwinds_when_far_and_opposing():
    # charged negative on a climb, now a big positive error (descent) -> must bleed toward 0 despite
    # being outside the gate (else the stale bias fights the descent -> stuck off-altitude)
    zi = _z_int_step(-2.0, ze=3.0, dt=0.5, ki=0.8, gate=1.5, clip=3.0)
    assert -2.0 < zi <= 0.0   # moved toward zero, not frozen


def test_line_guidance_on_line_drives_along_no_cross():
    g = NavGains()
    cam_live = np.array([1.0, 0.0]); lat_course = np.array([0.0, 1.0])
    # leg along +cam_live; drone ON the line at 3 m, moving along it -> pure along, zero cross
    a_al, a_lat = _line_guidance(pos=np.array([3.0, 0.0, -2.0]), vw=np.array([1.0, 0.0]),
                                 leg_start=np.array([0.0, 0.0, -2.0]), target=np.array([10.0, 0.0, -2.0]),
                                 cam_live=cam_live, lat_course=lat_course, gains=g)
    assert a_al > 0 and abs(a_lat) < 1e-6


def test_brake_to_stop_carries_speed_where_linear_brakes_early():
    g = NavGains(); g.MAX_SPEED = 5.0; g.DECEL_MAX = 2.0
    cam = np.array([1.0, 0.0]); lat = np.array([0.0, 1.0])
    # 4 m out at 4 m/s: on the constant-decel stop curve sqrt(2*2*4)=4 -> ~cruise (carry speed)
    a_brake, _ = _line_guidance(np.array([6.0, 0.0, -2.0]), np.array([4.0, 0.0]),
                                np.array([0.0, 0.0, -2.0]), np.array([10.0, 0.0, -2.0]),
                                cam, lat, g, brake_to_stop=True)
    # linear profile (KV*4=2.8 < 4) would already be braking here
    a_lin, _ = _line_guidance(np.array([6.0, 0.0, -2.0]), np.array([4.0, 0.0]),
                              np.array([0.0, 0.0, -2.0]), np.array([10.0, 0.0, -2.0]),
                              cam, lat, g, brake_to_stop=False)
    assert a_brake > a_lin                 # stop-profile carries speed; linear brakes early (creep)
    assert a_lin < 0


def test_brake_to_stop_hard_brake_near_target():
    g = NavGains(); g.MAX_SPEED = 5.0; g.DECEL_MAX = 2.0
    cam = np.array([1.0, 0.0]); lat = np.array([0.0, 1.0])
    # 0.5 m out at 3 m/s: sqrt(2*2*0.5)=1.41 < 3 -> decelerate hard
    a_al, _ = _line_guidance(np.array([9.5, 0.0, -2.0]), np.array([3.0, 0.0]),
                             np.array([0.0, 0.0, -2.0]), np.array([10.0, 0.0, -2.0]),
                             cam, lat, g, brake_to_stop=True)
    assert a_al < 0


def test_line_guidance_off_line_corrects_back():
    g = NavGains()
    cam_live = np.array([1.0, 0.0]); lat_course = np.array([0.0, 1.0])
    # leg along +x; drone 1 m off to +y, at rest -> cross term pushes back toward the line (-y)
    a_al, a_lat = _line_guidance(np.array([3.0, 1.0, -2.0]), np.array([0.0, 0.0]),
                                 np.array([0.0, 0.0, -2.0]), np.array([10.0, 0.0, -2.0]),
                                 cam_live, lat_course, g)
    assert a_lat < 0


def test_line_guidance_diagonal_stays_on_segment():
    g = NavGains()
    cam_live = np.array([1.0, 0.0]); lat_course = np.array([0.0, 1.0])
    d = 1.0 / np.sqrt(2.0)
    # diagonal leg (0,0)->(10,10); drone ON it at (3,3) moving along -> both course components drive +,
    # cross ~0 (a STRAIGHT diagonal, the property point-seeking lacked)
    a_al, a_lat = _line_guidance(np.array([3.0, 3.0, -2.0]), np.array([d, d]),
                                 np.array([0.0, 0.0, -2.0]), np.array([10.0, 10.0, -2.0]),
                                 cam_live, lat_course, g)
    assert a_al > 0 and a_lat > 0
    np.testing.assert_allclose(a_al, a_lat, atol=1e-6)   # symmetric diagonal -> equal course components


def test_course_guidance_brakes_when_overspeed():
    # moving fast forward, target close ahead -> along accel goes negative (brake), not accelerate
    g = NavGains()
    cam_live = np.array([1.0, 0.0]); lat_course = np.array([0.0, 1.0])
    a_al, _ = _course_guidance(pos=np.zeros(3), vel=np.array([5.0, 0.0, 0.0]),
                               target=np.array([2.0, 0.0, -2.0]),
                               cam_live=cam_live, lat_course=lat_course, gains=g)
    assert a_al < 0


def test_set_origin_computes_course_axes():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0], quat=(0, 0, 0, 1)))
    nav.set_origin(pos_ned=np.array([0, 0, -2.0]), yaw=0.0)   # raw spawn yaw 0
    np.testing.assert_allclose(nav._cam_live, [-1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(nav._lat_course, [0.0, -1.0], atol=1e-9)


def test_yaw_ref_tf_hold_and_fixed_are_yaw0t():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL))
    nav.set_origin()
    ds = nav.store.get_drone()
    assert nav._yaw_ref_tf("hold", ds) == nav._yaw0_t
    assert nav._yaw_ref_tf("fixed", ds) == nav._yaw0_t


def test_yaw_ref_tf_course_tracks_travel_direction():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL))
    nav.set_origin()
    nav._s_lat = 1.0
    ds = nav.store.get_drone()
    # travel along cam_live -> camera already aligned -> yaw_ref = yaw0_t
    nav._cur_travel = nav._cam_live.copy()
    assert abs(nav._yaw_ref_tf("course", ds) - nav._yaw0_t) < 1e-9
    # travel = cam_live rotated +90deg -> yaw_ref = yaw0_t + s_lat*(pi/2)
    nav._cur_travel = np.array([-nav._cam_live[1], nav._cam_live[0]])
    assert abs(nav._yaw_ref_tf("course", ds) - (nav._yaw0_t + np.pi / 2)) < 1e-6
    # s_lat = -1 mirrors the rotation sense
    nav._s_lat = -1.0
    assert abs(nav._yaw_ref_tf("course", ds) - (nav._yaw0_t - np.pi / 2)) < 1e-6


def test_strafe_attitude_level_hover_zero_rate():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL))
    nav.set_origin()
    nav._s_lat = 1.0
    nav.gains.Z_FF = 0.0          # isolate the recompose/attitude from the collective feed-forward
    ds = nav.store.get_drone()
    rate, thr, tilt, dbg = nav._strafe_attitude(ds, a_al=0.0, a_lat=0.0, z_sp=-2.0, yaw_ref_tf=0.0)
    assert abs(thr - 0.5) < 1e-6 and abs(tilt) < 1e-6
    np.testing.assert_allclose(rate, [0, 0, 0], atol=1e-6)

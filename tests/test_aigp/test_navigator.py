import json
import numpy as np
from aigp.navigator import NavGains, load_plant, attitude_command
from aigp.state import DroneState
from aigp.navigator import cruise_accel, settle_accel

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


from aigp.navigator import WaypointNavigator


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

from aigp.gate_detect import GateDetection
from aigp.visual_servo import servo, ServoCfg, ServoCmd

CFG = ServoCfg()  # defaults; signs +1 (pinned later in bring-up)


def _det(u, v):
    return GateDetection(u, v, 40, 40, 1600, (int(u) - 20, int(v) - 20, 40, 40))


def test_centered_no_correction():
    cmd = servo(_det(320, 180), yaw_cur=1.0, z_cur=-5.0, cfg=CFG)
    assert cmd.have_gate
    assert abs(cmd.yaw_sp - 1.0) < 1e-6
    assert abs(cmd.z_sp - (-5.0)) < 1e-6
    assert cmd.fwd_speed == CFG.fwd_speed


def test_gate_right_yaws_positive():
    cmd = servo(_det(480, 180), yaw_cur=0.0, z_cur=-5.0, cfg=CFG)
    ex = (480 - 320) / 320.0
    assert abs(cmd.yaw_sp - CFG.k_yaw * CFG.sign_x * ex) < 1e-6


def test_gate_low_descends():
    cmd = servo(_det(320, 300), yaw_cur=0.0, z_cur=-5.0, cfg=CFG)
    ey = (300 - 180) / 320.0
    # z_cur=-5.0 is NED z (positive-DOWN). Gate below centre -> ey>0 -> dz>0 ->
    # NED z increases from -5.0 toward 0 = descend. So z_sp > -5.0 is correct.
    assert cmd.z_sp > -5.0
    assert abs((cmd.z_sp + 5.0) - CFG.k_alt * CFG.sign_y * ey) < 1e-6


def test_no_gate_hovers_without_last():
    cmd = servo(None, yaw_cur=0.5, z_cur=-5.0, cfg=CFG)
    assert not cmd.have_gate and cmd.fwd_speed == 0.0 and cmd.yaw_sp == 0.5


def test_no_gate_coasts_with_last():
    last = ServoCmd(CFG.fwd_speed, 0.7, -4.0, True)
    cmd = servo(None, yaw_cur=0.0, z_cur=-5.0, cfg=CFG, last=last)
    assert cmd.yaw_sp == 0.7 and cmd.z_sp == -4.0
    assert 0.0 < cmd.fwd_speed < CFG.fwd_speed

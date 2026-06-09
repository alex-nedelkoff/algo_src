"""corner_ff.py -- pure helpers for the cornering controller: body-frame velocity + the weathervane
feedforward (cancel wv*v_body_y in the rate command, WV-DYNAMIC-validated coeffs). Pure numpy ->
unit-testable; the live frame/sign is verified on the dashboard (--wvff_sign, --no-wvff)."""
import numpy as np


def body_vel(R, vel_world):
    """World velocity -> body frame: v_body = R^T @ vel_world. R = body->world (quat_to_R)."""
    return np.asarray(R, float).T @ np.asarray(vel_world, float)


def weathervane_ff(vbx, vby, roll_wv0, roll_wv1, yaw_wv, g_roll, g_yaw, gain=1.0, sign=1.0):
    """Rate-command offset (wcmd units) that cancels the weathervane moment.
    Model: om_roll += (roll_wv0+roll_wv1*vbx)*vby*dt ; om_yaw += yaw_wv*vby*dt. Command goes through
    om=G*wcmd, so divide by the per-axis rate-loop gain. gain/sign tuned + verified live."""
    roll_ff = -sign * gain * (roll_wv0 + roll_wv1 * vbx) * vby / g_roll
    yaw_ff = -sign * gain * yaw_wv * vby / g_yaw
    return float(roll_ff), float(yaw_ff)

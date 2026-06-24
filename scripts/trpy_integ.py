"""COR-138: integrate the direct-TRPY inner loop into the navigator's PROVEN frame chain.
Reuse everything (set_origin -> s_cam/cam_live/lat_course, s_lat probe, _line_guidance ->
_strafe_recompose -> attitude_command_tf's q_des) and swap ONLY the final step: instead of
q_des -> body-rate -> sim rate loop, do q_des -> attitude error -> torque -> Minv -> raw motors.
Monkeypatch nav._strafe_attitude + redirect the commander to send_motor_command. Then fly a goto.
"""
import os, sys, time
import numpy as np
from pymavlink import mavutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from aigp.commander import Commander
from aigp.control_math import attitude_error_quat
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import (WaypointNavigator, NavGains, _qfix, WFIX, _strafe_recompose,
                            _z_int_step, attitude_command_tf)
from aigp.state import Store

IDLE=mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE; HOVER=0.234
KP_ATT_T, KD_ATT_T = 8.0, 2.0   # direct-TRPY attitude torque gains (from the hover/held-tilt validation)

def idle(m,boot):
    m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot,m.conn.target_system,
                                        m.conn.target_component,IDLE,[1.,0,0,0],0,0,0,0)
def fresh_start(s,c,m,boot):
    t=time.time()
    while time.time()-t<1.0: idle(m,boot); time.sleep(0.02)
    prev=s.get_race(); pb=prev["boot_ms"] if prev else None
    c.sim_reset(); t=time.time()
    while time.time()-t<30:
        idle(m,boot); r2=s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"]<pb)): break
        time.sleep(0.02)
    while time.time()-t<30:
        idle(m,boot); d=s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned)<3.0: return True
        time.sleep(0.02)
    return False
def tilt_deg(ds):
    R=quat_to_R(_qfix(ds.quat_wxyz)); return float(np.degrees(np.arccos(max(-1.,min(1.,R[2,2])))))
def calib(s,c,m,boot,i,bump=0.10):
    assert fresh_start(s,c,m,boot); c.arm(); u=np.full(4,HOVER); u[i]=HOVER+bump
    t0=time.time(); best=np.zeros(3); bn=0.0
    while time.time()-t0<0.30:
        c.send_motor_command(u.tolist()); time.sleep(0.01); ds=s.get_drone()
        if ds is not None:
            om=np.asarray(ds.omega,float)*WFIX
            if np.linalg.norm(om)>bn: bn=float(np.linalg.norm(om)); best=om.copy()
            if tilt_deg(ds)>50: break
    return best/bump

class TrpyCmd:
    """Wrap the real Commander: the navigator's send_attitude_target(motors, thr) -> send_motor_command."""
    def __init__(self, real): self.real=real
    def send_attitude_target(self, motors, thr): self.real.send_motor_command(np.asarray(motors,float).tolist())
    def arm(self): self.real.arm()
    def sim_reset(self): self.real.sim_reset()
    def send_motor_command(self, u): self.real.send_motor_command(u)

def make_trpy_strafe(nav, Minv):
    g=nav.gains; plant=nav.plant
    def _strafe(ds, a_al, a_lat, z_sp, yaw_ref_tf):
        now=time.time(); ze=float(z_sp)-float(ds.pos_ned[2])
        if nav._t_prev_zi is not None:
            dt=min(now-nav._t_prev_zi,0.05)
            nav._z_int=_z_int_step(nav._z_int,ze,dt,g.KI_Z,g.Z_INT_GATE,g.Z_INT_CLIP)
        nav._t_prev_zi=now
        fwd=np.array([np.cos(nav._yaw0_t),np.sin(nav._yaw0_t)])
        a_h=_strafe_recompose(a_al,a_lat,fwd,nav._s_lat)
        # reuse the navigator's frame-correct q_des + collective (ymirror/s_cam baked in)
        _,_,tilt,dbg=attitude_command_tf(ds,a_h,z_sp,yaw_ref_tf,plant,g,nav._s_cam,
                                         ymirror=nav._tf_ymirror,z_int=nav._z_int,z_ff=g.Z_FF)
        q_t=_qfix(ds.quat_wxyz); om_t=np.asarray(ds.omega,float)*WFIX
        ae=attitude_error_quat(q_t, dbg["q_des"])
        tau=KP_ATT_T*ae - KD_ATT_T*om_t
        motors=np.clip(dbg["thr"]+Minv@tau, 0., 1.)
        return motors, dbg["thr"], tilt, dbg
    return _strafe

def main():
    s=Store(); m=MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot=int(time.time()*1000); real=Commander(m.conn,boot)
    from aigp.navigator import load_plant; plant=load_plant("sysid/sim_response.json")
    M=np.array([calib(s,real,m,boot,i) for i in range(4)]).T; Minv=np.linalg.pinv(M)
    print(f"cond(M)={np.linalg.cond(M):.1f}", flush=True)
    assert fresh_start(s,real,m,boot)
    spawn=s.get_drone().pos_ned.copy()
    yaw0=float(np.arctan2(*(quat_to_R(s.get_drone().quat_wxyz)[[1,0],0])))

    g=NavGains(); g.MAX_SPEED=3.0; g.VLAT_MAX=1.5
    nav=WaypointNavigator(s, TrpyCmd(real), plant, gains=g)
    nav._strafe_attitude=make_trpy_strafe(nav, Minv)     # swap the inner loop -> direct-TRPY
    nav.set_origin(pos_ned=spawn, yaw=yaw0); real.arm()

    print("=== goto body(6,0,0) yaw=course via direct-TRPY inner loop ===", flush=True)
    t0=time.time()
    res=nav.goto(np.array([6.0,0.0,0.0]), yaw="course", v_cruise=3.0, engine="legs", frame="body")
    tgt=nav._resolve(np.array([6.0,0.0,0.0]),"body"); ds=s.get_drone()
    d=float(np.linalg.norm((tgt-ds.pos_ned)[:2]))
    print(f"  -> {res} in {time.time()-t0:.0f}s, final dist {d:.2f} m, tilt {tilt_deg(ds):.0f}", flush=True)
    print("=== RUNG 5: follow 3 wps (v3) ===", flush=True)
    t0=time.time(); res=nav.follow([np.array([4.0,3.0,-1.0]),np.array([8.0,-2.0,0.0]),np.array([3.0,-3.0,-1.5])],yaw="course",v_cruise=3.0,engine="legs",frame="body")
    print(f"  -> follow {res} in {time.time()-t0:.0f}s", flush=True)
    print("=== RUNG 6 signal: goto 8m @ v5 (sim-rate loop's runaway edge) ===", flush=True)
    nav.gains.MAX_SPEED=5.0; nav.gains.VLAT_MAX=1.5
    t0=time.time(); res=nav.goto(np.array([8.0,0.0,0.0]),yaw="course",v_cruise=5.0,engine="legs",frame="body")
    tgt=nav._resolve(np.array([8.0,0.0,0.0]),"body"); ds=s.get_drone()
    print(f"  -> v5 {res} in {time.time()-t0:.0f}s, dist {float(np.linalg.norm((tgt-ds.pos_ned)[:2])):.2f} m, tilt {tilt_deg(ds):.0f}", flush=True)

if __name__=="__main__":
    main()

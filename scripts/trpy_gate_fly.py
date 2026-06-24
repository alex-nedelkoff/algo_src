"""Fly the VQ1 race course (scored) on the direct-TRPY engine. Port of vq_track_wp's gate->waypoint
recipe, fixed to cross PERPENDICULAR through each gate aperture (through-points along the gate NORMAL
from its quat, not the spawn->gate chord) so the drone punches the hole instead of pinning the frame.
Pass = active_gate_index ticks. RESULT (2026-06-24): flies the full 157m VQ1 course (6 gates,
gate0 +8.5m up -> 26m descent) and SCORES gates on the direct-TRPY engine -- 6/6 best run, 3/6 typical
(gate-3/4 descent wall still flaky; teacher cracked it with gate-slowdown + per-gate raises). Keys:
perpendicular through-points (gate normal), gentle KP_Z (steep climb else saturates collective ->
runaway), and ZRAISE ~0.8m into the scoring tick zone (dz -0.43..-1.2, |lat|<0.8 -- project_vq_scoring_lpn).
Usage: python scripts/trpy_gate_fly.py [--v 2.5] [--ng 6] [--thru 2.5] [--zr 0.8]
"""
import os, sys, time, threading
import numpy as np
from pymavlink import mavutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from aigp.commander import Commander
from aigp.drone import Drone, FlightConfig
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import load_plant, calibrate_trpy_mixer, _qfix
from aigp.state import Store

IDLE=mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
def argf(f,d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
V=argf("--v",2.5); NG=int(argf("--ng",6)); THRU=argf("--thru",2.5); ZRAISE=argf("--zr",0.8)
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

def gate_normal(g, course_dir):
    """The gate's through-axis = the local axis (column of R_gate) most aligned with the course;
    signed so it points along travel."""
    R=quat_to_R(np.asarray(g.quat_ned_wxyz,float))
    al=[abs(float(R[:,i]@course_dir)) for i in range(3)]
    n=R[:,int(np.argmax(al))]
    return n if float(n@course_dir)>0 else -n

def main():
    s=Store(); m=MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot=int(time.time()*1000); real=Commander(m.conn,boot); plant=load_plant("sysid/sim_response.json")
    minv,M=calibrate_trpy_mixer(s, real, reset_fn=lambda: fresh_start(s,real,m,boot))
    print(f"cond(M)={np.linalg.cond(M):.1f}", flush=True)
    assert fresh_start(s,real,m,boot)
    gates=None; t=time.time()
    while gates is None and time.time()-t<8: gates=s.get_gates(); time.sleep(0.1)
    assert gates, "no gates"
    ds0=s.get_drone(); spawn=ds0.pos_ned.copy(); gi0=s.get_gate_idx()
    yaw0=float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1,0],0])))
    ng=min(NG,len(gates))
    print(f"gates={len(gates)} spawn={spawn.round(1)}", flush=True)

    # per gate: pre/center/post along the gate NORMAL (perpendicular crossing)
    wps=[]; prev=spawn.copy()
    for g in gates[:ng]:
        gp=np.asarray(g.pos_ned,float); cdir=gp-prev; cdir=cdir/max(np.linalg.norm(cdir),0.1)
        nrm=gate_normal(g, cdir)
        up=np.array([0.,0.,-ZRAISE])   # NED: -z = higher -> land in the tick zone
        wps.append(gp - THRU*nrm + up); wps.append(gp + up); wps.append(gp + THRU*nrm + up)
        print(f"  gate{g.id} pos={gp.round(1)} normal={nrm.round(2)} WxH={g.width:.1f}x{g.height:.1f}", flush=True)
        prev=gp
    print(f"{ng} gates -> {len(wps)} perpendicular through-wps, v={V}", flush=True)

    drone=Drone(s, real, plant, config=FlightConfig(vmax=max(V,3.0), vlat_max=min(max(V,2),3.0)), trpy_minv=minv)
    g=drone.nav.gains; g.KP_Z=0.6; g.KD_Z=1.8; g.TILT_MAX_DEG=22.0   # gentle climb -> no collective saturation
    drone.set_origin(pos_ned=spawn, yaw=yaw0); real.arm()
    drone.takeoff(2.0).wait(timeout=15)
    passes=[gi0]; stop=[False]
    def watch():
        while not stop[0]:
            gi=s.get_gate_idx()
            if gi>passes[-1]: passes.append(gi); print(f"  *** GATE {gi} PASSED t={time.time()-t0:.1f}s ***", flush=True)
            time.sleep(0.02)
    t0=time.time(); th=threading.Thread(target=watch,daemon=True); th.start()
    st=drone.follow([w for w in wps], yaw="course", speed=V, frame="world").wait(timeout=180)
    stop[0]=True; time.sleep(0.1)
    npass=passes[-1]-gi0
    print(f"\nDONE: follow={st.result.value}, GATES PASSED {npass}/{ng} (peak active_gate_index {passes[-1]})", flush=True)

if __name__=="__main__":
    main()

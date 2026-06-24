"""Fly the VQ1 race course (scored) on the direct-TRPY engine. Port of vq_track_wp's gate->waypoint
recipe, fixed to cross PERPENDICULAR through each gate aperture (through-points along the gate NORMAL
from its quat, not the spawn->gate chord) so the drone punches the hole instead of pinning the frame.
Pass = active_gate_index ticks. RESULT (2026-06-24): flies the full 157m VQ1 course (6 gates,
gate0 +8.5m up -> 26m descent) and SCORES gates on the direct-TRPY engine -- 6/6 best run, 3/6 typical
(gate-3/4 descent wall still flaky; teacher cracked it with gate-slowdown + per-gate raises). Keys:
perpendicular through-points (gate normal), gentle KP_Z (steep climb else saturates collective ->
runaway), and ZRAISE ~0.8m into the scoring tick zone (dz -0.43..-1.2, |lat|<0.8 -- project_vq_scoring_lpn).
COR-139: anticipatory gate-slowdown (vgate) -- carry --vmax on the between-gate straights, brake to
--vgate THROUGH each aperture (the teacher's gate-slowdown lever, now on TRPY where the low-speed
lateral limit cycle that made it counterproductive on the sim-rate plant -- WPTRACK-02 -- is gone).
Per-wp arrival speed via drone.follow(speeds=...). --zr accepts a per-gate comma-list (raise the
descent-wall gates 3/4 more if they undershoot the tick zone).
VISION Z-ANCHOR (COR-139): TRACK_INFO's absolute Z origin FLOATS per reset (~8.5 m; gate0 read z-8.5
one reset, z-0.05 the next -- the drone flew a ghost course offset from the real gates, 0/6). The
relative layout is consistent, so we pin it: localize gate0 with the camera at spawn (vq_gate_wp
pipeline, drone stationary looking forward) and shift the whole layout in Z so gate0 sits at its seen
height. Only Z is anchored -- TRACK x/y are stable across resets (~1 m) and single-shot vision RANGE
is biased (the teacher arc-calibrates for that), so anchoring vision-x would place gates short.
Usage: python scripts/trpy_gate_fly.py [--vmax 6] [--vgate 4] [--ng 6] [--thru 2.5] [--zr 0.8] [--noanchor]
"""
import os, sys, time, threading, json
import numpy as np
import cv2
from pymavlink import mavutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from aigp.commander import Commander
from aigp.drone import Drone, FlightConfig
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import load_plant, calibrate_trpy_mixer, _qfix
from aigp.state import Store
from aigp.gate_detect import load_params, red_mask, GateDetection
from fit_model import qfix
import aigp.flight_telemetry as ftm

IDLE=mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
def argf(f,d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
def args(f,d): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
VMAX=argf("--vmax",6.0); VGATE=argf("--vgate",4.0); NG=int(argf("--ng",6)); THRU=argf("--thru",2.5)
ZR=[float(x) for x in args("--zr","0.8").split(",")]   # per-gate raise: scalar or comma-list
ANCHOR="--noanchor" not in sys.argv                    # vision Z-anchor of the floating TRACK layout

# --- vq_gate_wp vision pipeline (copied VERBATIM, frame-critical) for the gate0 Z-anchor ---
GATE_AP=1.5; CX=320.0; FX=320.0; FY=320.0; CY=180.0; T20=np.radians(20.0)
H_DEFAULT=np.array([[1.,0,0],[0,-1.,0],[0,0,1.]])      # H_CANDS[1][1] = refl0 (y-flip), default until arc-calib
def r_opt_body_true(s_cam):
    S,C=np.sin(T20),np.cos(T20)
    return np.array([[0.,s_cam*S,s_cam*C],[s_cam,0.,0.],[0.,C,-S]])
def detect_gate(bgr,p):
    cnts,hier=cv2.findContours(red_mask(bgr,p),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    cands=[]
    for i,ct in enumerate(cnts):
        if hier[0][i][3]!=-1: continue
        area=float(cv2.contourArea(ct))
        if area<p["min_area_px"]: continue
        x,y,w,h=cv2.boundingRect(ct)
        if w==0 or h==0 or abs(w/float(h)-1.0)>p["square_tol"]: continue
        if (y+h/2.0)>p["max_v_frac"]*bgr.shape[0]: continue
        u_t,v_t=x+w/2.0,y+h/2.0; best_hole=0.0; hole_wh=None; j=hier[0][i][2]
        while j!=-1:
            ha=float(cv2.contourArea(cnts[j]))
            if ha>best_hole and ha>0.05*area:
                hx,hy,hw,hh=cv2.boundingRect(cnts[j]); u_t,v_t=hx+hw/2.0,hy+hh/2.0; best_hole=ha; hole_wh=(float(hw),float(hh))
            j=hier[0][j][0]
        if best_hole<=0.0: continue
        cands.append((GateDetection(u_t,v_t,float(w),float(h),area,(x,y,w,h)),hole_wh))
    return max(cands,key=lambda d:d[0].area) if cands else (None,None)   # spawn: biggest red square w/ hole = gate0
def localize_gate0_z(s,track_g0,secs=3.0):
    """Camera-localize gate0 at spawn (stationary, looking forward) -> its Z in the drone/odom frame
    (median over `secs`). Returns (vision_gate0_z, n_dets) or (None,0). Only Z is used (vision range is
    single-shot biased; x/y come from TRACK). Reuses est.point: gate = pos + Z*(H_DEFAULT @ ray_t)."""
    p=load_params(); ds0=s.get_drone()
    yaw0=float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1,0],quat_to_R(ds0.quat_wxyz)[0,0]))
    cam_live=-np.array([np.cos(yaw0),np.sin(yaw0)]); R_t0=quat_to_R(qfix(ds0.quat_wxyz))
    s_cam=1.0 if float(R_t0[:2,0]@cam_live)>0 else -1.0; R_OPT_T=r_opt_body_true(s_cam)
    zs=[]; t0=time.time()
    while time.time()-t0<secs:
        fr,_=s.get_frame(); bgr=fr[0] if fr else None; ds=s.get_drone()
        if bgr is not None and ds is not None:
            det,hole_wh=detect_gate(bgr,p)
            if det is not None and hole_wh is not None and max(hole_wh)>=6.0:
                d_opt=np.array([(det.u-CX)/FX,(det.v-CY)/FY,1.0]); d_opt/=np.linalg.norm(d_opt)
                ray_t=quat_to_R(qfix(ds.quat_wxyz))@(R_OPT_T@d_opt); Z=FX*GATE_AP/max(hole_wh)
                if Z<60.0: zs.append(float((ds.pos_ned+Z*(H_DEFAULT@ray_t))[2]))
        time.sleep(0.02)
    return (float(np.median(zs)),len(zs)) if zs else (None,0)
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
    # mixer M is a fixed property of the quad -> cache it (the calibration grinds the sim). --recal forces.
    CACHE="sysid/trpy_minv.json"
    if "--recal" not in sys.argv and os.path.exists(CACHE):
        minv=np.array(json.load(open(CACHE)),float); print(f"loaded cached minv {CACHE}", flush=True)
    else:
        minv,M=calibrate_trpy_mixer(s, real, reset_fn=lambda: fresh_start(s,real,m,boot))
        json.dump(minv.tolist(), open(CACHE,"w")); print(f"cond(M)={np.linalg.cond(M):.1f} -> cached {CACHE}", flush=True)
    assert fresh_start(s,real,m,boot)
    gates=None; t=time.time()
    while gates is None and time.time()-t<8: gates=s.get_gates(); time.sleep(0.1)
    assert gates, "no gates"
    ds0=s.get_drone(); spawn=ds0.pos_ned.copy(); gi0=s.get_gate_idx()
    yaw0=float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1,0],0])))
    ng=min(NG,len(gates))
    print(f"gates={len(gates)} spawn={spawn.round(1)}", flush=True)

    # VISION Z-ANCHOR: pin the floating TRACK_INFO Z origin by localizing gate0 with the camera at
    # spawn (drone stationary, looking forward). Shift the whole layout so gate0 sits at its seen Z.
    anchor=np.zeros(3)
    if ANCHOR:
        g0z_track=float(np.asarray(gates[0].pos_ned,float)[2])
        g0z_vis,nd=localize_gate0_z(s,gates[0])
        if g0z_vis is not None:
            anchor[2]=g0z_vis-g0z_track
            print(f"Z-ANCHOR: gate0 track_z={g0z_track:+.2f} vision_z={g0z_vis:+.2f} -> Tz={anchor[2]:+.2f} (n={nd})", flush=True)
        else:
            print("Z-ANCHOR: NO gate0 detection at spawn -> flying RAW track Z (likely offset)", flush=True)

    # per gate: pre/center/post along the gate NORMAL (perpendicular crossing). speeds = anticipatory
    # gate-slowdown (vgate): brake to VGATE for the pre+center (through the aperture), free-cruise out
    # (post=None -> accelerate onto the straight). The final post -> follow brakes to 0 (clean finish).
    wps=[]; speeds=[]; prev=spawn.copy()
    for gi,g in enumerate(gates[:ng]):
        gp=np.asarray(g.pos_ned,float)+anchor; cdir=gp-prev; cdir=cdir/max(np.linalg.norm(cdir),0.1)
        nrm=gate_normal(g, cdir)
        zr=ZR[gi] if gi<len(ZR) else ZR[-1]
        up=np.array([0.,0.,-zr])       # NED: -z = higher -> land in the tick zone
        wps.append(gp - THRU*nrm + up); speeds.append(VGATE)   # pre: brake into the aperture region
        wps.append(gp + up);            speeds.append(VGATE)   # center: through the hole at VGATE
        wps.append(gp + THRU*nrm + up); speeds.append(None)    # post: accelerate out onto the straight
        print(f"  gate{g.id} pos={gp.round(1)} normal={nrm.round(2)} zr={zr:.2f} WxH={g.width:.1f}x{g.height:.1f}", flush=True)
        prev=gp
    print(f"{ng} gates -> {len(wps)} perpendicular through-wps, vmax={VMAX} vgate={VGATE}", flush=True)

    flog=ftm.from_args(sys.argv, run_name="trpy_gate_fly", store=s)   # streams to the Mac Rerun (hard rule); --no-viz off
    drone=Drone(s, real, plant, config=FlightConfig(vmax=max(VMAX,3.0), vlat_max=min(max(VMAX,2),3.0)), flog=flog, trpy_minv=minv)
    g=drone.nav.gains; g.KP_Z=argf("--kpz",1.0); g.KD_Z=1.8; g.TILT_MAX_DEG=22.0   # gate0 is LEVEL (vision) -> no climb-saturation, track z harder on the descent
    drone.set_origin(pos_ned=spawn, yaw=yaw0); real.arm()
    drone.takeoff(2.0).wait(timeout=15)
    passes=[gi0]; stop=[False]; mref=[None]; STUCK_S=argf("--stuck",12.0)
    def watch():
        last_prog=time.time(); last_pos=spawn.copy()
        while not stop[0]:
            gi=s.get_gate_idx(); ds=s.get_drone(); now=time.time()
            if gi>passes[-1]:
                passes.append(gi); last_prog=now
                print(f"  *** GATE {gi} PASSED t={now-t0:.1f}s ***", flush=True)
            # stuck-abort: pinned (no gate tick + barely moving) for STUCK_S -> preempt so we don't
            # grind the frame for the whole 180 s follow timeout (the navigator only aborts on tilt).
            if ds is not None:
                if float(np.linalg.norm(ds.pos_ned-last_pos))>0.8: last_prog=now; last_pos=ds.pos_ned.copy()
                if now-last_prog>STUCK_S and mref[0] is not None:
                    print(f"  !!! STUCK {STUCK_S:.0f}s at gi={gi} (grinding) -> ABORT t={now-t0:.1f}s", flush=True)
                    mref[0].abort(); stop[0]=True; break
            time.sleep(0.05)
    t0=time.time(); th=threading.Thread(target=watch,daemon=True); th.start()
    mref[0]=drone.follow([w for w in wps], yaw="course", speed=VMAX, speeds=speeds, frame="world")
    st=mref[0].wait(timeout=180)
    stop[0]=True; time.sleep(0.1)
    if flog is not None: flog.close()
    npass=passes[-1]-gi0
    print(f"\nDONE: follow={st.result.value}, GATES PASSED {npass}/{ng} (peak active_gate_index {passes[-1]})", flush=True)

if __name__=="__main__":
    main()

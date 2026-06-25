"""COR-139 gate-order/frame probe: reset, then print everything about the gate layout to settle
ORDER (gates flown backward?) vs FRAME (TRACK absolute floats vs where the camera sees gate0).
Prints: spawn, active_gate_index, each gate (id, pos, horizontal dist from spawn, z), and the FULL
vision gate0 position (X,Y,Z from the camera) vs TRACK gate0. No flight. Usage: python scripts/probe_gates.py
"""
import os, sys, time
import numpy as np, cv2
from pymavlink import mavutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aigp.commander import Commander
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.geometry import quat_to_R
from aigp.gate_detect import load_params, red_mask, GateDetection
from fit_model import qfix

IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
GATE_AP=1.5; CX=320.0; FX=320.0; FY=320.0; CY=180.0; T20=np.radians(20.0)
H_DEFAULT=np.array([[1.,0,0],[0,-1.,0],[0,0,1.]])
def r_opt_body_true(sc):
    S,C=np.sin(T20),np.cos(T20); return np.array([[0.,sc*S,sc*C],[sc,0.,0.],[0.,C,-S]])
def detect_gate(bgr,p):
    cnts,hier=cv2.findContours(red_mask(bgr,p),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cands=[]
    for i,ct in enumerate(cnts):
        if hier[0][i][3]!=-1: continue
        area=float(cv2.contourArea(ct))
        if area<p["min_area_px"]: continue
        x,y,w,h=cv2.boundingRect(ct)
        if w==0 or h==0 or abs(w/float(h)-1.0)>p["square_tol"]: continue
        if (y+h/2.0)>p["max_v_frac"]*bgr.shape[0]: continue
        u_t,v_t=x+w/2.0,y+h/2.0; bh=0.0; hwh=None; j=hier[0][i][2]
        while j!=-1:
            ha=float(cv2.contourArea(cnts[j]))
            if ha>bh and ha>0.05*area:
                hx,hy,hw,hh=cv2.boundingRect(cnts[j]); u_t,v_t=hx+hw/2.0,hy+hh/2.0; bh=ha; hwh=(float(hw),float(hh))
            j=hier[0][j][0]
        if bh<=0.0: continue
        cands.append((GateDetection(u_t,v_t,float(w),float(h),area,(x,y,w,h)),hwh))
    return max(cands,key=lambda d:d[0].area) if cands else (None,None)


def main():
    s=Store(); m=MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot=int(time.time()*1000); c=Commander(m.conn,boot)
    def idle(): m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot,m.conn.target_system,m.conn.target_component,IDLE,[1.,0,0,0],0,0,0,0)
    prev=s.get_race(); pb=prev["boot_ms"] if prev else None
    c.sim_reset(); t=time.time()
    while time.time()-t<30:
        idle(); r2=s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"]<pb)): break
        time.sleep(0.02)
    t=time.time()
    while time.time()-t<30:
        idle(); d=s.get_drone()
        if s.get_race_live() and d is not None: break
        time.sleep(0.02)
    time.sleep(2.0)
    ds=s.get_drone(); gates=s.get_gates(); spawn=ds.pos_ned.copy()
    print(f"spawn={spawn.round(2)}  active_gate_index={s.get_gate_idx()}  race={s.get_race()}", flush=True)
    print("gates (id, pos, horiz_dist_from_spawn, z):", flush=True)
    for g in gates:
        gp=np.asarray(g.pos_ned,float); dh=float(np.linalg.norm(gp[:2]-spawn[:2]))
        print(f"  id{g.id} pos={gp.round(1)}  horiz_dist={dh:6.1f}  z={gp[2]:+6.1f}", flush=True)
    near=min(gates,key=lambda g:np.linalg.norm(np.asarray(g.pos_ned,float)[:2]-spawn[:2]))
    print(f"NEAREST gate to spawn = id{near.id} (active_gate_index={s.get_gate_idx()})", flush=True)
    # full vision gate0 (X,Y,Z) vs TRACK gate0
    p=load_params(); yaw0=float(np.arctan2(quat_to_R(ds.quat_wxyz)[1,0],quat_to_R(ds.quat_wxyz)[0,0]))
    cam_live=-np.array([np.cos(yaw0),np.sin(yaw0)]); R_t0=quat_to_R(qfix(ds.quat_wxyz))
    sc=1.0 if float(R_t0[:2,0]@cam_live)>0 else -1.0; ROPT=r_opt_body_true(sc); pts=[]
    t=time.time()
    while time.time()-t<3.0:
        fr,_=s.get_frame(); bgr=fr[0] if fr else None; d2=s.get_drone()
        if bgr is not None and d2 is not None:
            det,hwh=detect_gate(bgr,p)
            if det is not None and hwh is not None and max(hwh)>=6.0:
                do=np.array([(det.u-CX)/FX,(det.v-CY)/FY,1.0]); do/=np.linalg.norm(do)
                ray=quat_to_R(qfix(d2.quat_wxyz))@(ROPT@do); Z=FX*GATE_AP/max(hwh)
                if Z<60.0: pts.append(d2.pos_ned+Z*(H_DEFAULT@ray))
        time.sleep(0.02)
    if pts:
        v=np.median(np.array(pts),0)
        print(f"VISION sees the front gate at {v.round(1)} (drone frame); TRACK gate id0 = {np.asarray(gates[0].pos_ned,float).round(1)}", flush=True)
        print(f"  vision horiz_dist={np.linalg.norm(v[:2]-spawn[:2]):.1f}  (this is the REAL distance to the gate ahead)", flush=True)
    else:
        print("VISION: no gate detected ahead", flush=True)
    idle()


if __name__ == "__main__":
    main()

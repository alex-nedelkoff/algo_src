"""COR-138 task 1: diagnose the ~6s direct-TRPY hover instability -- WHICH axis diverges first.
Runs the proven hover (peak-M mixer, sp=+1, KP3.5/KD3.0, fixed alt sign) + logs roll/pitch/yaw
angles + body rates each step. Prints a per-0.5s table and flags the first axis to grow.
"""
import os, sys, time
import numpy as np
from pymavlink import mavutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import _qfix, WFIX
from aigp.state import Store

IDLE=mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE; HOVER=0.234
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
def rpy(R):
    return (np.degrees(np.arctan2(R[2,1],R[2,2])), np.degrees(np.arcsin(max(-1,min(1,-R[2,0])))),
            np.degrees(np.arctan2(R[1,0],R[0,0])))
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

def main():
    s=Store(); m=MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot=int(time.time()*1000); c=Commander(m.conn,boot)
    M=np.array([calib(s,c,m,boot,i) for i in range(4)]).T; Minv=np.linalg.pinv(M)
    print(f"cond(M)={np.linalg.cond(M):.1f}", flush=True)
    assert fresh_start(s,c,m,boot); c.arm()
    KP,KD,KPZ,KDZ=3.5,3.0,0.5,0.35
    ds=s.get_drone(); z_ref=float(ds.pos_ned[2]); yaw0=rpy(quat_to_R(_qfix(ds.quat_wxyz)))[2]
    LOG=[]; t0=time.time(); last=-1
    print("   t   roll  pitch  yaw(rel)  omx   omy   omz   tilt", flush=True)
    while time.time()-t0<15.0:
        ds=s.get_drone()
        if ds is not None:
            R=quat_to_R(_qfix(ds.quat_wxyz)); om=np.asarray(ds.omega,float)*WFIX
            r,p,y=rpy(R); yrel=((y-yaw0+180)%360)-180
            e=np.array([-float(R[2,1]),float(R[2,0])])   # sr=-1 sp=+1 (reduced-attitude righting)
            tau=np.array([KP*e[0]-KD*om[0],KP*e[1]-KD*om[1],-KD*om[2]])
            coll=HOVER+KPZ*(float(ds.pos_ned[2])-z_ref)+KDZ*float(ds.vel_ned[2])
            c.send_motor_command(np.clip(coll+Minv@tau,0.,1.).tolist())
            ti=tilt_deg(ds); now=time.time()-t0
            LOG.append((now,r,p,yrel,om[0],om[1],om[2],ti))
            k=int(now*2)
            if k!=last: last=k; print(f" {now:4.1f} {r:+5.1f} {p:+5.1f}  {yrel:+6.1f}  {om[0]:+4.1f} {om[1]:+4.1f} {om[2]:+4.1f}  {ti:4.1f}",flush=True)
            if ti>55: print(f"   tumbled @ {now:.1f}s",flush=True); break
        time.sleep(0.01)
    A=np.array(LOG)
    # which axis grows first: time each |signal| first exceeds a threshold after settle (t>1.5)
    print("\n=== onset analysis (first axis to break out, t>1.5s) ===", flush=True)
    names=["roll","pitch","yaw_rel","omx","omy","omz"]; ths=[10,10,15,1.0,1.0,1.0]
    for j,(nm,th) in enumerate(zip(names,ths)):
        col=np.abs(A[:,1+j]); idx=np.where((A[:,0]>1.5)&(col>th))[0]
        t_break = A[idx[0],0] if len(idx) else None
        print(f"   {nm:8s} |.|>{th}: {'t='+format(t_break,'.1f')+'s' if t_break is not None else 'never'}", flush=True)
    # yaw drift rate over the stable window (1.5-4s)
    w=(A[:,0]>1.5)&(A[:,0]<4.5)
    if w.sum()>5:
        dy=np.polyfit(A[w,0],A[w,3],1)[0]
        print(f"   yaw drift rate (1.5-4.5s) = {dy:+.1f} deg/s", flush=True)

if __name__=="__main__":
    main()

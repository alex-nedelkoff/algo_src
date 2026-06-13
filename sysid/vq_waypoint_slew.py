"""vq_waypoint_slew.py -- ANALYTIC waypoint guidance through the SLEW-LIMITED RATE interface (COR-127 RING-06 follow-up).

Question: now that the wire-rate slew limit makes the rate path stable (no tumble, 28 s
controlled flight in run 11), can the PROVEN analytic guidance (ff_batch, the RL teacher /
race_cruise lineage) fly waypoints directly on the rate path -- i.e. is RL needed for
NAVIGATION at all, or only for racing speed?

Reuses vq_deploy4's connection + live->ENU frame chain + rate mapping + slew. Guidance =
single-drone ff_batch toward an ENU waypoint sequence (same gate-spacing course as the deploy
test), with a gate-braking speed taper and anti-velocity (camera-forward) yaw. NO policy net.

  python vq_waypoint_slew.py [--maxw 6] [--thrmax 0.6] [--slew 40] [--space 14] [--turn 0.2] [--vdes 6] [--wpr 2.0]
"""
import sys, json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import quat_mul, desired_attitude, mat_to_quat, attitude_error_quat, collective_accel, accel_to_thrust_norm
import aigp.flight_telemetry as ftm
from aigp.recorder import Recorder

def argf(f, d): return float(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
G = 9.81; MAXW = argf("--maxw", 6.0); THRMAX = argf("--thrmax", 0.6); SLEW = argf("--slew", 40.0)
NG = int(argf("--gates", 6)); SPACE = argf("--space", 14.0); TURN = argf("--turn", 0.2)
VDES = argf("--vdes", 6.0); WPR = argf("--wpr", 2.0)
LEVEL_T = 3.0; MAX_T = 45.0
vqm = json.load(open("sysid/vq_model.json")); F0 = vqm["thrust"]["f0"]; DF = vqm["thrust"]["df_dthr"]
GAIN = np.array([vqm["rate_loop"][n]["gain_G"] for n in ("roll","pitch","yaw")])
HOVER_U0 = 2*((F0+G)/(-DF))-1
r = json.load(open("sysid/sim_response.json")); RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
# ff_batch gains (rl_finetune teacher, verbatim)
KP_POS=np.array([0.6,0.6,2.0]); KD_POS=np.array([1.2,1.2,3.0])
KP_ATT=np.array([6.0,6.0,4.0]); KD_ATT=np.array([1.2,1.2,0.0]); KP_YAW=4.0; KD_YAW=0.5
TILT_MAX=np.tan(np.radians(35))*G; ROLL_WV0,ROLL_WV1,YAW_WV=-0.105,-0.019,-0.149; YR_CAP=1.5

def qfix(q): return np.array([q[1],q[2],q[3],q[0]])
B=np.array([1.0,-1.0,-1.0]); bq=np.array([0.,1.,0.,0.]); bqc=np.array([0.,-1.,0.,0.])
def to_enu(ds):
    qt=qfix(ds.quat_wxyz); Rt=quat_to_R(qt); vw=Rt@ds.vel_ned; om=ds.omega*np.array([1.,-1.,1.])
    return ds.pos_ned*B, vw*B, quat_mul(quat_mul(bq,qt),bqc), om*B
def mat_to_quat1(m):
    t=m[0,0]+m[1,1]+m[2,2]
    if t>0:
        s=np.sqrt(t+1)*2; return np.array([.25*s,(m[2,1]-m[1,2])/s,(m[0,2]-m[2,0])/s,(m[1,0]-m[0,1])/s])
    if m[0,0]>=m[1,1] and m[0,0]>=m[2,2]:
        s=np.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2; return np.array([(m[2,1]-m[1,2])/s,.25*s,(m[0,1]+m[1,0])/s,(m[0,2]+m[2,0])/s])
    if m[1,1]>=m[2,2]:
        s=np.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2; return np.array([(m[0,2]-m[2,0])/s,(m[0,1]+m[1,0])/s,.25*s,(m[1,2]+m[2,1])/s])
    s=np.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2; return np.array([(m[1,0]-m[0,1])/s,(m[0,2]+m[2,0])/s,(m[1,2]+m[2,1])/s,.25*s])

def ff_one(pe, ve, qe, ome, tgt_pos, tgt_vel, tgt_yaw):
    """single-drone ff_batch: ENU state + ENU target -> normalized u [thr,wx,wy,wz]."""
    a = KP_POS*(tgt_pos-pe) + KD_POS*(tgt_vel-ve)
    n=np.linalg.norm(a[:2]);
    if n>TILT_MAX: a[:2]*=TILT_MAX/n
    R=quat_to_R(qe); yaw=np.arctan2(R[1,0],R[0,0])
    t=a+np.array([0,0,G]); zb=t/np.linalg.norm(t)
    xc=np.array([np.cos(tgt_yaw),np.sin(tgt_yaw),0.0]); yb=np.cross(zb,xc); yb/=np.linalg.norm(yb); xb=np.cross(yb,zb)
    qd=mat_to_quat1(np.stack([xb,yb,zb],axis=1))
    w0,x0,y0,z0=qe; w1,x1,y1,z1=qd
    qer=np.array([w0*w1+x0*x1+y0*y1+z0*z1, w0*x1-x0*w1-y0*z1+z0*y1, w0*y1+x0*z1-y0*w1-z0*x1, w0*z1-x0*y1+y0*x1-z0*w1])
    if qer[0]<0: qer=-qer
    w=KP_ATT*(2*qer[1:4]) - KD_ATT*ome
    w[2]=KP_YAW*(((tgt_yaw-yaw+np.pi)%(2*np.pi))-np.pi) - KD_YAW*ome[2]
    vb=R.T@ve
    w[0]+=(ROLL_WV0+ROLL_WV1*vb[0])*vb[1]; w[2]+=-YAW_WV*vb[1]
    cos_t=max(R[2,2],0.5); c=(G+a[2])/cos_t
    thr=np.clip((F0+c)/(-DF),0.0,1.0)
    u=np.empty(4); u[0]=np.clip(2*thr-1,-1,1); u[1:4]=np.clip(w/GAIN/MAXW,-1,1)
    cap=YR_CAP/abs(GAIN[2])/MAXW; u[3]=np.clip(u[3],-cap,cap)
    return u

s=Store(); m=MavlinkIO(s); print("connecting...",flush=True); assert m.wait_heartbeat(10),"NO HEARTBEAT"
m.start(); VisionIO(s).start(); boot=int(time.time()*1000); c=Commander(m.conn,boot)
def idle(): m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot,m.conn.target_system,m.conn.target_component,IDLE,[1.,0,0,0],0,0,0,0)
def fresh_start():
    t=time.time()
    while time.time()-t<1.0: idle(); time.sleep(0.02)
    prev=s.get_race(); pb=prev["boot_ms"] if prev else None; c.sim_reset(); t=time.time()
    while time.time()-t<30:
        idle(); r2=s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"]<pb)): break
        time.sleep(0.02)
    while time.time()-t<30:
        idle(); d=s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned)<3.0: return True
        time.sleep(0.02)
    return False

print("fresh_start...",flush=True); assert fresh_start(),"not live"; c.arm()
ds0=s.get_drone(); pos0,_,_,_=to_enu(ds0)
camfwd=-(quat_to_R(qfix(ds0.quat_wxyz))[:,0])*B; camfwd[2]=0; camfwd/=(np.linalg.norm(camfwd)+1e-9)
hd0=float(np.arctan2(camfwd[1],camfwd[0]))   # course along CAMERA (matches vq_deploy4)
wps=[]; hd=hd0; p=pos0.copy()
for i in range(NG):
    hd+=TURN; p=p+SPACE*np.array([np.cos(hd),np.sin(hd),0.0]); wps.append(p.copy())
wps=np.array(wps)
Rd0=quat_to_R(ds0.quat_wxyz); yaw_flip_tgt=float(np.arctan2(Rd0[1,0],Rd0[0,0]))
print(f"spawn_enu={pos0.round(1)} wp0={wps[0].round(1)} maxw={MAXW} slew={SLEW} vdes={VDES} wpr={WPR}",flush=True)
rec_d=Recorder(s_store:=s,script="vq_waypoint_slew",mode="rate",notes="analytic ff_batch waypoints through the slew-limited rate interface (RING-06)",extra_meta={"cmd_layout":["wx","wy","wz","thrust"]})
flog=ftm.from_args(sys.argv,RG,run_name="vq_waypoint_slew",store=s)
t0=time.time(); wi=0; last=-1; slew_prev=None; reached=[]
while time.time()-t0<MAX_T and wi<NG:
    ds=s.get_drone()
    if ds is None: time.sleep(0.01); continue
    t=time.time()-t0; pe,ve,qe,ome=to_enu(ds)
    wp=wps[wi]; dist=float(np.linalg.norm((wp-pe)[:2]))
    if dist<WPR:
        reached.append((wi,round(t,1),round(dist,2))); print(f"  WP {wi+1}/{NG} t={t:.1f} dist={dist:.2f}",flush=True); wi+=1
        if wi>=NG: break
        continue
    if t<LEVEL_T:
        a=np.array([0.0,0.0,1.8*(ds0.pos_ned[2]-ds.pos_ned[2])-3.0*ds.vel_ned[2]])
        Rd=quat_to_R(ds.quat_wxyz); yc=float(np.arctan2(Rd[1,0],Rd[0,0]))
        yerr=(yaw_flip_tgt-yc+np.pi)%(2*np.pi)-np.pi
        qd=mat_to_quat(desired_attitude(a,yc)); wd=np.array([0.5,1.6,1.0])*attitude_error_quat(ds.quat_wxyz,qd)
        wd[2]=float(np.clip(3.0*yerr,-1.2,1.2))-0.3*float(ds.omega[2])
        thr=accel_to_thrust_norm(collective_accel(a,ds.quat_wxyz),0.2675,62.0)
        c.send_attitude_target(np.clip(wd/RG,-4,4),thr); phase="LVL"
    else:
        # guidance: gate-braking speed taper + anti-velocity (camera-forward) yaw
        dxy=(wp-pe)[:2]; dn=np.linalg.norm(dxy)+1e-6
        vtgt=np.clip(0.9*dist,2.5,VDES)
        tv=np.zeros(3); tv[:2]=vtgt*dxy/dn
        spd=np.linalg.norm(ve[:2])
        yaw_cur=np.arctan2(quat_to_R(qe)[1,0],quat_to_R(qe)[0,0])
        if "--noseyaw" in sys.argv:
            tgt_yaw=float(np.arctan2(dxy[1],dxy[0]))  # nose AT waypoint (stable wv branch, non-racing)
        else:
            tgt_yaw=np.arctan2(-ve[1],-ve[0]) if spd>1.0 else yaw_cur  # anti-velocity (camera-forward racing)
        u=ff_one(pe,ve,qe,ome,wp,tv,tgt_yaw); uc=np.clip(u,-1,1)
        thr=float((uc[0]+1)/2*THRMAX); rates=np.array([uc[1],-uc[2],-uc[3]])*MAXW
        if SLEW>0:
            if slew_prev is None: slew_prev=rates.copy()
            mx=SLEW*(1/72.0); rates=slew_prev+np.clip(rates-slew_prev,-mx,mx); slew_prev=rates.copy()
        c.send_attitude_target(rates,thr); phase="POL"
        rec_d.log([float(rates[0]),float(rates[1]),float(rates[2]),float(thr)])
    Rc=quat_to_R(ds.quat_wxyz); tilt=float(np.degrees(np.arccos(max(-1,min(1,Rc[2,2])))))
    vmag=float(np.linalg.norm(ds.vel_ned))
    if flog is not None: flog.push(t,ds,{"thr":thr if 'thr' in dir() else 0.0},cruise=VDES,running=s.get_race_live(),armed=True)
    if int(t*2)!=last:
        last=int(t*2); print(f"  t={t:4.1f} {phase} wp{wi} dist={dist:5.1f} v={vmag:4.1f} tilt={tilt:3.0f}",flush=True)
    if tilt>110 and t>LEVEL_T+1: print(f"  ABORT tilt={tilt:.0f} t={t:.1f} wp {wi}/{NG}",flush=True); break
    time.sleep(0.004)
idle(); rec_d.close()
if flog is not None: flog.close()
print(f"WAYPOINT-SLEW DONE: reached {wi}/{NG} waypoints | {reached}",flush=True)

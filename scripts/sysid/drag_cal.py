"""Replay-calibrate mid-band drag (ENVELOPE-01 method): sweep (Dx,qx), replay recorded cmds
through vq_matched, score |v| match vs live at t=1-5 (the approach band). Frame-invariant |v|."""
import json, numpy as np, sys
sys.path.insert(0,"/Users/alex/Documents/drone-ai-grand-prix/algo_src")
from sim.dynamics.vq_matched import VQMatchedDynamics, POS, VEL, QUAT, OMEGA
from scripts.sysid.closed_loop_policy import quat_to_R
R="/Users/alex/Documents/drone-ai-grand-prix/algo_src"; DT=1/72
base=json.load(open(f"{R}/sysid/vq_model.json"))
runs=["/tmp/vq_replay/run_brk2.npz","/tmp/vq_replay/run12_trueabort.npz","/tmp/vq_replay/20260612T103925_vq_deploy4.npz"]
def replay(model,d):
    dyn=VQMatchedDynamics(model,dt=DT,frame="NED"); s=dyn.reset(1)
    q0=d["quat"][0][[1,2,3,0]]; q0/=np.linalg.norm(q0)
    s[0,QUAT]=q0; s[0,VEL]=quat_to_R(q0)@d["vel"][0]; s[0,OMEGA]=d["omega"][0]*np.array([1.,-1.,1.])
    t=(d["t_us"]-d["t_us"][0])/1e6; cmd=d["cmd"]; out=[]
    for k in range(min(len(t)-1,int(6/DT))):
        dtk=t[k+1]-t[k]
        if not(0.3*DT<dtk<3*DT): out.append(np.linalg.norm(s[0,VEL])); continue
        a=np.array([[cmd[k,3],cmd[k,0],cmd[k,1],cmd[k,2]]]); s=dyn.step(s,a,dt=dtk); out.append(np.linalg.norm(s[0,VEL]))
    return np.array(out),t[:len(out)]
def score(Dx,qx):
    m=json.loads(json.dumps(base)); m["drag_linear_body"]["Dx"]=Dx; m["drag_linear_body"]["Dy"]=Dx
    m["drag_quadratic_body"]["qx"]=qx; m["drag_quadratic_body"]["qy"]=qx
    errs=[]
    for p in runs:
        d=np.load(p); sv,tt=replay(m,d); t=(d["t_us"]-d["t_us"][0])/1e6
        livev=np.linalg.norm(d["vel"],axis=1)
        for tk in (1,2,3,4,5):
            i=int(np.searchsorted(tt,tk))
            if i<len(sv): errs.append(sv[i]-np.interp(tk,t,livev))
    e=np.array(errs); return e.mean(), np.sqrt((e**2).mean())
print("Dx    qx     mean(sim-live)  rms   (current Dx0.1 qx0.032)")
for Dx,qx in [(0.075,0.028),(0.065,0.027),(0.06,0.026),(0.07,0.029)]:
    mb,rms=score(Dx,qx); print(f"{Dx:.2f}  {qx:.3f}   {mb:+6.2f}        {rms:5.2f}")

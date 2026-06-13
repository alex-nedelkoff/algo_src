"""From run-8's handoff state, step the offline closed loop AND read the live recorded
trajectory frame-by-frame. Both start identical. Print where they diverge + on what."""
import io, json, sys, zipfile
import numpy as np
sys.path.insert(0, "/Users/alex/Documents/drone-ai-grand-prix/algo_src")
import torch
from sim.dynamics.vq_matched import VQMatchedDynamics, POS, VEL, QUAT, OMEGA
from scripts.sysid.closed_loop_policy import quat_to_R, to_enu, rotxy, wrap, quat_to_euler, qfix, B

R = "/Users/alex/Documents/drone-ai-grand-prix/algo_src"
DT = 1.0/72.0; MAXW = 6.0; THRMAX = 0.6; NG = 6
ZVD_AMP = np.array([0.371, 0.476, 0.153]); ZVD_DELAY = (7, 7, 4); TURN = 0.2; SPACE = 14.0
d = np.load("/tmp/vq_replay/run8.npz")
t = (d["t_us"] - d["t_us"][0]) / 1e6
q_sp = d["quat"][0]; pos0_enu = d["pos"][0] * B
camfwd = -(quat_to_R(qfix(q_sp))[:, 0]) * B; camfwd[2]=0; camfwd/=np.linalg.norm(camfwd)+1e-9
hd0 = float(np.arctan2(camfwd[1], camfwd[0]))
gates,gyaws=[],[]; hd=hd0; p=pos0_enu.copy()
for _ in range(NG):
    hd+=TURN; p=p+SPACE*np.array([np.cos(hd),np.sin(hd),0.0]); gates.append(p.copy()); gyaws.append(hd)
gates=np.array(gates); gyaws=np.array(gyaws)
z = zipfile.ZipFile(f"{R}/sysid/ft_cap6rec3_v6c_best.zip")
sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
Wt = {k:v.numpy() for k,v in sd.items()}
def policy(o):
    h=o.astype(np.float64)
    for i in (0,2,4): h=np.tanh(Wt[f"mlp_extractor.policy_net.{i}.weight"]@h+Wt[f"mlp_extractor.policy_net.{i}.bias"])
    return Wt["action_net.weight"]@h+Wt["action_net.bias"]
k0 = 0
model = json.load(open(f"{R}/sysid/vq_model.json"))
dyn = VQMatchedDynamics(model, dt=DT, frame="NED")
def build_obs(qc, vw, pos, om, gi):
    quat_live=qc[[3,0,1,2]]; oml=om*np.array([1.0,-1.0,1.0]); Rb=quat_to_R(qc); vl=Rb.T@vw
    pe,ve,qe,ome=to_enu(quat_live,vl,pos,oml); gate=gates[gi]; gi1=(gi+1)%NG; gyaw=gyaws[gi]
    roll,pitch,dyaw=quat_to_euler(qe); o=np.zeros(27,dtype=np.float32)
    o[0:2]=rotxy((pe-gate)[:2],gyaw); o[2]=pe[2]-gate[2]; o[3:5]=rotxy(ve[:2],gyaw); o[5]=ve[2]
    o[6]=roll;o[7]=pitch;o[8]=wrap(dyaw-gyaw);o[9:12]=ome; o[12:16]=0.5114015
    dg=gates[gi1]-gate; o[20:22]=rotxy(dg[:2],gyaw); o[22]=dg[2]; o[23]=wrap(gyaws[gi1]-gyaw)
    o[24]=2.0;o[25]=2.0;o[26]=12.0; return o, float(np.linalg.norm((gate-pe)[:2])), float(np.linalg.norm(ve))
s = dyn.reset(1)
qt=d["quat"][k0][[1,2,3,0]]; qt/=np.linalg.norm(qt)
s[0,QUAT]=qt; s[0,VEL]=quat_to_R(qt)@d["vel"][k0]; s[0,OMEGA]=d["omega"][k0]*np.array([1.0,-1.0,1.0]); s[0,POS]=d["pos"][k0]
prev=np.zeros(4); zbuf=np.zeros((15,3))
print(" fr |  SIM dist  v  tilt om | LIVE dist  v  tilt om | du(cmd)")
for k in range(240):
    kl = k0 + k
    if kl+1 >= len(t): break
    qc=s[0,QUAT]
    obs,distS,vS=build_obs(qc, s[0,VEL], s[0,POS], s[0,OMEGA], 0)
    obs[0:2]=np.clip(obs[0:2],-44,44); obs[2]=np.clip(obs[2],-8,8)
    uc=np.clip(policy(obs),-1,1); prev=uc
    tiltS=np.degrees(np.arccos(np.clip(quat_to_R(qc)[2,2],-1,1))); omS=np.linalg.norm(s[0,OMEGA])
    # live
    ql=d["quat"][kl][[1,2,3,0]]; ql/=np.linalg.norm(ql)
    vwl=quat_to_R(ql)@d["vel"][kl]
    _,distL,vL=build_obs(ql, vwl, d["pos"][kl], d["omega"][kl]*np.array([1.0,-1.0,1.0]), 0)
    tiltL=np.degrees(np.arccos(np.clip(quat_to_R(ql)[2,2],-1,1))); omL=np.linalg.norm(d["omega"][kl])
    ul=d["cmd"][kl]   # [wx,wy,wz,thr] recorded (post-ZVD wire cmd)
    if k%12==0:
        print(f"{k:3d} | {distS:6.1f} {vS:4.1f} {tiltS:4.0f} {omS:3.0f} | {distL:6.1f} {vL:4.1f} {tiltL:4.0f} {omL:3.0f} | uc_thr={(uc[0]+1)/2*THRMAX:.2f} live_thr={ul[3]:.2f}")
    thr=float((uc[0]+1)/2*THRMAX); rates=np.array([uc[1],-uc[2],-uc[3]])*MAXW
    zbuf=np.roll(zbuf,1,axis=0);zbuf[0]=rates
    rates=np.array([ZVD_AMP[0]*zbuf[0,ax]+ZVD_AMP[1]*zbuf[ZVD_DELAY[ax],ax]+ZVD_AMP[2]*zbuf[2*ZVD_DELAY[ax],ax] for ax in range(3)])
    s=dyn.step(s,np.array([[thr,*rates]]))

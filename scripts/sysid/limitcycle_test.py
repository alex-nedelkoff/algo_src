"""Is the live 11 Hz a closed-loop limit cycle (gain x loop-delay) or a plant ring?
Run cap6brake from the run-9 handoff on a WELL-DAMPED plant (rate_loop_2nd removed ->
plain MIMO, no artificial resonance) + an obs delay. FFT the turn-in omega. Sweep maxw
and delay: a limit cycle scales with both; a plant ring doesn't appear at all here."""
import io, json, sys, zipfile
import numpy as np
sys.path.insert(0, "/Users/alex/Documents/drone-ai-grand-prix/algo_src")
import torch
from sim.dynamics.vq_matched import VQMatchedDynamics, POS, VEL, QUAT, OMEGA
from scripts.sysid.closed_loop_policy import quat_to_R, to_enu, rotxy, wrap, quat_to_euler, qfix, B

R="/Users/alex/Documents/drone-ai-grand-prix/algo_src"; DT=1/72; THRMAX=0.6; NG=6; TURN=0.2; SPACE=14.0
ZVD_AMP=np.array([0.371,0.476,0.153]); ZVD_DELAY=(7,7,4)
d=np.load("/tmp/vq_replay/run9.npz"); t=(d["t_us"]-d["t_us"][0])/1e6
q_sp=d["quat"][0]; pos0=d["pos"][0]*B
cf=-(quat_to_R(qfix(q_sp))[:,0])*B; cf[2]=0; cf/=np.linalg.norm(cf)+1e-9
hd0=float(np.arctan2(cf[1],cf[0])); gates,gyaws=[],[]; hd=hd0; p=pos0.copy()
for _ in range(NG): hd+=TURN; p=p+SPACE*np.array([np.cos(hd),np.sin(hd),0.0]); gates.append(p.copy()); gyaws.append(hd)
gates=np.array(gates); gyaws=np.array(gyaws)
z=zipfile.ZipFile(f"{R}/sysid/ft_cap6brake_v6c_best.zip"); sd=torch.load(io.BytesIO(z.read("policy.pth")),map_location="cpu")
Wt={k:v.numpy() for k,v in sd.items()}
def policy(o):
    h=o.astype(np.float64)
    for i in (0,2,4): h=np.tanh(Wt[f"mlp_extractor.policy_net.{i}.weight"]@h+Wt[f"mlp_extractor.policy_net.{i}.bias"])
    return Wt["action_net.weight"]@h+Wt["action_net.bias"]
# well-damped model: strip rate_loop_2nd
m0=json.load(open(f"{R}/sysid/vq_model.json")); m_damped=json.loads(json.dumps(m0)); m_damped.pop("rate_loop_2nd",None)

def run(model, MAXW, odelay, zvd=True):
    dyn=VQMatchedDynamics(model,dt=DT,frame="NED"); s=dyn.reset(1)
    qt=d["quat"][0][[1,2,3,0]]; qt/=np.linalg.norm(qt)
    s[0,QUAT]=qt; s[0,VEL]=quat_to_R(qt)@d["vel"][0]; s[0,OMEGA]=d["omega"][0]*np.array([1.,-1.,1.]); s[0,POS]=d["pos"][0]
    prev=np.zeros(4); zbuf=np.zeros((15,3)); hist=[]; omlog=[]; gi=0; mind=99
    for k in range(int(8/DT)):
        hist.append(s.copy())
        so=hist[max(0,len(hist)-1-odelay)]
        qc=so[0,QUAT]; ql=qc[[3,0,1,2]]; oml=so[0,OMEGA]*np.array([1.,-1.,1.]); Rb=quat_to_R(qc); vl=Rb.T@so[0,VEL]
        pe,ve,qe,ome=to_enu(ql,vl,so[0,POS],oml); gate=gates[gi]; gi1=(gi+1)%NG; gyaw=gyaws[gi]
        roll,pitch,dyaw=quat_to_euler(qe); o=np.zeros(27,dtype=np.float32)
        o[0:2]=rotxy((pe-gate)[:2],gyaw); o[2]=pe[2]-gate[2]; o[3:5]=rotxy(ve[:2],gyaw); o[5]=ve[2]
        o[6]=roll;o[7]=pitch;o[8]=wrap(dyaw-gyaw);o[9:12]=ome; o[12:16]=0.5114015; o[16:20]=prev
        dg=gates[gi1]-gate; o[20:22]=rotxy(dg[:2],gyaw); o[22]=dg[2]; o[23]=wrap(gyaws[gi1]-gyaw)
        o[24]=2.;o[25]=2.;o[26]=12.; o[0:2]=np.clip(o[0:2],-44,44); o[2]=np.clip(o[2],-8,8)
        uc=np.clip(policy(o),-1,1); prev=uc
        thr=float((uc[0]+1)/2*THRMAX); rates=np.array([uc[1],-uc[2],-uc[3]])*MAXW
        if zvd:
            zbuf=np.roll(zbuf,1,axis=0); zbuf[0]=rates
            rates=np.array([ZVD_AMP[0]*zbuf[0,ax]+ZVD_AMP[1]*zbuf[ZVD_DELAY[ax],ax]+ZVD_AMP[2]*zbuf[2*ZVD_DELAY[ax],ax] for ax in range(3)])
        # truth omega (not delayed) for the FFT
        omlog.append(s[0,OMEGA].copy())
        dist=float(np.linalg.norm((gate-pe)[:2]))
        if gi==0: mind=min(mind,dist)
        if dist<1.5: gi+=1
        if gi>=NG: break
        s=dyn.step(s,np.array([[thr,*rates]]))
    om=np.array(omlog)
    # FFT pitch omega over the turn-in window (last 1.5s before end)
    seg=om[-108:,1] if len(om)>=108 else om[:,1]; seg=seg-seg.mean()
    if len(seg)<32: return gi,mind,0,0
    f=np.fft.rfftfreq(len(seg),DT); P=np.abs(np.fft.rfft(seg))**2; mask=f>2
    pk=f[mask][np.argmax(P[mask])]; rms=np.std(seg)
    return gi,mind,pk,rms

print("DAMPED plant (no rate_loop_2nd ring) + obs delay -- does an 11 Hz osc EMERGE?")
print(" maxw delay | gates mindist  FFTpeak  omRMS")
for MAXW in (6,4,3):
    for od in (0,1,2,3):
        gi,md,pk,rms=run(m_damped,MAXW,od)
        print(f"  {MAXW}    {od}    |  {gi}/6   {md:5.1f}   {pk:5.1f}Hz  {rms:.2f}")

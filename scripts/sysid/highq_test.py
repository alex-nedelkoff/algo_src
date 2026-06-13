"""Build high-Q ring variants (poles -> measured zeta~0.14, |p|~0.86-0.90) and run the
offline closed loop from run-8 handoff. Does a higher-Q model reproduce the live stall?"""
import json, numpy as np, subprocess, sys
R = "/Users/alex/Documents/drone-ai-grand-prix/algo_src"
m0 = json.load(open(f"{R}/sysid/vq_model.json"))
r2 = m0["rate_loop_2nd"]
A1 = np.array(r2["A1"]); A2 = np.array(r2["A2"])
def poles(A1,A2):
    out=[]
    for ax in range(3):
        a1,a2=A1[ax,ax],A2[ax,ax]; r=np.sqrt(max(-a2,1e-9)); ct=a1/(2*r); th=np.arccos(np.clip(ct,-1,1))
        out.append((r, th*72/(2*np.pi)))
    return out
print("current poles |r|,f:", [(round(r,3),round(f,1)) for r,f in poles(A1,A2)])
for rtarget in (0.86, 0.92):
    A1b=A1.copy(); A2b=A2.copy()
    for ax in range(3):
        r0,f=poles(A1b,A2b)[ax]; th=2*np.pi*f/72
        A2b[ax,ax]=-rtarget**2; A1b[ax,ax]=2*rtarget*np.cos(th)
    mv=json.loads(json.dumps(m0)); mv["rate_loop_2nd"]["A1"]=A1b.tolist(); mv["rate_loop_2nd"]["A2"]=A2b.tolist()
    p=f"/tmp/vq_model_q{int(rtarget*100)}.json"; json.dump(mv,open(p,"w"))
    print(f"wrote {p}  poles now |{rtarget}|")

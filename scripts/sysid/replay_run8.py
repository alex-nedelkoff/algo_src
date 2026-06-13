"""Closed-loop policy from run-8's recorded spawn, course rebuilt exactly as vq_deploy4
anchors it (camfwd from spawn quat). ZVD + obs-clip on. Compare to live dist trajectory."""
import io, json, sys, zipfile
import numpy as np
sys.path.insert(0, "/Users/alex/Documents/drone-ai-grand-prix/algo_src")
import torch
from sim.dynamics.vq_matched import VQMatchedDynamics, POS, VEL, QUAT, OMEGA
from scripts.sysid.closed_loop_policy import quat_to_R, to_enu, rotxy, wrap, quat_to_euler, qfix, B

R = "/Users/alex/Documents/drone-ai-grand-prix/algo_src"
DT = 1.0/72.0; MAXW = 6.0; THRMAX = 0.6; NG = 6; GATE_R = 1.5
ZVD_AMP = np.array([0.371, 0.476, 0.153]); ZVD_DELAY = (7, 7, 4); TURN = 0.2; SPACE = 14.0
d = np.load("/tmp/vq_replay/run8.npz")
# find handoff row (first POL): vq_deploy4 LVL until t=3.0; recorder logs from spawn.
# spawn = row 0. Course anchored at spawn (vq_deploy4: pos0 = to_enu(ds0)).
q_sp = d["quat"][0]; pos_sp = d["pos"][0]
pos0_enu = pos_sp * B
camfwd = -(quat_to_R(qfix(q_sp))[:, 0]) * B; camfwd[2] = 0; camfwd /= np.linalg.norm(camfwd)+1e-9
hd0 = float(np.arctan2(camfwd[1], camfwd[0]))   # vq_deploy4: course along CAMERA (no flip)
gates, gyaws = [], []; hd = hd0; p = pos0_enu.copy()
for _ in range(NG):
    hd += TURN; p = p + SPACE*np.array([np.cos(hd), np.sin(hd), 0.0])
    gates.append(p.copy()); gyaws.append(hd)
gates = np.array(gates); gyaws = np.array(gyaws)

z = zipfile.ZipFile(f"{R}/sysid/ft_cap6rec3_v6c_best.zip")
sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
Wt = {k: v.numpy() for k, v in sd.items()}
def policy(obs):
    h = obs.astype(np.float64)
    for i in (0, 2, 4):
        h = np.tanh(Wt[f"mlp_extractor.policy_net.{i}.weight"] @ h + Wt[f"mlp_extractor.policy_net.{i}.bias"])
    return Wt["action_net.weight"] @ h + Wt["action_net.bias"]

# start the closed loop at the LIVE handoff state (first POL frame ~ t=3.0; find it)
t = (d["t_us"] - d["t_us"][0]) / 1e6
k0 = int(np.searchsorted(t, t[0] + 3.0))
model = json.load(open(f"{R}/sysid/vq_model.json"))
dyn = VQMatchedDynamics(model, dt=DT, frame="NED")
s = dyn.reset(1)
q_t = d["quat"][k0][[1,2,3,0]]; q_t /= np.linalg.norm(q_t)
s[0, QUAT] = q_t; s[0, VEL] = quat_to_R(q_t) @ d["vel"][k0]
s[0, OMEGA] = d["omega"][k0]*np.array([1.0,-1.0,1.0]); s[0, POS] = d["pos"][k0]
prev_act = np.zeros(4); gi = 0; zbuf = np.zeros((15,3)); mind = 99
for k in range(int(28/DT)):
    tt = k*DT
    q_c = s[0, QUAT]; quat_live = q_c[[3,0,1,2]]
    om_live = s[0, OMEGA]*np.array([1.0,-1.0,1.0])
    Rb = quat_to_R(q_c); vel_live = Rb.T @ s[0, VEL]
    pe, ve, qe, ome = to_enu(quat_live, vel_live, s[0, POS], om_live)
    gate = gates[gi]; gi1 = (gi+1)%NG; gyaw = gyaws[gi]
    roll, pitch, dyaw = quat_to_euler(qe)
    obs = np.zeros(27, dtype=np.float32)
    obs[0:2] = rotxy((pe-gate)[:2], gyaw); obs[2] = pe[2]-gate[2]
    obs[3:5] = rotxy(ve[:2], gyaw); obs[5] = ve[2]
    obs[6] = roll; obs[7] = pitch; obs[8] = wrap(dyaw-gyaw); obs[9:12] = ome
    obs[12:16] = 0.5114015; obs[16:20] = prev_act
    dg = gates[gi1]-gate
    obs[20:22] = rotxy(dg[:2], gyaw); obs[22] = dg[2]; obs[23] = wrap(gyaws[gi1]-gyaw)
    obs[24]=2.0; obs[25]=2.0; obs[26]=12.0
    obs[0:2] = np.clip(obs[0:2], -44, 44); obs[2] = np.clip(obs[2], -8, 8)
    uc = np.clip(policy(obs), -1, 1); prev_act = uc
    thr = float((uc[0]+1)/2*THRMAX); rates = np.array([uc[1],-uc[2],-uc[3]])*MAXW
    zbuf = np.roll(zbuf,1,axis=0); zbuf[0]=rates
    rates = np.array([ZVD_AMP[0]*zbuf[0,ax]+ZVD_AMP[1]*zbuf[ZVD_DELAY[ax],ax]+ZVD_AMP[2]*zbuf[2*ZVD_DELAY[ax],ax] for ax in range(3)])
    dist = float(np.linalg.norm((gate-pe)[:2])); 
    if gi==0: mind=min(mind,dist)
    tilt = float(np.degrees(np.arccos(np.clip(Rb[2,2],-1,1))))
    if dist < GATE_R:
        gi+=1; print(f"  GATE {gi} t={tt:.1f} tilt={tilt:.0f}")
        if gi>=NG: break
    if tilt>110: print(f"  ABORT tilt={tilt:.0f} t={tt:.1f} gate{gi}"); break
    if int(tt*2)!=int((tt-DT)*2) and tt<7:
        print(f"  t={tt:.1f} gate{gi} dist={dist:.1f} v={np.linalg.norm(ve):.1f} tilt={tilt:.0f} beta_obs8={np.degrees(obs[8]):.0f}")
    s = dyn.step(s, np.array([[thr, *rates]]))
print(f"RUN8 CLOSED-LOOP FROM HANDOFF: reached {gi}/{NG}  min gate0 dist {mind:.1f}")

"""Run the of-record GateNet multi-instance pipeline over recorded VQ2 frames."""
import sys, os, json, glob, time
MAIN_REPO = os.environ.get('VQ2_MAIN_REPO', r'C:\Users\Administrator\Documents\algo_src_main')
if os.path.isdir(MAIN_REPO):
    sys.path.insert(0, MAIN_REPO)
sys.path.insert(0, os.environ.get('VQ2_DEPLOY_ROOT', r'C:\Users\Administrator'))
import numpy as np
import cv2
import torch
# fp8 shim for transformers 5.x on torch 2.5.1 (same as vq2wp.py)
for _fp8 in ('float8_e8m0fnu', 'float8_e4m3fn', 'float8_e5m2'):
    if not hasattr(torch, _fp8):
        setattr(torch, _fp8, torch.uint8)
from scripts.dcl import vq1_detect_overlay as OV
from perception.training import train_gatenet as TG
from perception.decode import associate as AS

CKPT = os.environ.get('GATENET_CKPT', r'C:\Users\Administrator\gatenet_b2_cov.pt')
CFG = os.environ.get('GATENET_CFG', os.path.join(
    MAIN_REPO, 'configs', 'perception', 'gatenet_b2_multi_pb_cov.yaml'))
FRAMES_DIR = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\Administrator\vq2_motion\frames'
OUT = os.environ.get('VQ2_DETECT_OUT', os.path.join(os.path.dirname(FRAMES_DIR), 'detections.jsonl'))

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lm = OV.load_model(CKPT, CFG, dev)
print('decode_path', lm.decode_path, 'stride', lm.stride, 'pad_to_h', lm.pad_to_h,
      'heads', lm.heads, flush=True)
dk, sk = OV._multi_decode_knobs(lm.cfg, stride=lm.stride)

frames = sorted(glob.glob(os.path.join(FRAMES_DIR, '*.jpg')))
print('frames:', len(frames), flush=True)
f_out = open(OUT, 'w')
t0 = time.time()
n_with_det = 0
for i, fp in enumerate(frames):
    bgr = cv2.imread(fp)
    if bgr is None:
        continue
    padded = OV.pad_bottom(bgr, lm.pad_to_h)
    bgr_t = torch.from_numpy(padded[None])
    with torch.no_grad():
        x = TG.frames_to_input(bgr_t, dev)
        out = TG._float_output(lm.model(x))
        insts = AS.decode_frame_multi(out, 0, **dk)
    img_w, img_h = int(x.shape[-1]), int(x.shape[-2])
    recs = []
    for dec in insts:
        ip = AS.solve_instance(dec, image_size=(img_w, img_h), direct_pose_vec=None, **sk)
        R = None
        if ip.pnp is not None and getattr(ip.pnp, 'R_cam_gate', None) is not None:
            R = np.asarray(ip.pnp.R_cam_gate).reshape(-1).round(6).tolist()
        recs.append({
            'center_xy': dec.center_xy.round(2).tolist(),
            'center_score': round(float(dec.center_score), 4),
            'corner_xy': dec.corner_xy.round(2).tolist(),
            'sigma_diag': np.sqrt(np.maximum(dec.sigma[:, [0, 1], [0, 1]], 0)).round(3).tolist(),
            'visibility': dec.visibility.round(3).tolist() if dec.visibility is not None else None,
            'usable': dec.usable.astype(int).tolist(),
            'suspect': dec.suspect.astype(int).tolist(),
            'solved': bool(ip.solved),
            'source': ip.source,
            't_cam': ip.t.round(4).tolist() if ip.t is not None else None,
            'R_cam_gate': R,
            'low_confidence': bool(ip.low_confidence),
            'n_corners_inframe': int(ip.n_corners_inframe),
        })
    if recs:
        n_with_det += 1
    f_out.write(json.dumps({'sim_ns': int(os.path.basename(fp)[:-4]), 'insts': recs}) + '\n')
    if (i + 1) % 200 == 0:
        print(f'{i+1}/{len(frames)} det-rate {n_with_det/(i+1):.2f} '
              f'{(time.time()-t0)/(i+1)*1000:.0f} ms/frame', flush=True)
f_out.close()
print(f'DONE {len(frames)} frames, {n_with_det} with detections, '
      f'{(time.time()-t0):.0f}s total -> {OUT}', flush=True)

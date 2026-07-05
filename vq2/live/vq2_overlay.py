import json, cv2, numpy as np
det = {}
for line in open(r'C:\Users\alexj\vq2_motion\detections.jsonl'):
    d = json.loads(line)
    if d['insts']:
        det[d['sim_ns']] = d['insts']
keys = sorted(det.keys())
picks = [keys[int(q * (len(keys) - 1))] for q in (0.05, 0.45, 0.8)]
for ns in picks:
    img = cv2.imread(rf'C:\Users\alexj\vq2_motion\frames\{ns}.jpg')
    for inst in det[ns]:
        pts = np.array(inst['corner_xy'])
        usable = np.array(inst['usable'], bool)
        col = (0, 255, 0) if inst['solved'] and not inst['low_confidence'] else (0, 200, 255)
        for j, (x, y) in enumerate(pts):
            if 0 <= x < 640 and 0 <= y < 384:
                cv2.circle(img, (int(x), int(y)), 4, col if usable[j] else (0, 0, 255), 1 + usable[j])
        cx, cy = inst['center_xy']
        cv2.drawMarker(img, (int(cx), int(cy)), col, cv2.MARKER_CROSS, 10, 2)
        if inst['t_cam']:
            r = np.linalg.norm(inst['t_cam'])
            cv2.putText(img, f"{r:.0f}m", (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.imwrite(rf'C:\Users\alexj\overlay_{ns}.jpg', img)
    print('wrote', ns, 'insts', len(det[ns]))

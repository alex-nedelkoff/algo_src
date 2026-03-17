"""Visualize covisibility pairs: annotate shared pixels, serve as HTML.

Usage (in Docker):
  python -m perception.training.visualize_pairs \
      --dataset replica --data-dir /data/Replica \
      --pairs /data/processed/replica/pairs/room0.parquet \
      --port 8888
"""

from __future__ import annotations

import argparse
import base64
import http.server
import logging
import socketserver
from pathlib import Path

import cv2
import numpy as np

from perception.training.data.covisibility import unproject_pixels
from perception.training.data.pairs import PairTable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _get_scene(dataset: str, data_dir: Path, scene_id: str):
    if dataset == "replica":
        from perception.training.data.replica.loader import ReplicaScene
        return ReplicaScene(data_dir, scene_id)
    raise ValueError(f"Unknown dataset: {dataset}")


def _compute_overlap_mask(
    scene, frame_i: int, frame_j: int, num_samples: int = 10_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute which pixels in frame_i are visible in frame_j and vice versa.

    Returns:
        uv_i: (M, 2) pixels in frame i that are visible in frame j
        uv_j_proj: (M, 2) where those pixels land in frame j
        uv_j: (M2, 2) pixels in frame j visible in frame i
        uv_i_proj: (M2, 2) where those pixels land in frame i
    """
    K = scene.intrinsics().astype(np.float64)
    poses = scene.poses().astype(np.float64)
    depth_i = scene.depth(frame_i)
    depth_j = scene.depth(frame_j)
    H, W = depth_i.shape

    rng = np.random.default_rng(42)

    results = []
    for src_depth, src_pose, dst_depth, dst_pose in [
        (depth_i, poses[frame_i], depth_j, poses[frame_j]),
        (depth_j, poses[frame_j], depth_i, poses[frame_i]),
    ]:
        pts_cam, pixel_coords = unproject_pixels(src_depth, K, num_samples, rng)
        S = len(pts_cam)
        if S == 0:
            results.append((np.empty((0, 2)), np.empty((0, 2))))
            continue

        pts_h = np.vstack([pts_cam.T, np.ones((1, S))])
        pts_world = src_pose @ pts_h
        T_dst_w = np.linalg.inv(dst_pose)
        pts_dst = T_dst_w @ pts_world
        z = pts_dst[2, :]

        valid = z > 0
        z_safe = np.where(valid, z, 1.0)

        uv_dst = K @ pts_dst[:3, :]
        u_dst = uv_dst[0, :] / z_safe
        v_dst = uv_dst[1, :] / z_safe

        u_int = np.round(u_dst).astype(np.int64)
        v_int = np.round(v_dst).astype(np.int64)
        valid &= (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

        valid_idx = np.where(valid)[0]
        if len(valid_idx) > 0:
            d_actual = dst_depth[v_int[valid_idx], u_int[valid_idx]].astype(np.float64)
            d_reproj = z[valid_idx]
            depth_ok = np.abs(d_reproj - d_actual) < 0.1 * d_actual
            depth_ok &= d_actual > 0
            valid[valid_idx] = depth_ok

        mask = valid
        src_uv = pixel_coords[mask].astype(np.int32)
        dst_uv = np.stack([u_int[mask], v_int[mask]], axis=1).astype(np.int32)
        results.append((src_uv, dst_uv))

    return results[0][0], results[0][1], results[1][0], results[1][1]


def _annotate_image(img: np.ndarray, points: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
    """Draw semi-transparent circles on overlap pixels."""
    out = img.copy()
    overlay = img.copy()
    for pt in points:
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 3, color, -1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def _img_to_base64(img: np.ndarray, quality: int = 80) -> str:
    """Encode BGR image as base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")


def _draw_correspondence_lines(
    img_i: np.ndarray, img_j: np.ndarray,
    uv_i: np.ndarray, uv_j_proj: np.ndarray,
    max_lines: int = 50,
) -> np.ndarray:
    """Create side-by-side image with correspondence lines."""
    H, W = img_i.shape[:2]
    canvas = np.hstack([img_i, img_j])

    if len(uv_i) == 0:
        return canvas

    rng = np.random.default_rng(0)
    n = min(max_lines, len(uv_i))
    idx = rng.choice(len(uv_i), size=n, replace=False)

    for k in idx:
        color = tuple(int(c) for c in rng.integers(100, 255, size=3))
        pt1 = (int(uv_i[k, 0]), int(uv_i[k, 1]))
        pt2 = (int(uv_j_proj[k, 0]) + W, int(uv_j_proj[k, 1]))
        cv2.circle(canvas, pt1, 4, color, -1)
        cv2.circle(canvas, pt2, 4, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

    return canvas


def generate_html(
    dataset: str,
    data_dir: Path,
    pairs_path: Path,
    samples_per_bin: int = 3,
) -> str:
    """Generate HTML page with annotated pair visualizations."""
    pairs = PairTable.load(pairs_path)
    scores = pairs.table.column("overlap_score").to_numpy()

    bins = [
        ("No Overlap (0.0)", 0.0, 0.0),
        ("Hard (0.05-0.3)", 0.05, 0.3),
        ("Medium (0.3-0.7)", 0.3, 0.7),
        ("Easy (0.7-1.0)", 0.7, 1.0),
    ]

    color_i = (0, 200, 0)   # green (BGR)
    color_j = (200, 100, 0) # blue (BGR)

    scene_cache = {}
    cards_html = ""
    rng = np.random.default_rng(42)

    # Build set of pairs that DO have overlap, for finding non-overlapping pairs
    all_scene_ids = pairs.table.column("scene_id").to_pylist()
    all_fi = pairs.table.column("frame_i").to_pylist()
    all_fj = pairs.table.column("frame_j").to_pylist()
    pair_set = set(zip(all_scene_ids, all_fi, all_fj))
    pair_set |= set(zip(all_scene_ids, all_fj, all_fi))
    scenes_in_table = set(all_scene_ids)

    for bin_name, lo, hi in bins:
        # Special case: no-overlap bin — sample random pairs NOT in the pair table
        if lo == 0.0 and hi == 0.0:
            cards_html += f"<h2>{bin_name}</h2>\n"
            no_overlap_samples = []
            for sid in scenes_in_table:
                if sid not in scene_cache:
                    scene_cache[sid] = _get_scene(dataset, data_dir, sid)
                sc = scene_cache[sid]
                attempts = 0
                while len(no_overlap_samples) < samples_per_bin and attempts < 200:
                    fi_ = int(rng.integers(0, sc.n_frames))
                    fj_ = int(rng.integers(0, sc.n_frames))
                    if fi_ != fj_ and (sid, fi_, fj_) not in pair_set:
                        no_overlap_samples.append((sid, fi_, fj_))
                    attempts += 1

            for sid, fi, fj in no_overlap_samples:
                scene = scene_cache[sid]
                score = 0.0
                log.info("Rendering %s frame %d <-> %d (no overlap)", sid, fi, fj)
                img_i = scene.rgb(fi)
                img_j = scene.rgb(fj)
                uv_i, uv_j_proj, uv_j, uv_i_proj = _compute_overlap_mask(scene, fi, fj, num_samples=5000)
                ann_i = _annotate_image(img_i, uv_i, color_i)
                ann_j = _annotate_image(img_j, uv_j, color_j)
                corr = _draw_correspondence_lines(img_i, img_j, uv_i, uv_j_proj)
                b64_ann_i = _img_to_base64(ann_i)
                b64_ann_j = _img_to_base64(ann_j)
                b64_corr = _img_to_base64(corr, quality=70)
                cards_html += f"""
<div class="card">
  <div class="header">
    <span class="scene">{sid}</span>
    frame {fi} &harr; frame {fj}
    <span class="score" style="background: hsl(0, 70%, 35%)">
      overlap: 0.000
    </span>
  </div>
  <div class="pair">
    <div>
      <div class="label">Frame {fi}</div>
      <img src="data:image/jpeg;base64,{b64_ann_i}">
    </div>
    <div>
      <div class="label">Frame {fj}</div>
      <img src="data:image/jpeg;base64,{b64_ann_j}">
    </div>
  </div>
  <div class="corr">
    <div class="label">Correspondences ({len(uv_i)} matches found)</div>
    <img src="data:image/jpeg;base64,{b64_corr}">
  </div>
</div>
"""
            continue

        mask = (scores >= lo) & (scores < hi)
        bin_idx = np.where(mask)[0]
        if len(bin_idx) == 0:
            cards_html += f"<h2>{bin_name}: no pairs</h2>\n"
            continue

        n = min(samples_per_bin, len(bin_idx))
        chosen = rng.choice(bin_idx, size=n, replace=False)

        cards_html += f"<h2>{bin_name} ({mask.sum():,} pairs total)</h2>\n"

        for idx in chosen:
            row = pairs.table.slice(int(idx), 1)
            scene_id = row.column("scene_id")[0].as_py()
            fi = row.column("frame_i")[0].as_py()
            fj = row.column("frame_j")[0].as_py()
            score = row.column("overlap_score")[0].as_py()

            if scene_id not in scene_cache:
                scene_cache[scene_id] = _get_scene(dataset, data_dir, scene_id)
            scene = scene_cache[scene_id]

            log.info("Rendering %s frame %d <-> %d (overlap=%.3f)", scene_id, fi, fj, score)

            img_i = scene.rgb(fi)
            img_j = scene.rgb(fj)

            uv_i, uv_j_proj, uv_j, uv_i_proj = _compute_overlap_mask(scene, fi, fj, num_samples=5000)

            ann_i = _annotate_image(img_i, uv_i, color_i)
            ann_j = _annotate_image(img_j, uv_j, color_j)
            corr = _draw_correspondence_lines(img_i, img_j, uv_i, uv_j_proj)

            b64_ann_i = _img_to_base64(ann_i)
            b64_ann_j = _img_to_base64(ann_j)
            b64_corr = _img_to_base64(corr, quality=70)

            cards_html += f"""
<div class="card">
  <div class="header">
    <span class="scene">{scene_id}</span>
    frame {fi} &harr; frame {fj}
    <span class="score" style="background: hsl({int(score*120)}, 70%, 45%)">
      overlap: {score:.3f}
    </span>
  </div>
  <div class="pair">
    <div>
      <div class="label">Frame {fi} <span style="color:#00c800">&#9632;</span> visible pixels</div>
      <img src="data:image/jpeg;base64,{b64_ann_i}">
    </div>
    <div>
      <div class="label">Frame {fj} <span style="color:#c86400">&#9632;</span> visible pixels</div>
      <img src="data:image/jpeg;base64,{b64_ann_j}">
    </div>
  </div>
  <div class="corr">
    <div class="label">Correspondences (50 random matches)</div>
    <img src="data:image/jpeg;base64,{b64_corr}">
  </div>
</div>
"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VPR Covisibility Pairs &mdash; {dataset}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee;
         max-width: 1400px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #e94560; }}
  h2 {{ color: #e0e0e0; background: #16213e; padding: 10px 16px; border-radius: 8px;
        border-left: 4px solid #e94560; }}
  .card {{ background: #16213e; border-radius: 12px; margin: 20px 0; padding: 16px;
           box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
  .header {{ font-size: 14px; margin-bottom: 10px; display: flex; align-items: center; gap: 12px; }}
  .scene {{ background: #0f3460; padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
  .score {{ padding: 2px 8px; border-radius: 4px; color: white; font-weight: bold; }}
  .pair {{ display: flex; gap: 8px; }}
  .pair > div {{ flex: 1; }}
  .pair img, .corr img {{ width: 100%; border-radius: 6px; }}
  .corr {{ margin-top: 10px; }}
  .label {{ font-size: 12px; color: #aaa; margin-bottom: 4px; }}
  .stats {{ background: #0f3460; padding: 12px 16px; border-radius: 8px; margin: 10px 0; }}
</style>
</head>
<body>
<h1>VPR Covisibility Pairs</h1>
<div class="stats">
  Dataset: <b>{dataset}</b> |
  Pairs file: <b>{pairs_path.name}</b> |
  Total pairs: <b>{len(pairs):,}</b>
</div>
{cards_html}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize covisibility pairs")
    parser.add_argument("--dataset", required=True, choices=["replica"])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path, help="Path to .parquet")
    parser.add_argument("--output", type=Path, default=None, help="Save HTML to file instead of serving")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--samples-per-bin", type=int, default=3)
    args = parser.parse_args()

    html = generate_html(args.dataset, args.data_dir, args.pairs, args.samples_per_bin)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html)
        log.info("HTML saved to %s (%d bytes)", args.output, len(html))
        return

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        def log_message(self, fmt, *a):
            pass

    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        log.info("Serving visualization at http://localhost:%d", args.port)
        log.info("Press Ctrl+C to stop.")
        httpd.serve_forever()


if __name__ == "__main__":
    main()

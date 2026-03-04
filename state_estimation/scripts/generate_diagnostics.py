#!/usr/bin/env python3
"""Generate diagnostic image grids to verify camera-pose alignment."""

import csv
import os

import numpy as np
from PIL import Image, ImageDraw
from evo.core import sync
from evo.tools import file_interface


def main():
    traj_est = file_interface.read_tum_trajectory_file("/results/aligned_trajectory.tum")
    traj_gt = file_interface.read_euroc_csv_trajectory(
        "/data/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    )
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)

    cam_entries = []
    with open("/data/euroc/MH_01_easy/mav0/cam0/data.csv") as f:
        for row in csv.reader(f):
            if row[0].startswith("#"):
                continue
            cam_entries.append(
                (int(row[0]) / 1e9, "/data/euroc/MH_01_easy/mav0/cam0/data/" + row[1].strip())
            )
    cam_entries.sort()
    cam_ts = np.array([e[0] for e in cam_entries])
    cam_files = [e[1] for e in cam_entries]

    N = len(traj_gt.timestamps)
    gt_xyz = traj_gt.positions_xyz
    timestamps = traj_est.timestamps
    t0 = timestamps[0]

    # 10 frames spaced 5 apart = 2.5s window per segment
    segments = {
        "A_start": list(range(0, 50, 5)),
        "B_one_third": list(range(N // 3, N // 3 + 50, 5)),
        "C_two_thirds": list(range(2 * N // 3, 2 * N // 3 + 50, 5)),
    }

    for seg_name, indices in segments.items():
        frames = []
        seg_gt = []
        for i in indices:
            t = timestamps[i]
            idx = np.searchsorted(cam_ts, t)
            best_idx = idx
            best_dt = abs(cam_ts[idx] - t) if idx < len(cam_ts) else 999
            if idx > 0 and abs(cam_ts[idx - 1] - t) < best_dt:
                best_idx = idx - 1

            img = Image.open(cam_files[best_idx]).convert("RGB")
            img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
            draw = ImageDraw.Draw(img)

            gt_p = gt_xyz[i]
            rel_t = t - t0
            seg_gt.append(gt_p)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    draw.text((10 + dx, 8 + dy), f"frame {i}  t={rel_t:.2f}s", fill=(0, 0, 0))
                    draw.text(
                        (10 + dx, 36 + dy),
                        f"GT x={gt_p[0]:.2f} y={gt_p[1]:.2f} z={gt_p[2]:.2f}",
                        fill=(0, 0, 0),
                    )
            draw.text((10, 8), f"frame {i}  t={rel_t:.2f}s", fill=(0, 255, 0))
            draw.text(
                (10, 36), f"GT x={gt_p[0]:.2f} y={gt_p[1]:.2f} z={gt_p[2]:.2f}", fill=(0, 255, 0)
            )
            frames.append(img)

        w, h = frames[0].size

        # Trajectory plot dimensions
        plot_h = 300
        grid = Image.new("RGB", (w * 5, h * 2 + plot_h), (20, 20, 20))
        for j, img in enumerate(frames):
            grid.paste(img, ((j % 5) * w, (j // 5) * h))

        # Draw XY trajectory plot
        draw = ImageDraw.Draw(grid)
        plot_top = h * 2 + 30
        plot_left = 60
        pw = w * 5 - 120
        ph = plot_h - 50

        all_x, all_y = gt_xyz[:, 0], gt_xyz[:, 1]
        xmin, xmax = all_x.min(), all_x.max()
        ymin, ymax = all_y.min(), all_y.max()
        xr = (xmax - xmin) or 1
        yr = (ymax - ymin) or 1
        scale = min(pw / xr, ph / yr) * 0.85
        cx = plot_left + pw / 2
        cy = plot_top + ph / 2
        mx = (xmin + xmax) / 2
        my = (ymin + ymax) / 2

        def to_px(pos):
            return (int(cx + (pos[0] - mx) * scale), int(cy + (pos[1] - my) * scale))

        # Full trajectory in dark gray
        for k in range(0, len(gt_xyz) - 1, 2):
            p1 = to_px(gt_xyz[k])
            p2 = to_px(gt_xyz[k + 1])
            draw.line([p1, p2], fill=(50, 50, 50), width=1)

        # Segment in green with numbered markers
        for k in range(len(seg_gt) - 1):
            p1 = to_px(seg_gt[k])
            p2 = to_px(seg_gt[k + 1])
            draw.line([p1, p2], fill=(0, 255, 0), width=3)

        for k in range(len(seg_gt)):
            px = to_px(seg_gt[k])
            r = 7
            color = (255, 255, 0) if k == 0 else ((255, 0, 0) if k == 9 else (0, 200, 0))
            draw.ellipse([px[0] - r, px[1] - r, px[0] + r, px[1] + r], fill=color)
            draw.text((px[0] + 10, px[1] - 8), str(k), fill=(255, 255, 255))

        seg_t0 = timestamps[indices[0]] - t0
        seg_t1 = timestamps[indices[-1]] - t0
        dx = seg_gt[-1][0] - seg_gt[0][0]
        dy = seg_gt[-1][1] - seg_gt[0][1]
        dz = seg_gt[-1][2] - seg_gt[0][2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        draw.text(
            (plot_left, plot_top - 25),
            f"{seg_name}: t={seg_t0:.1f}s-{seg_t1:.1f}s | XY plot (green=segment, yellow=start, red=end)",
            fill=(255, 255, 255),
        )
        draw.text(
            (plot_left, plot_top + ph + 5),
            f"Movement: dx={dx:.3f}m dy={dy:.3f}m dz={dz:.3f}m |total|={dist:.3f}m",
            fill=(200, 200, 200),
        )

        out = f"/results/diag_{seg_name}.png"
        grid.save(out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()

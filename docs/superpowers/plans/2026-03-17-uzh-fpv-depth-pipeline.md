# UZH-FPV Depth Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download UZH-FPV stereo drone racing data, set up FoundationStereo, undistort+rectify fisheye stereo pairs, and generate dense metric depth maps for VPR covisibility processing.

**Architecture:** UZH-FPV Snapdragon stereo (640×480 fisheye, 186° FOV) → undistort with equidistant model → rectify stereo pairs → FoundationStereo disparity → metric depth via `depth = fx * baseline / disparity`. FoundationStereo runs in its own conda env; all scripts live under `perception/training/`.

**Tech Stack:** FoundationStereo (ViT-Large, PyTorch), OpenCV stereo rectification, conda, RTX 3090

---

## Key Research Findings

### UZH-FPV Dataset
- **Stereo source:** Snapdragon Flight board — two OV7251 global shutter cameras, 640×480, 186° FOV fisheye
- **Distortion model:** Equidistant (fisheye) — Kalibr convention. **Must undistort + rectify before FoundationStereo.**
- **Calibration:** Kalibr YAML files with per-camera intrinsics, distortion coefficients, and `T_cam_imu` transforms
- **GT poses:** `groundtruth.txt` — `[idx, timestamp, tx, ty, tz, qx, qy, qz, qw]` (body frame in world)
- **Download format:** Rosbag (`.bag`) contains both left+right images. ZIP text format may only have left camera.
- **Sequences with Snapdragon stereo + GT:** indoor_forward (3,5,6,7,9,10), indoor_45 (2,4,9,12,13,14), outdoor_forward (1,3,5), outdoor_45 (1) — 16 total
- **Size:** ~2-3GB per sequence in rosbag format. Start with 3-4 sequences to validate.

### FoundationStereo
- **Checkpoints:** Google Drive, ViT-Large (`23-51-11/model_best_bp2.pth`) ~1-2GB
- **Input:** Rectified, undistorted stereo pairs (any resolution, padded to div-by-32)
- **Output:** Disparity map (float32 pixels) → `depth = fx_rectified * baseline / disparity`
- **Env:** conda with torch 2.4.1, xformers, flash-attn
- **Performance:** RTX 3090 PyTorch ~49ms/frame, TensorRT ~8ms/frame

---

## File Structure

```
~/corvidx/
├── model_checkpoints/
│   └── foundation_stereo/
│       └── 23-51-11/
│           ├── model_best_bp2.pth
│           └── cfg.yaml
├── repos/
│   └── FoundationStereo/          # cloned repo (conda env lives here)
└── data/
    └── uzh-fpv/
        ├── raw/                    # downloaded rosbags
        │   ├── indoor_forward_3_snapdragon_with_gt.bag
        │   └── ...
        ├── extracted/              # extracted images + GT
        │   └── indoor_forward_3_snapdragon/
        │       ├── left/           # left camera PNGs
        │       ├── right/          # right camera PNGs
        │       ├── groundtruth.txt
        │       └── calibration.yaml
        ├── rectified/              # undistorted + rectified stereo pairs
        │   └── indoor_forward_3_snapdragon/
        │       ├── left/
        │       ├── right/
        │       └── rectification_params.json
        └── depth/                  # FoundationStereo output
            └── indoor_forward_3_snapdragon/
                ├── depth_000000.npy
                └── ...

algo_src/perception/training/
├── data/
│   └── uzh_fpv/
│       ├── __init__.py
│       ├── constants.py            # sequence list, camera topics, download URLs
│       ├── extract_rosbag.py       # extract stereo images + GT from rosbag
│       ├── rectify_stereo.py       # undistort fisheye + stereo rectification
│       └── loader.py               # UZHFPVScene for covisibility pipeline
├── run_foundation_stereo.py        # CLI: batch FoundationStereo on rectified pairs
└── tests/
    └── test_uzh_fpv_rectify.py
```

---

## Task 1: Set up model checkpoint directory + download FoundationStereo

**Files:**
- Create: `~/corvidx/model_checkpoints/foundation_stereo/`
- Create: `~/corvidx/repos/FoundationStereo/` (git clone)

- [ ] **1.1 Create organized checkpoint directory**
  ```bash
  mkdir -p ~/corvidx/model_checkpoints/foundation_stereo
  ```

- [ ] **1.2 Clone FoundationStereo repo**
  ```bash
  mkdir -p ~/corvidx/repos
  cd ~/corvidx/repos
  git clone https://github.com/NVlabs/FoundationStereo.git
  ```

- [ ] **1.3 Download ViT-Large checkpoint**
  Download from Google Drive folder: https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf
  - Download `23-51-11/model_best_bp2.pth` and `23-51-11/cfg.yaml`
  - Save to `~/corvidx/model_checkpoints/foundation_stereo/23-51-11/`
  ```bash
  # Use gdown to download from Google Drive
  pip install gdown
  gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf -O ~/corvidx/model_checkpoints/foundation_stereo/
  ```

- [ ] **1.4 Set up FoundationStereo conda env**
  ```bash
  cd ~/corvidx/repos/FoundationStereo
  conda env create -f environment.yml
  conda run -n foundation_stereo pip install flash-attn
  ```

- [ ] **1.5 Verify FoundationStereo runs on sample images**
  ```bash
  conda activate foundation_stereo
  cd ~/corvidx/repos/FoundationStereo
  python scripts/run_demo.py \
      --left_file ./assets/left.png \
      --right_file ./assets/right.png \
      --ckpt_dir ~/corvidx/model_checkpoints/foundation_stereo/23-51-11/model_best_bp2.pth \
      --out_dir /tmp/fs_test/
  ```
  Expected: `vis.png` and disparity output in `/tmp/fs_test/`

---

## Task 2: Download UZH-FPV sequences (start with 3 for validation)

**Files:**
- Create: `perception/training/data/uzh_fpv/__init__.py`
- Create: `perception/training/data/uzh_fpv/constants.py`

Start with 3 diverse sequences that have Snapdragon stereo + GT:
- `indoor_forward_3` — indoor, forward-facing
- `indoor_45_2` — indoor, 45° downward
- `outdoor_forward_3` — outdoor, forward-facing

- [ ] **2.1 Create UZH-FPV constants**
  ```python
  # perception/training/data/uzh_fpv/constants.py
  ROSBAG_BASE_URL = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3"
  CALIB_BASE_URL = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib"

  # Sequences with Snapdragon stereo + ground truth
  SEQUENCES_WITH_GT = [
      "indoor_forward_3", "indoor_forward_5", "indoor_forward_6",
      "indoor_forward_7", "indoor_forward_9", "indoor_forward_10",
      "indoor_45_2", "indoor_45_4", "indoor_45_9",
      "indoor_45_12", "indoor_45_13", "indoor_45_14",
      "outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5",
      "outdoor_45_1",
  ]

  # Start with these 3 for validation
  VALIDATION_SEQUENCES = [
      "indoor_forward_3",
      "indoor_45_2",
      "outdoor_forward_3",
  ]

  # Snapdragon stereo camera ROS topics
  LEFT_IMAGE_TOPIC = "/snappy_cam/stereo_l"
  RIGHT_IMAGE_TOPIC = "/snappy_cam/stereo_r"
  IMU_TOPIC = "/snappy_imu"
  ```

- [ ] **2.2 Download validation rosbags**
  ```bash
  mkdir -p ~/corvidx/data/uzh-fpv/raw
  cd ~/corvidx/data/uzh-fpv/raw
  for seq in indoor_forward_3 indoor_45_2 outdoor_forward_3; do
      wget -c "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/${seq}_snapdragon_with_gt.bag"
  done
  ```

- [ ] **2.3 Download calibration files**
  ```bash
  mkdir -p ~/corvidx/data/uzh-fpv/calib
  cd ~/corvidx/data/uzh-fpv/calib
  # Download Snapdragon calibration for each environment
  for env in indoor_forward indoor_45 outdoor_forward outdoor_45; do
      wget -c "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/camchain-imucam-..${env}_calib_snapdragon_imu.yaml" \
          -O "${env}_snapdragon.yaml" || true
  done
  ```

---

## Task 3: Extract stereo images + GT from rosbags

**Files:**
- Create: `perception/training/data/uzh_fpv/extract_rosbag.py`

- [ ] **3.1 Install rosbag reader (no ROS needed)**
  ```bash
  pip install rosbags  # pure-Python rosbag reader, no ROS dependency
  ```

- [ ] **3.2 Write rosbag extraction script**
  `perception/training/data/uzh_fpv/extract_rosbag.py`:
  - Read `.bag` file using `rosbags` library
  - Extract left + right stereo images from Snapdragon topics
  - Extract ground truth poses from GT topic or the embedded `groundtruth.txt`
  - Copy calibration YAML to output directory
  - Save images as numbered PNGs: `left/000000.png`, `right/000000.png`
  - Save timestamps for frame-to-GT alignment
  - CLI: `python -m perception.training.data.uzh_fpv.extract_rosbag --bag PATH --output-dir DIR`

- [ ] **3.3 Extract validation sequences**
  ```bash
  for seq in indoor_forward_3 indoor_45_2 outdoor_forward_3; do
      python -m perception.training.data.uzh_fpv.extract_rosbag \
          --bag ~/corvidx/data/uzh-fpv/raw/${seq}_snapdragon_with_gt.bag \
          --output-dir ~/corvidx/data/uzh-fpv/extracted/${seq}_snapdragon/
  done
  ```

- [ ] **3.4 Verify extraction**
  Check: left/ and right/ dirs have same frame count, images are 640×480 grayscale, groundtruth.txt exists.

---

## Task 4: Stereo rectification (undistort fisheye + rectify)

**Files:**
- Create: `perception/training/data/uzh_fpv/rectify_stereo.py`
- Create: `perception/training/tests/test_uzh_fpv_rectify.py`

FoundationStereo requires rectified, undistorted input. UZH-FPV Snapdragon has 186° fisheye with equidistant distortion model.

- [ ] **4.1 Write the test for rectification**
  ```python
  # test: given known calibration, rectified images have horizontal epipolar lines
  def test_rectification_produces_horizontal_epilines():
      # create synthetic stereo pair with known calibration
      # rectify
      # verify: corresponding points have same y-coordinate
  ```

- [ ] **4.2 Write rectification script**
  `perception/training/data/uzh_fpv/rectify_stereo.py`:
  - Load Kalibr YAML calibration
  - Parse equidistant distortion model (k1, k2, k3, k4) and intrinsics (fx, fy, cx, cy) for both cameras
  - Compute stereo baseline from `T_cam_imu` transforms: `T_left_right = T_left_imu @ inv(T_right_imu)`
  - Use `cv2.fisheye.stereoRectify()` to compute rectification maps
  - Use `cv2.fisheye.initUndistortRectifyMap()` for both cameras
  - Apply `cv2.remap()` to undistort + rectify each frame
  - Save rectified images to output directory
  - Save `rectification_params.json`: rectified K, baseline, image size
  - CLI: `python -m perception.training.data.uzh_fpv.rectify_stereo --input-dir DIR --calib YAML --output-dir DIR`

- [ ] **4.3 Run rectification on extracted sequences**
  ```bash
  for seq in indoor_forward_3 indoor_45_2 outdoor_forward_3; do
      env=$(echo $seq | sed 's/_[0-9]*$//')
      python -m perception.training.data.uzh_fpv.rectify_stereo \
          --input-dir ~/corvidx/data/uzh-fpv/extracted/${seq}_snapdragon/ \
          --calib ~/corvidx/data/uzh-fpv/calib/${env}_snapdragon.yaml \
          --output-dir ~/corvidx/data/uzh-fpv/rectified/${seq}_snapdragon/
  done
  ```

- [ ] **4.4 Visually verify rectification**
  Draw horizontal lines across a rectified stereo pair — corresponding features should sit on the same scanline.

---

## Task 5: Run FoundationStereo → dense depth maps

**Files:**
- Create: `perception/training/run_foundation_stereo.py`

- [ ] **5.1 Write FoundationStereo batch runner**
  `perception/training/run_foundation_stereo.py`:
  - Load FoundationStereo model from checkpoint
  - Iterate over rectified stereo pairs in a directory
  - Run inference: left + right → disparity
  - Convert disparity to metric depth: `depth = fx_rect * baseline / disparity`
  - Save depth maps as `.npy` files (float32, meters)
  - Save a few visualization PNGs (colorized depth) for sanity checking
  - CLI: `python -m perception.training.run_foundation_stereo --input-dir DIR --ckpt PATH --output-dir DIR`
  - Args: `--scale` (default 1.0), `--valid-iters` (default 32), `--batch-size` (default 1)
  - Progress bar with ETA

  **Note:** This script must be run inside the `foundation_stereo` conda env, not the Docker container.

- [ ] **5.2 Run on validation sequences**
  ```bash
  conda activate foundation_stereo
  for seq in indoor_forward_3 indoor_45_2 outdoor_forward_3; do
      python -m perception.training.run_foundation_stereo \
          --input-dir ~/corvidx/data/uzh-fpv/rectified/${seq}_snapdragon/ \
          --ckpt ~/corvidx/model_checkpoints/foundation_stereo/23-51-11/model_best_bp2.pth \
          --output-dir ~/corvidx/data/uzh-fpv/depth/${seq}_snapdragon/ \
          --scale 1.0
  done
  ```
  At ~49ms/frame on RTX 3090, each 30Hz sequence of ~60s = ~1800 frames → ~90 seconds per sequence.

- [ ] **5.3 Verify depth maps**
  - Check depth range makes sense (indoor: 0.5-10m, outdoor: 1-50m)
  - Visually inspect colorized depth vs. original images
  - Check for artifacts or invalid regions

- [ ] **5.4 Commit**
  ```bash
  git add perception/training/data/uzh_fpv/ perception/training/run_foundation_stereo.py
  git commit -m "feat: UZH-FPV stereo extraction, rectification, and FoundationStereo depth pipeline"
  ```

---

## Task 6: Wire UZH-FPV into covisibility pipeline

**Files:**
- Create: `perception/training/data/uzh_fpv/loader.py`
- Modify: `perception/training/process_dataset.py` — add `uzh-fpv` to dataset choices

- [ ] **6.1 Write UZHFPVScene loader**
  Same interface as `ReplicaScene`: `n_frames`, `intrinsics()`, `poses()`, `rgb(idx)`, `depth(idx)`, `depths()`
  - Load rectified left images as "RGB" (grayscale → 3-channel)
  - Load depth from `.npy` files
  - Load GT poses from `groundtruth.txt`, interpolate to match frame timestamps
  - Intrinsics from `rectification_params.json` (rectified K)

- [ ] **6.2 Add uzh-fpv to process_dataset.py**
  - Add `uzh-fpv` to `--dataset` choices
  - Wire up loader in `_get_loader()` and `_get_scene_list()`

- [ ] **6.3 Test E2E: covisibility on one UZH-FPV sequence**
  ```bash
  python -m perception.training.process_dataset \
      --dataset uzh-fpv \
      --data-dir ~/corvidx/data/uzh-fpv \
      --scenes indoor_forward_3_snapdragon \
      --num-samples 5000
  ```

- [ ] **6.4 Commit**

---

## Memory / Disk Budget

| Item | Size |
|---|---|
| FoundationStereo checkpoint (ViT-Large) | ~1-2 GB |
| FoundationStereo repo + conda env | ~10 GB |
| 3 UZH-FPV rosbags | ~6-9 GB |
| Extracted stereo images (3 seqs) | ~3 GB |
| Rectified images (3 seqs) | ~3 GB |
| Depth maps (3 seqs) | ~2 GB |
| **Total** | **~25-30 GB** |

Available: 491 GB. Plenty of room.

## Important Notes

- UZH-FPV uses **fisheye lenses** (186° FOV, equidistant distortion). FoundationStereo expects rectified input. The rectification step is critical.
- FoundationStereo runs in its **own conda env** (torch 2.4.1, xformers, flash-attn) — separate from the main pipeline.
- Snapdragon images are **grayscale** (OV7251). FoundationStereo handles monochrome input.
- GT poses are in **body (IMU) frame** — need `T_cam_imu` from calibration to get camera-frame poses for covisibility.
- Rosbag extraction requires `rosbags` Python package (pure Python, no ROS dependency).

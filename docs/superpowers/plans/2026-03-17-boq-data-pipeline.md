# BoQ VPR Data Pipeline — Implementation Plan (v2)

**Goal:** Build a multi-dataset VPR training pipeline: download datasets, compute pairwise covisibility/pose-distance matrices, store as Parquet + scene data on R2, build a PyTorch dataloader with binned sampling for BoQ VPR training.

**Infrastructure:**
- **Data processing + depth estimation:** Desktop `martian` (24 threads, 32GB RAM, RTX 3090 24GB, 1TB NVMe + 1.8TB HDD)
- **SSH access:** `ssh martian` from MacBook (jsmoothie@martian.lan)
- **GPU training (later):** RunPod 2x V100-32GB ($0.38/hr total)
- **Target inference:** RTX 2060 Super (competition constraint)

**All code lives under `perception/training/`** — treated as its own sub-repo.

---

## Datasets

| Dataset | Size | Depth Source | Pair Strategy | Purpose |
|---------|------|-------------|---------------|---------|
| **Replica** (18 scenes) | ~5GB | Sensor GT | Covisibility | Pipeline validation + indoor VPR |
| **TartanAir** (5-10 envs) | ~50-150GB | Render GT | Covisibility | Diverse indoor/outdoor, aggressive drone-like motion |
| **EuRoC** (11 sequences) | ~3GB | FoundationStereo (3090) | Covisibility | Real MAV data |
| **UZH-FPV** (27 sequences) | ~3GB | FoundationStereo (3090) | Covisibility | Real FPV drone racing data |

Total: ~60-160GB on NVMe (524GB free).

---

## Phase 1: Data Acquisition & Exploration (on desktop)

Download all datasets and examine actual formats before writing processing code.

- [ ] **1.1 Download Replica**
  ```bash
  mkdir -p ~/data && cd ~/data
  wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
  unzip Replica.zip
  ```

- [ ] **1.2 Download TartanAir (subset)**
  ```bash
  # Install tartanair tools
  pip install tartanair
  # Download 5-10 environments, RGB + depth only, left camera, easy difficulty
  python -m tartanair.download --env abandonedfactory hospital office soulcity --modality rgb depth --camera left --difficulty easy --output ~/data/tartanair
  ```

- [ ] **1.3 Download EuRoC**
  ```bash
  # Download ASL format (contains stereo images + IMU + GT poses)
  cd ~/data && mkdir euroc && cd euroc
  # Download MH and V sequences
  for seq in MH_01_easy MH_02_easy MH_03_medium V1_01_easy V1_02_medium V2_01_easy; do
    wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/${seq}/${seq}.zip
    unzip ${seq}.zip
  done
  ```

- [ ] **1.4 Download UZH-FPV**
  ```bash
  cd ~/data && mkdir uzh-fpv && cd uzh-fpv
  # Download select sequences (indoor/outdoor with stereo)
  # Check https://fpv.ifi.uzh.ch/datasets/ for URLs
  ```

- [ ] **1.5 Run FoundationStereo on EuRoC + UZH-FPV**
  ```bash
  # Clone FoundationStereo
  cd ~/repos && git clone https://github.com/NVlabs/FoundationStereo.git
  cd FoundationStereo && pip install -r requirements.txt
  # Run on EuRoC stereo pairs → dense depth maps
  # Run on UZH-FPV stereo pairs → dense depth maps
  # Output: depth/ directory per sequence matching frame numbering
  ```
  RTX 3090 performance: ~21ms/frame with TensorRT, ~49ms/frame PyTorch.
  EuRoC grayscale stereo is supported (tested on monochrome images).

- [ ] **1.6 Examine data formats & document findings**
  For each dataset, verify and record:
  - Scene/sequence names
  - Frame count per scene
  - Image resolution (RGB and depth)
  - Depth format (dtype, scale factor, units)
  - Pose format (4x4 c2w? convention?)
  - Intrinsics (per-scene or global?)
  - Any coordinate system gotchas

  **These findings are CRITICAL — all Phase 2 code depends on them.**

---

## Phase 2: Write Processing Code

Write all code under `perception/training/` informed by Phase 1 findings.

### File structure

```
perception/training/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── replica/
│   │   ├── __init__.py
│   │   ├── download.py          # wget + unzip
│   │   ├── loader.py            # ReplicaScene: frames, depths, poses, intrinsics
│   │   └── constants.py         # Intrinsics, depth scale, scene list
│   ├── tartanair/
│   │   ├── __init__.py
│   │   ├── download.py          # tartanair API download
│   │   └── loader.py            # TartanAirScene: RGB, depth, poses
│   ├── euroc/
│   │   ├── __init__.py
│   │   └── loader.py            # EuRoCScene: stereo → FoundationStereo depth, poses
│   ├── uzh_fpv/
│   │   ├── __init__.py
│   │   └── loader.py            # UZHFPVScene: stereo → FoundationStereo depth, poses
│   ├── covisibility.py          # Vectorized N×N covisibility matrix
│   ├── pairs.py                 # Parquet pair table + binned sampling
│   ├── upload.py                # Upload to R2
│   └── dataloader.py            # PyTorch covisibility-binned sampling
├── process_dataset.py           # CLI entrypoint: load → covisibility → parquet → upload
├── run_foundation_stereo.py     # CLI: run FoundationStereo on stereo datasets
├── Dockerfile                   # GPU image (CUDA + python3.11)
└── tests/
    ├── __init__.py
    ├── test_covisibility.py
    ├── test_pairs.py
    └── test_dataloader.py
```

- [ ] **2.1 Package scaffold + constants**
  - Create `__init__.py` stubs for all packages
  - Constants per dataset (from Phase 1 findings)
  - Add `pyarrow>=14.0` to perception deps in `pyproject.toml`

- [ ] **2.2 Dataset loaders** (one per dataset)
  - Common interface: `Scene` with `frames()`, `depths()`, `poses()`, `intrinsics()`
  - **Replica**: parse `traj.txt` → `(N, 4, 4)` poses, lazy load RGB/depth
  - **TartanAir**: parse pose files, load RGB/depth from env directories
  - **EuRoC**: load pre-computed FoundationStereo depth, parse GT poses from CSV
  - **UZH-FPV**: load pre-computed FoundationStereo depth, parse GT poses

- [ ] **2.3 Download scripts** (Replica + TartanAir)
  - `wget -c` + `unzip` for Replica, skip if exists
  - TartanAir API wrapper for selective env download

- [ ] **2.4 FoundationStereo runner** (`run_foundation_stereo.py`)
  - CLI: `--dataset euroc|uzh-fpv --data-dir DIR --output-dir DIR`
  - Loads stereo pairs, runs FoundationStereo, saves depth PNGs
  - Converts disparity → metric depth using calibration (baseline × fx / disparity)
  - Batch processing with progress bar

- [ ] **2.5 Covisibility computation** (`covisibility.py`)
  - `unproject_pixels(depth, K, num_samples)` → subsample S random valid pixels → 3D
  - `compute_pairwise_overlap(depth_i, T_w_ci, depth_j, T_w_cj, K)` → overlap score
  - `compute_scene_covisibility(depths, poses, K)` → full N×N matrix (symmetric)
  - All NumPy-vectorized, memory-conscious (process one scene at a time, 32GB limit)

- [ ] **2.6 Parquet pair table** (`pairs.py`)
  - `PairTable`: schema `(dataset, scene_id, frame_i, frame_j, overlap_score)`
  - `from_overlap_matrix()` → upper triangle, filter zero pairs
  - `save()`/`load()` Parquet, `merge()` across scenes and datasets
  - `sample_uniform(bins, n)` → uniform sampling across covisibility bins with quota redistribution

- [ ] **2.7 R2 upload** (`upload.py`)
  - `upload_scene()` — rgb/*.jpg, depth/*.png, poses.npy, intrinsics.json
  - `upload_pairs()` — *.parquet files
  - Uses existing `artifacts.r2` client

- [ ] **2.8 E2E processing script** (`process_dataset.py`)
  - CLI: `--dataset replica|tartanair|euroc|uzh-fpv --data-dir DIR --num-samples N --scenes SCENES --upload`
  - Orchestrates: load scenes → covisibility → Parquet → upload R2
  - Writes `metadata.json` per dataset

- [ ] **2.9 Dataloader** (`dataloader.py`)
  - `CovisibilityVPRDataset(Dataset)`: loads pair tables from multiple datasets
  - Default bins: hard (0.05-0.3), medium (0.3-0.7), easy (0.7-1.0)
  - Returns `{img_i, img_j, overlap_score, dataset, scene_id, frame_i, frame_j}`
  - `reshuffle()` for new epoch sampling
  - Supports mixed-dataset batches

- [ ] **2.10 Dockerfile** (`perception/training/Dockerfile`)
  ```dockerfile
  FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
  RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3-pip wget unzip libgl1-mesa-glx libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*
  WORKDIR /app
  COPY . .
  RUN pip install --no-cache-dir numpy scipy opencv-python-headless \
      "pyarrow>=14.0" boto3 torch torchvision && pip install --no-cache-dir -e .
  CMD ["bash"]
  ```

- [ ] **2.11 Commit all code**

---

## Phase 3: Run Pipeline (on desktop `martian`)

- [ ] **3.1 Run FoundationStereo on EuRoC + UZH-FPV**
  ```bash
  python -m perception.training.run_foundation_stereo \
      --dataset euroc --data-dir ~/data/euroc --output-dir ~/data/euroc-depth
  python -m perception.training.run_foundation_stereo \
      --dataset uzh-fpv --data-dir ~/data/uzh-fpv --output-dir ~/data/uzh-fpv-depth
  ```

- [ ] **3.2 Process Replica (test one scene first)**
  ```bash
  python -m perception.training.process_dataset \
      --dataset replica --data-dir ~/data/Replica --scenes room0 --num-samples 5000
  ```
  Verify: Parquet exists, overlap distribution looks reasonable.

- [ ] **3.3 Process all Replica scenes**
  ```bash
  python -m perception.training.process_dataset \
      --dataset replica --data-dir ~/data/Replica --num-samples 10000 --upload
  ```

- [ ] **3.4 Process TartanAir subset**
  ```bash
  python -m perception.training.process_dataset \
      --dataset tartanair --data-dir ~/data/tartanair --num-samples 10000 --upload
  ```

- [ ] **3.5 Process EuRoC**
  ```bash
  python -m perception.training.process_dataset \
      --dataset euroc --data-dir ~/data/euroc --depth-dir ~/data/euroc-depth \
      --num-samples 10000 --upload
  ```

- [ ] **3.6 Process UZH-FPV**
  ```bash
  python -m perception.training.process_dataset \
      --dataset uzh-fpv --data-dir ~/data/uzh-fpv --depth-dir ~/data/uzh-fpv-depth \
      --num-samples 10000 --upload
  ```

- [ ] **3.7 Verify R2 uploads**
  Check all datasets present on R2 under `datasets/vpr/`.

---

## Phase 4: Verify Dataloader

- [ ] **4.1 Download one scene per dataset from R2**

- [ ] **4.2 Smoke test dataloader**
  - Verify: images load from all 4 datasets
  - Bins are populated across datasets
  - Overlap score distributions look correct
  - Mixed-dataset batching works

---

## R2 Storage Layout

```
datasets/vpr/
├── metadata.json                  # {datasets, total_pairs, processing_date}
├── replica/
│   ├── scenes/
│   │   ├── room0/
│   │   │   ├── rgb/frame000000.jpg, ...
│   │   │   ├── depth/depth000000.png, ...
│   │   │   ├── poses.npy          # (N, 4, 4) float64 c2w
│   │   │   └── intrinsics.json    # {fx, fy, cx, cy, w, h}
│   │   └── ...
│   └── pairs/
│       ├── room0.parquet
│       └── ...
├── tartanair/
│   ├── scenes/{env_name}/...
│   └── pairs/{env_name}.parquet
├── euroc/
│   ├── scenes/{seq_name}/...
│   └── pairs/{seq_name}.parquet
└── uzh-fpv/
    ├── scenes/{seq_name}/...
    └── pairs/{seq_name}.parquet
```

## Covisibility Algorithm

```
1. Subsample S random pixels from frame i where depth > 0
2. Unproject to 3D:  pts_cam = depth * K_inv @ [u, v, 1]^T
3. To world frame:   pts_world = T_w_ci @ pts_cam
4. To frame j cam:   pts_cam_j = inv(T_w_cj) @ pts_world
5. Project to j:     uv_j = K @ pts_cam_j / z
6. Check valid:      in_bounds & z > 0
7. Depth check:      |z_reprojected - depth_j[v', u']| < threshold
8. Score:            overlap = count(valid) / S
```

Vectorization: pre-compute all per-frame point clouds, then batch matrix ops across pairs. ~10K pixel subsample per frame for ~30x speedup over full resolution.

Memory budget: process one scene at a time to stay within 32GB RAM. For Replica N≈2000 frames, peak ~4GB per scene.

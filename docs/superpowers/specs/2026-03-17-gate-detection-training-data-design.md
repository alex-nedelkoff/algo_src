# Gate Detection Training Data Pipeline

**Date**: 2026-03-17
**Status**: Draft
**Author**: Claude + Janahan
**Related**: COR-59 (Perception Architecture), COR-60 (BoQ VPR Data Pipeline)

## Problem

COR-59 specifies a DA3 multi-head perception backbone (frozen DINOv2-ViT-S + DPT decoders) with heads for gate segmentation, corner heatmaps, obstacle/drone segmentation, depth, and BoQ descriptors. The gate seg, corner, and obstacle heads have **no training data**. GateNet exists as a stub (`perception/detectors/gatenet.py`) with no training harness or dataset.

The AI Grand Prix (Anduril + DCL partnership) uses DCL-style gates (~150cm square, LED-illuminated). Competition gate specs and simulator release in May 2026. We need a training data pipeline that works now with available data and adapts when DCL-specific assets arrive.

## Solution

A 4-stage offline data pipeline producing 60k labeled images from 3 diverse sources (real, semi-synthetic, fully synthetic), stored in a unified format. All processing runs in Docker on desktop `martian` (RTX 3090, 24 threads, 32GB RAM).

## Dataset Format

All 3 sources produce identical per-sample output at **native resolution** (no resizing — DataLoader handles resize/crop/augmentation at train time).

### Per-Sample Structure

```
{source}/{scene_id}_{frame_id}/
  rgb.png              # (H, W, 3) uint8, native resolution
  gate_mask.png        # (H, W) uint8, 0=bg, 1+=gate instance IDs
  corner_heatmaps.npy  # (4, H_h, W_h) float32, CenterNet Gaussian peaks (TL/TR/BR/BL)
  corner_coords.json   # [{gate_id, corners: [[x,y]x4], confidence}]
  obstacle_mask.png    # (H, W) uint8, 0=bg, 1=obstacle, 2=drone
  metadata.json        # {source, resolution, intrinsics, distortion_model, distortion_coeffs, gate_dims_m}
```

### Corner Heatmap Encoding

CenterNet-style: one channel per corner type (TL, TR, BR, BL). Gaussian kernel (sigma = 2px) at output stride 4 centered on each corner's grid cell. Multiple gates produce multiple peaks per channel. Sub-pixel coordinates stored in `corner_coords.json`.

### Camera Model

Images stored at native resolution with camera intrinsics and distortion coefficients in `metadata.json`. Rectification is a DataLoader-time decision — synthetic images are already pinhole, TII images include fisheye distortion parameters. This keeps the dataset camera-agnostic for when competition camera calibration arrives in May.

## Stage 1: TII Dataset (20k Real Images)

**Source**: [tii-racing/drone-racing-dataset](https://github.com/tii-racing/drone-racing-dataset), CC BY 4.0

### What We Get

- ~1M frames at 640x480, 120Hz
- Per-frame labels: `0 cx cy w h tlx tly tlv trx try trv brx bry brv blx bly blv` (YOLO bbox + 4 corner keypoints with COCO visibility flags)
- Up to 7 gates visible per frame
- 3 lighting conditions (high 2480 lx, medium 955 lx, low 216 lx)
- Gates: 152cm square inner opening (close to DCL 150cm)
- Arducam IMX219, 175° diagonal FOV

### Processing Pipeline

1. **Download**: Clone repo, download image archives
2. **Subsample 20k frames**: Stratified by lighting condition and gate count. Skip 0-gate frames. Prefer diverse gate distances and viewing angles
3. **Convert annotations** (at native 640x480):
   - Denormalize corner coords from [0,1] to pixel space
   - Render filled quadrilateral from 4 corners per gate → `gate_mask.png` with instance IDs
   - Generate CenterNet heatmaps from corner coords → `corner_heatmaps.npy`
   - `obstacle_mask.png` = all zeros (no obstacle/drone annotations in TII)
4. **Write metadata**: Camera intrinsics + distortion model from TII calibration files

### Limitations

- No obstacle/drone annotations — obstacle head gets no supervision from this source
- Gate appearance differs from DCL (no LEDs, different frame material)
- Wide-angle lens distortion present (stored in metadata, rectification at load time)

### Dependencies

Python + OpenCV + NumPy. No GPU needed.

## Stage 2: Hybrid Dataset Factory (20k Semi-Synthetic)

**Source**: [M4gicT0/hybrid-dataset-factory](https://github.com/M4gicT0/hybrid-dataset-factory) — OpenGL renders of 3D meshes composited onto real FPV backgrounds.

### Inputs

- **Gate mesh (OBJ)**: DCL-style 150cm square LED-frame gate. Built in Blender — a square frame with emissive LED strips and metallic frame material
- **Drone mesh (OBJ)**: TII drone STL (from their dataset repo) converted to OBJ
- **Background images**: Real FPV frames from TII dataset (0-gate frames skipped in Stage 1) + UZH-FPV frames. Real indoor/outdoor drone-perspective backgrounds

### Domain Randomization Per Render

- **Gate position**: Random distance (2-15m), random bearing, random roll/pitch/yaw
- **Gate count**: 1-3 gates per image
- **Drone count**: 0-2 drones per image (random positions, not overlapping gates)
- **Lighting**: Ambient + emissive variation on gate LEDs
- **Gate LED color**: Randomized (blue, purple, green, white — typical DCL colors)
- **Motion blur**: 5-15px kernel (critical for high-speed realism, per MonoRace findings)
- **Gaussian noise**: Camera sensor simulation

### Outputs (Auto-Generated by Renderer)

- RGB composited image at background's native resolution
- Gate instance mask (rendered from gate mesh silhouette, per-instance IDs)
- Corner coords (known exactly from 3D projection)
- Obstacle/drone mask (class 2 for drone meshes)
- Metadata with camera intrinsics + gate world positions

### Dependencies

Python + ModernGL with EGL backend (headless GPU rendering). RTX 3090.

**Estimated generation speed**: ~50-100 images/sec → 20k in ~5 min.

## Stage 3: BlenderProc (20k Fully Synthetic)

**Source**: [DLR-RM/BlenderProc](https://github.com/DLR-RM/BlenderProc) — ray-traced renders with full scene control.

### Key Difference from Stage 2

Full scene rendering (not compositing). Realistic lighting interactions — shadows, reflections on gate LED frames, ambient occlusion — that compositing cannot produce.

### Inputs

- Same gate + drone OBJ meshes from Stage 2
- **HDRI environments**: PolyHaven indoor HDRIs (warehouses, gyms, hangars, arenas — DCL-venue-like). Free, CC0 licensed. ~20-30 diverse HDRIs

### Scene Composition Per Render

- Random HDRI background + lighting
- 1-3 gates at random positions/orientations (2-15m from camera)
- 0-2 drones at random positions
- Camera with randomized intrinsics (FOV 90-120°, matching wide-angle FPV cameras)
- Randomize gate material properties (LED emissive intensity, frame metallic/roughness)

### Domain Randomization

- HDRI rotation (effectively infinite backgrounds from ~30 HDRIs)
- Gate LED color/intensity
- Camera exposure, white balance
- Motion blur (vector blur compositing node)
- Chromatic aberration, lens distortion

### Outputs

- RGB render at configurable resolution (~640x480 to match TII scale)
- Instance segmentation mask (gate IDs + drone class)
- Depth map (free bonus — useful for depth head fine-tuning later)
- Corner coords (computed from gate mesh vertex positions + camera projection)
- Metadata with full camera + scene parameters

### Rendering Strategy

- **Eevee** (rasterized) for bulk: ~0.3-0.5 sec/image → 20k in ~2-3 hours
- **Cycles** (ray-traced) for a high-fidelity subset (~2-3k): ~2-5 sec/image
- Mixed dataset gives fast generation + photorealistic diversity

### Dependencies

`blenderproc` (pip-installable, runs Blender headless). GPU for rendering.

## Stage 4: Merge + Validation

### Merge

All 3 sources write to a shared directory. A global manifest indexes everything:

**`manifest.parquet` schema:**
```
sample_id: string
source: string          # "tii" | "hybrid" | "blenderproc"
scene_id: string
frame_id: string
n_gates: int32
n_drones: int32
resolution_h: int32
resolution_w: int32
has_distortion: bool
```

**Train/val split**: 90/10, stratified by source + gate count. Written to `splits/train.txt` and `splits/val.txt`.

### Validation Checks

1. **Visual sanity**: Render 50 random samples per source — overlay masks + corners on RGB
2. **Distribution stats**: Gate count histogram, gate pixel area distribution, corner spread across image, lighting diversity
3. **Cross-check**: Verify corners land on mask edges (within 2px tolerance)

## Storage Layout

```
~/corvidx/data/
  raw/
    tii/                        # Raw TII download
    uzh-fpv/                    # Already exists (from COR-60)
  gate_detection/
    tii/                        # 20k converted samples
    hybrid/                     # 20k semi-synthetic samples
    blenderproc/                # 20k fully synthetic samples
    manifest.parquet
    splits/
      train.txt                 # ~54k sample IDs
      val.txt                   # ~6k sample IDs
```

**Estimated disk**: ~20-30GB total.

## Code Structure

All code under `perception/training/` to match COR-60 pattern:

```
perception/training/
  data/
    gate_detection/
      tii_converter.py          # Download + subsample + convert TII
      hybrid_factory.py         # Wrapper around hybrid-dataset-factory
      blenderproc_pipeline.py   # BlenderProc scene generation script
      merge.py                  # Merge sources, build manifest + splits
      validate.py               # Visual sanity checks + distribution stats
      format.py                 # Shared format constants + I/O helpers
      assets/
        gate.blend              # DCL-style gate model (Blender source)
        gate.obj                # Exported OBJ for renderers
        drone.obj               # TII drone mesh converted from STL
  Dockerfile.gate-data          # CUDA + Blender headless + ModernGL
```

### CLI Entry Points

```bash
# Stage 1: TII conversion
python -m perception.training.data.gate_detection.tii_converter \
  --data-dir ~/corvidx/data/raw/tii \
  --output-dir ~/corvidx/data/gate_detection/tii \
  --n-samples 20000

# Stage 2: Hybrid factory
python -m perception.training.data.gate_detection.hybrid_factory \
  --backgrounds ~/corvidx/data/raw/tii \
  --output-dir ~/corvidx/data/gate_detection/hybrid \
  --n-samples 20000

# Stage 3: BlenderProc
python -m perception.training.data.gate_detection.blenderproc_pipeline \
  --output-dir ~/corvidx/data/gate_detection/blenderproc \
  --n-samples 20000

# Stage 4: Merge + validate
python -m perception.training.data.gate_detection.merge \
  --data-dir ~/corvidx/data/gate_detection \
  --val-ratio 0.1

python -m perception.training.data.gate_detection.validate \
  --data-dir ~/corvidx/data/gate_detection \
  --n-vis 50
```

## Docker

Single Dockerfile (`Dockerfile.gate-data`) based on `nvidia/cuda:12.1-runtime`:
- Python 3.11
- Blender 4.x (headless) + BlenderProc
- ModernGL + PyOpenGL + EGL
- OpenCV, NumPy, PyArrow
- No conda (per project convention)

## Infrastructure

- **Processing**: Desktop `martian` (RTX 3090 24GB, 24 threads, 32GB RAM, 524GB NVMe free)
- **SSH**: `ssh martian` from MacBook

## Gate Model

A DCL-style square LED gate, modeled in Blender:
- 150cm x 150cm inner opening (matching A2RL/DCL spec)
- Square frame (~10cm wide) with metallic material
- LED strip geometry on inner frame edge with emissive material
- Configurable LED color via material parameter
- Exported as OBJ for renderers, source .blend kept in assets/

The TII drone STL is converted to OBJ and used as the obstacle mesh.

## Key References

| Resource | URL | Role |
|----------|-----|------|
| TII Race Against the Machine | https://github.com/tii-racing/drone-racing-dataset | Real training data |
| Hybrid Dataset Factory | https://github.com/M4gicT0/hybrid-dataset-factory | Semi-synthetic renderer |
| BlenderProc | https://github.com/DLR-RM/BlenderProc | Fully synthetic renderer |
| PolyHaven HDRIs | https://polyhaven.com/hdris | Indoor environments |
| MonoRace (arXiv:2601.15222) | https://arxiv.org/abs/2601.15222 | Gate detection baseline |
| CenterNet | https://github.com/xingyizhou/CenterNet | Corner heatmap encoding |
| Simple Copy-Paste (Ghiasi et al.) | https://arxiv.org/abs/2012.07177 | Augmentation reference |
| Deep Drone Racing Sim-to-Real | https://arxiv.org/abs/1905.09727 | Domain randomization approach |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TII gate appearance too different from DCL | Trained model doesn't generalize | Synthetic sources (Stage 2+3) use DCL-style gate model; real data adds noise robustness |
| Hybrid factory compositing artifacts | Unrealistic training data | BlenderProc (Stage 3) provides physically-correct renders as complement |
| DCL gate specs change from 150cm assumption | Wrong gate dimensions baked in | Gate dims are a config parameter, meshes easily rebuilt; format stores `gate_dims_m` per sample |
| BlenderProc rendering too slow | Can't generate 20k in time | Use Eevee for bulk (~3h), Cycles only for 2-3k high-fidelity subset |
| Competition camera has extreme distortion | Model trained on mild distortion fails | Store raw + distortion metadata; DataLoader applies rectification; retrain when calibration available |
| Obstacle/drone class underrepresented in TII source | Obstacle head poorly trained | Stages 2+3 render drones in every scene; 40k/60k samples have drone annotations |

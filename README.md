# msplat

A 3D Gaussian Splatting training engine for Apple Silicon, built entirely on Metal. No external dependencies beyond system frameworks.

The entire training pipeline: projection, sorting, rasterization, SSIM loss, backward pass, Adam optimizer, and densification runs as fused Metal compute shaders.

The result is a self-contained engine that trains a full-resolution Mip-NeRF 360 scene in ~70 seconds and renders it at ~350 FPS on an M4 Max.

Python and Swift bindings are provided, as well as a standalone C++ CLI.

<div align="center">
  <video src="https://github.com/user-attachments/assets/cb942a38-cf6a-4b06-9899-675396550c57" />
</div>

## Why this exists

The original [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) implementation is CUDA-only. Ports to other frameworks (gsplat, taichi-3dgs, etc.) still depend on PyTorch for autograd, optimizer state, and tensor management. This means ~2GB of framework overhead, Python GIL contention, and no straightforward path to native macOS/iOS integration.

## Architecture

```
core/metal/msplat_metal.metal    ← Compute kernels
core/src/                        ← C++ training loop, dataset loaders, SSIM eval
core/include/                    ← MTensor (lightweight GPU tensor), Model, API headers
python/bindings.cpp              ← nanobind Python module
swift/Sources/Msplat/            ← Swift package (via C API bridge)
cli/msplat.cpp                   ← C++ CLI
```

### Training pipeline (single iteration)

Each training step dispatches all work into one Metal command encoder:

```
Forward:
  project_and_sh_forward     ← fused 3D→2D projection + spherical harmonics
  prefix_sum + scatter       ← gaussian→tile intersection mapping
  bitonic_sort_per_tile      ← tile-local depth sort + inline data packing
  nd_rasterize_forward       ← per-pixel alpha compositing (16x16 tiles)
  ssim_h_fwd + ssim_v_fwd   ← separable 11-tap SSIM + L1 loss

Backward:
  ssim_h_bwd + ssim_v_bwd   ← separable SSIM gradient
  rasterize_backward         ← per-pixel backward compositing
  project_and_sh_backward    ← fused projection + SH VJP + SH Adam update
  fused_adam (×4 groups)     ← optimizer step (means, scales, quats, opacity)
  accumulate_grad_stats      ← gradient norms for densification
```

### Key design decisions

**Tile-local bitonic sort** instead of global radix sort. Each 16x16 tile independently sorts its gaussians (up to 4096) in threadgroup shared memory. The sort kernel also packs per-gaussian data (xy, opacity, conic, color) inline, eliminating a separate scatter dispatch.

**GPU-resident densification.** The split/clone/cull cycle never leaves the GPU. Classification, growth, and compaction are all compute kernels operating on device buffers. No CPU readback of gradient statistics or gaussian counts.

**Fused kernels.** Projection and spherical harmonic evaluation share registers (avoid a device memory round-trip for world-space position). The backward pass recomputes 3D covariance from scales/quaternions on-the-fly rather than storing it. SH backward gradients are computed in registers and fed directly into Adam updates, eliminating a separate gradient buffer write/read cycle. The remaining four parameter groups use fused Adam dispatches.

**Separable SSIM.** The 11x11 Gaussian-weighted SSIM window decomposes into two 1D passes (horizontal then vertical), reducing per-pixel work from 121 to 22 multiply-adds. Forward and backward each take two kernels, using threadgroup shared memory for the intermediate statistics.

**Depth-chunked rasterization.** For tiles with extreme gaussian counts, the forward pass splits into 512-gaussian chunks with a merge kernel that reconstructs absolute transmittance. The backward pass uses precomputed prefix/suffix transmittance to avoid re-traversal.

## Installation & Usage

### Python

```bash
pip install msplat
```

```python
import msplat

dataset = msplat.load_dataset("path/to/colmap/", eval_mode=True)
config = msplat.TrainingConfig(iterations=7000, num_downscales=0)
trainer = msplat.GaussianTrainer(dataset, config)

trainer.train(lambda s: print(f"step={s.iteration} splats={s.splat_count:,}"),
              callback_every=100)

trainer.export_ply("output.ply")
trainer.save_checkpoint("checkpoint.msplat")  # save/resume training
metrics = trainer.evaluate()
print(f"PSNR: {metrics['psnr']:.2f}  SSIM: {metrics['ssim']:.3f}")

# Render from arbitrary viewpoints
pose = dataset.camera_pose(0)   # (4, 4) cam-to-world matrix
img = trainer.render_from_pose(pose)  # numpy (H, W, 3) float32
```

Supported dataset formats: COLMAP, Nerfstudio, Polycam.
Evaluation metrics follow Brush's convention by applying an 8-bit roundtrip to
the rendered RGB before PSNR, SSIM, and L1 are computed.
Transparent PNG targets are composited against the configured background color
(`bg_color` / `--bg-color`, default black) during training. Training also uses Brush-style
background jitter (`background_noise_strength` / `--background-noise-strength`)
and an alpha L1 term controlled by `match_alpha_weight` / `--match-alpha-weight`.
Brush's optional LPIPS loss is available through `lpips_loss_weight` /
`--lpips-loss-weight` using vendored VGG weights and the native Metal backend.
Brush-style opacity and scale shrink are applied at refinement steps via
`opac_decay` / `--opac-decay` and `scale_decay` / `--scale-decay`.
Visible low-opacity splats also receive Brush-style mean noise controlled by
`mean_noise_weight` / `--mean-noise-weight`.
Datasets without an input point cloud fall back to Brush-style random splats in
camera frustums; `random_init_scene_scale` / `--random-init-scene-scale`
overrides the estimated scene scale.
Splat growth stops at `growth_stop_iter` / `--growth-stop-iter`, defaulting to
Brush's 15,000-step cutoff instead of scaling with total iteration count.
Growth is also bounded by `max_splats` / `--max-splats` and sampled by
`growth_select_fraction` / `--growth-select-fraction`.
Brush-style optimizer knobs are exposed as `lr_mean`, `lr_mean_end`,
`lr_scale`, `lr_scale_end`, `lr_rotation`, `lr_coeffs_dc`,
`lr_coeffs_sh_scale`, and `lr_opac` with msplat's tuned defaults.
`reduce_second_moment` / `--reduce-second-moment` matches Brush's scalar
second-moment Adam math for SH coefficients.
Nerfstudio `mask_path` frames and sibling `masks/<image-stem>.*` files are used
as loss masks.
MIP splatting opacity compensation is available with `render_mip=True` in
Python or `--render-mip` in the C++ CLI.
The C++ CLI exports PLYs and `cameras.json` in the input dataset coordinate
frame by default, matching Brush. Use `--normalize-crs` to write msplat's
normalized internal coordinate frame instead.
Training can also emit Brush-style LOD PLYs with `--lod-levels`; each level
keeps `--lod-keep-ratio` or `--lod-decimation-keep` of the previous splat set.
LOD decimation uses PUP sensitivity scores accumulated from per-view
`[d_mean, d_log_scale]` gradients, then optionally optimizes each decimated
LOD with `--lod-refine-steps` and Brush's cumulative `--lod-image-scale`.

Type stubs (`_core.pyi`) are included for IDE autocompletion.

#### CLI

```bash
pip install msplat[cli]
msplat-train path/to/dataset -n 7000 --eval
```

If the dataset directory contains `args.txt`, the C++ CLI loads it as
Brush-style whitespace-separated options before parsing the command line.
Explicit CLI options override matching `args.txt` options.

### Swift

Requires Xcode and CMake (`brew install cmake`).

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/rayanht/msplat.git", from: "1.1.0")
]
```

Build the XCFramework (one-time, from repo root):

```bash
./scripts/build-xcframework.sh
```

```swift
import Msplat

let dataset = GaussianDataset(path: "path/to/colmap/", downscaleFactor: 4.0)
let trainer = GaussianTrainer(dataset: dataset)

for _ in 0..<1000 {
    let stats = trainer.step()
    print("step=\(stats.iteration) splats=\(stats.splatCount)")
}

trainer.exportPly(to: "output.ply")

// Render from arbitrary viewpoints
let pose = dataset.cameraPose(at: 0)  // [Float] cam-to-world matrix
let img = trainer.renderFromPose(camToWorld: pose)
```

### C++ CLI

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/msplat path/to/dataset -n 7000 --eval
```

### Build from source

```bash
git clone https://github.com/rayanht/msplat.git && cd msplat

# Python
pip install -e .

# C++ CLI + static lib
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Swift XCFramework
./scripts/build-xcframework.sh
cd swift && swift build
```

Requires macOS 14+, Apple Silicon. No external dependencies.

## Benchmarks

mipnerf360, M4 Max. msplat runs 7K iterations with no downscales:

```bash
msplat-train path/to/scene -n 7000 --num-downscales 0 --eval
```

| Scene | msplat PSNR | msplat SSIM | msplat wall time | gsplat PSNR | gsplat SSIM | gsplat wall time
|-------|-------------|-------------|-----------|-------------|-------------|-------------|
| bicycle | 23.23 | 0.602 | 59s | 23.71 | 0.668 | ~335s
| counter | 27.45 | 0.880 | 80s | 27.14 | 0.878 | ~335s
| garden | 25.68 | 0.783 | 77s | 26.30 | 0.833 | ~335s
| room | 30.12 | 0.897 | 74s | 29.21 | 0.893 | ~335s

### 30K iterations (garden)

```bash
msplat-train path/to/garden -n 30000 --num-downscales 0 --eval
```

| | msplat | gsplat |
|---|---|---|
| PSNR | 27.14 | 27.32 |
| SSIM | 0.853 | 0.865 |
| Gaussians | 3.51M | — |
| Wall time | 700s | ~2149s |

gsplat numbers from [docs.gsplat.studio](https://docs.gsplat.studio/main/tests/eval.html) (TITAN RTX). gsplat wall times are the reported average across *all* mipnerf360 scenes (per-scene times not published).

### Performance history (wall time, M4 Max)

| Scene | v1.0 | v1.1.3 | Speedup |
|-------|------|--------|---------|
| bicycle 7K | 82s | 59s | 1.39x |
| counter 7K | 91s | 80s | 1.14x |
| garden 7K | 107s | 77s | 1.39x |
| room 7K | 85s | 74s | 1.15x |
| garden 30K | 1039s | 700s | 1.48x |

v1.1.3 fuses SH backward gradients into Adam optimizer updates, fuses the SSIM vertical-forward and horizontal-backward passes into a single kernel, and replaces the count→prefix-sum→scatter intersection pipeline with pre-allocated per-tile bins. Speedup scales with gaussian count.

## License

Apache 2.0

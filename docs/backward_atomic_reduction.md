# Backward atomic reduction design

The per-pixel backward rasterizer reduces each gaussian contribution inside a
SIMD group, then lane 0 issues gradient atomics for RGB, conic, xy, opacity, and
refine. This keeps the pixel traversal cheap, but the same gaussian is flushed
once per active pixel warp. The debug counter reports this as
`pixel atomic estimate`.

The per-splat backward rasterizer is the existing tile-level reduction path. A
SIMD lane owns one splat, walks the tile pixels with diagonal replay, accumulates
the full tile gradient in registers, then flushes one atomic bundle for that
splat. This removes the pixel atomic storm, but it can lose when tile replay
dominates. A fixed-splat 1024px garden A/B showed the per-pixel path still ahead
for this mid-resolution case, so the current auto threshold should stay
conservative.

## Baseline observation

A fixed-splat 1024px garden run with stage profiling and backward debug counters
reported the following shape:

| mode | rast_bwd median | pixel atomic groups | tile merge floor | replay active | replay diagonal | saturated pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pixel | 2.389 ms | 0.8M/sample | 0.1M/sample | 27.1M/sample | 0.2M/sample | 2071.7/sample |
| persplat | 4.894 ms | 0.8M/sample | 0.1M/sample | 27.1M/sample | 0.2M/sample | 2070.8/sample |

The pixel path has an apparent 87% cross-warp merge ceiling, but the existing
per-splat replay path is still slower on this case. That means a new reduction
path must prove it can capture some of the atomic reduction without paying the
full replay cost.

## Candidate paths

1. Keep the current pixel/per-splat split as the production baseline. It already
   has explicit benchmark controls and a conservative auto threshold.

2. Add a debug-only decision model before adding new kernels. The useful inputs
   are already available: post-prune tile splat count, pixel warp atomic groups,
   tile-level merge floor, replay active pairs, diagonal replay steps, and
   saturated pixels. A candidate heuristic should be validated against stage
   medians, not only end-to-end throughput.

3. Prototype a small-batch threadgroup merge only as an opt-in benchmark hook.
   A full 256-splat, 8-warp accumulator would require roughly 80 KB for ten
   float channels before existing batch storage, so it is not viable. A 32-splat
   subbatch needs around 10 KB and could reduce atomics across warps, but it adds
   more barriers and repeated subbatch passes. Earlier subtile merge experiments
   regressed, so this should not become production code without a clear A/B win.

4. Consider a two-pass contribution compaction only for large/full-resolution
   scenes. Emitting per-tile or per-gaussian contribution records and segmented
   reducing them would attack global atomics directly, but it adds memory
   bandwidth, temporary storage, and either sorting or prefix/segment work. This
   is only plausible if debug counters show atomics dominate and per-splat replay
   remains too expensive.

## Benchmark gate

Any new reduction path should be behind an explicit environment flag and
benchmarked with:

```bash
python3 scripts/benchmark_backward_rasterizers.py DATASET \
  --binary build/msplat \
  --iters 600 \
  --modes pixel persplat \
  --quality-metrics \
  --debug \
  --profile-stages \
  -- \
  --save-every -1 \
  --quality default \
  --max-splats 500000 \
  --max-resolution 1024 \
  --sh-degree 3 \
  --with-viewer false \
  --no-log-image-loading
```

The minimum promotion bar is a lower `rast_bwd` median and no final-quality
regression on garden, a masked object dataset, and an indoor scene. If the path
only wins under profiling or only changes end-to-end throughput while
`rast_bwd` regresses, keep it as a benchmark hook.

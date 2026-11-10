# Phase 0 pitch visuals: how to reproduce

Everything here is prototype code for the 2026-11-10 pitch. Reusable pieces live in
`src/tractwise/viz/` (`colors.py`, `render.py`).

## Commands

Run from the repo root, in order:

```bash
uv sync
uv run python scripts/phase0_track.py     # ~1 min  -> outputs/phase0/stanford_det.trk (289,375 streamlines), gfa.nii.gz
uv run python scripts/phase0_render.py    # ~1 min  -> outputs/phase0/tractogram_{axial,sagittal,oblique}.png (2400x2400)
uv run python scripts/phase0_comet.py --gif  # ~several min on CPU -> outputs/phase0/comet.mp4 (+ comet.gif)
```

All randomness is seeded (`--seed 0` is the default everywhere). Every parameter is
a CLI flag; see `--help`.

## Pipeline

- **Data:** DIPY Stanford HARDI (`get_fnames(name="stanford_hardi")`, 150 directions,
  b=2000) and `stanford_labels`. The data is cached in `~/.dipy/stanford_hardi`.
- **Tracking:** CSA ODF (`sh_order_max=6`) → `peaks_from_model` (relative threshold 0.8,
  min separation 45°) → `dipy.tracking.tracker.deterministic_tracking`.
  - Stopping criterion: GFA ≥ 0.25.
  - Seeds: the full white-matter mask (labels 1 and 2), density 2.
  - Step 0.5 mm, max angle 30°.
  - Streamlines shorter than 30 mm are dropped.
- **Static render:** 150k random streamlines, compressed with
  `compress_streamlines(tol_error=0.1)`. Compression keeps the vertex buffer under
  wgpu's 128 MB binding limit.
  - Colors are per-vertex |tangent|: R = L-R, G = A-P, B = S-I.
  - Opacity 0.6, zoom 1.2.
- **Comet prototype:** 3,000 random streamlines of at least 40 mm, resampled to 100 points.
  - Colors come from a QuickBundles clustering (20 mm threshold, about 99 clusters), as a stand-in for bundle labels.
  - There are two layers: a dim static copy, and a comet copy with additive blending whose per-vertex
    colors are rewritten every frame. Head speed is 40 mm/s, the tail is 20 mm, and each comet has a random phase and direction.
  - The camera does a half-turn turntable over 8 s at 30 fps, 1280x720, encoded with libx264
    via `imageio-ffmpeg`.

The motion is a visualization device. Diffusion MRI gives fiber *orientation*, not direction.

## Environment used

- Python 3.13, dipy 1.12.1, fury 2.1.0 (pygfx 0.17.0 / wgpu 0.33.0), numpy 2.5.4,
  nibabel 5.5.0, imageio 2.38.1, imageio-ffmpeg 0.8.1.
- Rendering is headless and offscreen with FURY 2.x via wgpu, so no X server or Xvfb is needed.
- On 2026-11-10 the NVIDIA kernel module was not loaded (missing for kernel
  7.0.0-42), so wgpu used **llvmpipe (CPU) via Vulkan**. Renders will be much faster
  once `linux-modules-nvidia-595-open-$(uname -r)` is installed.

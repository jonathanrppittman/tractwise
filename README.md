# tractwise

Uncertainty-aware classification of tractography streamlines into white matter
bundles. A deep ensemble of point-cloud classifiers (PointNet/DGCNN) labels each
streamline and flags the ones it is unsure about. The ensemble is compared against
DIPY's RecoBundles and visualized as an animated "comet" rendering.

Final project for *Open Source Programming in Medical Image Analysis* (CS 6357).
**Status: early work in progress.** Only the Phase 0 prototype visuals exist so far.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/phase0_track.py    # whole-brain tractography on DIPY's Stanford HARDI
uv run python scripts/phase0_render.py   # directional-RGB PNG renders
uv run python scripts/phase0_comet.py --gif  # comet animation prototype
uv run pytest
```

Rendering uses FURY 2.x (pygfx/wgpu) offscreen, so it runs headless with no X server.
See [`scripts/README.md`](scripts/README.md) for exact parameters and versions.

Note: diffusion MRI gives fiber *orientation*, not direction. The animated motion is
a visualization device, not signal flow.

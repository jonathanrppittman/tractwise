# Project Brief: Uncertainty-Aware Streamline Bundle Classification

**For Claude Code (plan mode):** Read this whole brief, then propose a plan before writing any code. Ask me clarifying questions where something is ambiguous. Items marked **[VERIFY]** are assumptions you should check (docs, package versions, dataset pages) rather than trust.

## Context

This is my final project for a graduate course, *Open Source Programming in Medical Image Analysis*. I have two deadlines:

1. **Today:** a ~5-minute verbal pitch of the project. I need one or two strong visuals to show (see Phase 0).
2. **~4 weeks from now:** the finished project — code, results, animation, and writeup.

The course emphasizes open-source engineering quality, so packaging, testing, and documentation matter as much as the model.

**Compute:** two strong GPUs on my local lab machine, plus access to a lab server with more GPUs if needed. Rendering may need to run headless.

## Project Summary

Tractography produces millions of streamlines per brain that must be grouped into named white matter bundles (e.g., corticospinal tract, arcuate fasciculus). Atlas-based methods like DIPY's RecoBundles work but give no indication of when they are wrong, and they struggle in crossing-fiber regions.

I'm building:

- A **deep learning classifier** that labels each streamline individually. Streamlines are resampled to a fixed number of 3D points (32 or 64) and treated as point clouds (PointNet or DGCNN-style architecture), with an extra "other/implausible" class.
- A **deep ensemble** (~5 members, different seeds) whose disagreement gives a per-streamline **uncertainty score** (predictive entropy and/or member disagreement).
- An **evaluation** against RecoBundles, plus an analysis of whether high-uncertainty streamlines are disproportionately the misclassified ones.
- An **animated 3D rendering**: bright "comet" segments travel along each streamline with random phase offsets, colored by predicted bundle, with brightness/opacity encoding confidence (uncertain streamlines dim or flicker).
- A **uv-managed Python package** with a CLI, tests, CI, and docs.

Note for docs/writeup: diffusion MRI gives fiber orientation, not direction, so the animated motion is a visualization device, not signal flow.

## Tech Stack

- Python, managed with **uv** (`pyproject.toml` + `uv.lock`). Do not use pip/requirements.txt as the primary workflow; the package should still be pip-installable via the standard `pyproject.toml`.
- **DIPY** (tractography, RecoBundles, data fetchers, streamline utilities)
- **FURY** (3D rendering and keyframe/timeline animation, export to MP4/GIF) **[VERIFY]** current FURY animation API and how to render offscreen
- **PyTorch** (models, multi-GPU or parallel ensemble training)
- nibabel for I/O (`.trk`/`.tck`, NIfTI)
- pytest, ruff, GitHub Actions CI
- Working package name: `tractwise` (placeholder — suggest alternatives if you like)

## Phase 0 — TODAY: Pitch Visuals (do this first, keep it small)

Goal: within an hour or two, produce visuals I can show during a 5-minute talk. This is throwaway-quality code living in `scripts/` or `demo/`, but it should seed the real project structure.

1. Set up the uv project skeleton (`uv init`, add dipy, fury, nibabel, numpy).
2. Using a DIPY built-in fetcher (e.g., Stanford HARDI) **[VERIFY]** fetcher name and API:
   - Fit a simple model (CSA or CSD), run deterministic or probabilistic local tracking from a white matter mask.
   - Render the whole-brain tractogram colored by standard directional RGB (red = L-R, green = A-P, blue = S-I) and save a high-resolution PNG.
3. Stretch: a short prototype of the comet animation on a subset of streamlines (a few thousand), exported as MP4 or GIF. Since bundle labels aren't available yet, color by direction or by a quick clustering (e.g., DIPY QuickBundles) as a stand-in.
4. Handle headless rendering if needed (offscreen VTK, xvfb-run, or EGL) **[VERIFY]** what works on my machine.

Keep a note of exactly which commands produced the visuals so they're reproducible.

## Phase 1 — Week 1: Data Pipeline and Baseline

- Download and organize **TractoInferno** from OpenNeuro **[VERIFY]** dataset ID, size, license, file layout, number of subjects, which bundles are provided, and whether the official train/val/test split is included.
- Build a preprocessing step that loads bundle tractograms, resamples each streamline to N points, builds per-streamline labels, and caches to an efficient format (e.g., NumPy memmap, HDF5, or Zarr) for fast training. This is the most likely bottleneck — design it carefully.
- Define how "other/implausible" training examples are generated (e.g., streamlines from whole-brain tractography not belonging to any curated bundle) **[VERIFY]** what the dataset provides.
- Get **RecoBundles** running as a baseline on the test split, with an atlas **[VERIFY]** which atlas/bundle templates are compatible.

## Phase 2 — Week 2: Single Model

- Implement a PointNet-style baseline first; DGCNN as an optional upgrade.
- Coordinate handling: decide on space (MNI vs. native) and normalization; consider flip augmentation (streamlines have no inherent direction — make the model invariant to point order reversal).
- Training loop with config files, checkpointing, logging (TensorBoard or similar), seeded runs.
- Metrics: per-bundle streamline-level precision/recall/F1, plus voxel-level Dice on bundle masks.

## Phase 3 — Week 3: Ensemble and Uncertainty

- Train ~5 ensemble members, in parallel across GPUs where possible.
- Per-streamline uncertainty: predictive entropy, mutual information / member disagreement.
- Analysis: calibration (reliability diagram, ECE), error rate vs. uncertainty curves, and spatial analysis of where uncertainty is high (hypothesis: crossing-fiber regions).
- Head-to-head comparison table vs. RecoBundles.

## Phase 4 — Week 4: Animation, Packaging, Writeup

- Final animation: comet segments colored by predicted bundle, opacity/brightness by confidence, MP4 export, configurable via CLI.
- CLI commands roughly like: `preprocess`, `train`, `predict`, `evaluate`, `animate`.
- Tests: unit tests for resampling, label building, model forward pass shapes, uncertainty math; a small synthetic-data smoke test that runs in CI without GPUs or big downloads.
- GitHub Actions CI using uv (lint, tests).
- Docs: README with the animation front and center, install via uv, quickstart, results summary. Optional Dockerfile.

## Visuals Plan

Strong visuals are a priority for this project. Design the code so visuals are cheap to produce (e.g., save predictions and per-streamline uncertainty in a standard format after inference so any renderer can consume them).

**Core (must have):**

1. **Hero animation:** comet segments colored by predicted bundle, brightness/opacity by confidence, with a slow turntable camera rotation. MP4 export.
2. **Uncertainty map on anatomy:** project per-streamline uncertainty into a voxel map (e.g., mean entropy of streamlines passing through each voxel), overlaid on FA slices (axial/coronal/sagittal) and optionally as a 3D glass-brain render. Expected to highlight crossing-fiber regions — this is the key scientific figure.
3. **Side-by-side comparison:** ground truth vs. RecoBundles vs. our model, rendered with a synchronized camera, with disagreements highlighted.

**Stretch (in priority order):**

4. **"Chaos to order" training video:** save predictions on one fixed held-out subject after every epoch (subsample streamlines to keep it light), then animate the sequence. Build the per-epoch saving hook into training early, since it can't be added retroactively.
5. **Ensemble disagreement flicker:** for high-uncertainty streamlines, cycle their color through each ensemble member's predicted label over time.
6. **Embedding explorer:** UMAP of learned streamline embeddings colored by bundle, with uncertainty as point size/opacity; ideally an interactive browser viewer linking embedding points to 3D streamlines, suitable for the README or GitHub Pages **[VERIFY]** good options for browser-based tractogram viewing.

All visuals should be reproducible via the CLI (e.g., `tractwise animate`, `tractwise figures`) with config files, not one-off notebooks.

## Repository Layout (suggested — propose changes if better)

```
tractwise/
  pyproject.toml
  uv.lock
  README.md
  src/tractwise/
    data/        # loading, resampling, caching, datasets
    models/      # pointnet, dgcnn
    train.py
    predict.py
    uncertainty.py
    evaluate.py  # metrics, RecoBundles baseline
    viz/         # static renders, comet animation
    cli.py
  configs/
  scripts/       # one-off scripts, incl. Phase 0 demo
  tests/
  .github/workflows/ci.yml
```

## Priorities and Risk Management

- Get one small model trained end to end before building the ensemble or polishing visualization.
- GPU time is not the constraint; data wrangling is. Front-load it.
- If time gets tight, scale down the animation first, then DGCNN, then ensemble size. Keep RecoBundles comparison and uncertainty analysis.
- Don't download the full dataset until the pipeline works on a handful of subjects.

## What I Want From Plan Mode

1. A concrete plan for **Phase 0** I can execute today, with time estimates.
2. A higher-level plan for Phases 1–4 with milestones and checkpoints.
3. A list of the **[VERIFY]** items you checked and what you found, plus any open questions for me.
